import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from typing import List, Dict, Generator, Optional
from glob import glob
import itertools

# ==============================================================================
# 1. YOUR CUSTOM DATASET (with the chunking optimization)
#    This class acts as the "sampler" or "meta-dataset".
# ==============================================================================
class StatefulShardedDataset(IterableDataset):
    """
    A self-contained, high-performance IterableDataset that reads from sharded
    .npy files and performs dynamic, weighted sampling across different data domains.
    Designed to be robust with multiprocessing (num_workers > 0).
    """

    def __init__(
        self,
        domain_dirs: Dict[str, str],
        initial_weights: Optional[List[float]] = None,
        chunk_size: int = 1,
        seed: int = 42,
    ):
        super().__init__()
        # 1. Store configurations - these are simple and safe to pickle
        self.domain_names  = list(domain_dirs)
        self.domain_shards : List[List[str]] = [
            glob(os.path.join(path, "*.npy")) for path in domain_dirs.values()
        ]
        self.chunk_size = chunk_size
        self.seed = seed

        # 2. Create the shared weights array here. It's designed to be safely
        #    pickled and shared with worker processes.
        if initial_weights:
            assert len(initial_weights) == len(self.domain_names)
            weights = initial_weights
        else:
            weights = [1.0 / len(self.domain_names)] * len(self.domain_names)
        
        # Use a buffer for shared memory, which is more reliable.
        self.weights_buffer = torch.zeros(len(weights), dtype=torch.double).share_memory_()
        self.update_weights(weights)

        print(f"Initialized dataset with {len(self.domain_names)} domains.")
        for name, shards in zip(self.domain_names, self.domain_shards):
            print(f"  -> Found {len(shards)} shards for domain '{name}'")

    def update_weights(self, new_weights: list[float]):
        """Updates the shared weights array from the main process."""
        with torch.no_grad():
            self.weights_buffer[:] = torch.tensor(new_weights, dtype=torch.double)

    def _get_worker_info(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    def __iter__(self):
        worker_id, num_workers = self._get_worker_info()
        
        # Each worker gets its own random generator for deterministic shuffling
        g = torch.Generator()
        # Seed each worker differently but deterministically
        g.manual_seed(self.seed + worker_id)

        # Create worker-specific shard lists by splitting the domain shards
        # across workers. This ensures each worker gets a unique subset of shards
        # for each domain, which is crucial for reproducibility.
        worker_domain_shards = [domain_shard_list[worker_id::num_workers] 
                                for domain_shard_list in self.domain_shards]

        # Create a generator for each domain for this worker
        domain_gens = [self._create_domain_generator(shards, g) 
                       for shards in worker_domain_shards]

        while True:
            # Use the shared weights buffer
            probs = self.weights_buffer / self.weights_buffer.sum()
            
            # Sample a chunk of domains to draw from
            domain_indices = torch.multinomial(
                probs, self.chunk_size, replacement=True, generator=g
            )

            for domain_idx in domain_indices:
                try:
                    yield next(domain_gens[domain_idx])
                except StopIteration:
                    # This worker has exhausted its shards for this domain,
                    # so we recreate its generator to loop over its shards again.
                    domain_gens[domain_idx] = self._create_domain_generator(
                        worker_domain_shards[domain_idx], g
                    )
                    # Yield a sample from the newly created generator
                    try: 
                        yield next(domain_gens[domain_idx])
                    except StopIteration:
                        # This happens if a worker has NO shards for the selected
                        # domain (too many num_workers for few shards).
                        # In this case, we skip yielding from this domain.
                        continue


    def _create_domain_generator(self, shard_list: List[str], generator: torch.Generator):
        # 1. Shuffle shard list for this epoch
        if not shard_list:
            return        
        indices = torch.randperm(len(shard_list), generator=generator).tolist()
        shuffled_shard_list = [shard_list[i] for i in indices]

        for shard_path in shuffled_shard_list:
            # 2. Load and shuffle samples within the shard
            data = np.load(shard_path)
            
            # Use torch generator for shuffling for better reproducibility
            perm = torch.randperm(len(data), generator=generator)
            for idx in perm:
                yield {
                    "input_ids": torch.from_numpy(data[idx]).long(),
                    # "set": os.path.basename(shard_path)  # Keep track of the shard
                }

# ==============================================================================


class EvaluationShardedDataset(IterableDataset):
    """
    IterableDataset for evaluation: 
    - visits each .npy shard exactly once, in a reproducible, shuffled order
    - correctly splits shards across DataLoader workers
    - uses only PyTorch RNG for determinism
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir     = data_dir
        # Gather and sort all shards
        self.shard_files = sorted(glob(os.path.join(self.data_dir, '*.npy')))
        if not self.shard_files:
            raise FileNotFoundError(f"No .npy shards in {self.data_dir!r}")

    def __iter__(self):
        # 1) Deterministically shuffle shard *indices* once per epoch
        seed = torch.initial_seed() % (2**32)
        shuf_gen = torch.Generator().manual_seed(seed)
        all_idxs = torch.randperm(len(self.shard_files), generator=shuf_gen).tolist()

        # 2) Split among workers
        worker = get_worker_info()
        if worker is None:
            my_idxs = all_idxs
        else:
            wid, nw = worker.id, worker.num_workers
            my_idxs = all_idxs[wid :: nw]

        # 3) Iterate assigned shards
        for shard_idx in my_idxs:
            path = self.shard_files[shard_idx]

            # load and make a tensor
            data = torch.from_numpy(np.load(path)).long() 

            # 4) Shuffle *within* this shard in a reproducible way
            #    seed = global_seed + shard_idx ensures unique but deterministic per-shard order
            shard_seed = (seed + shard_idx) % (2**32)
            shard_gen  = torch.Generator().manual_seed(shard_seed)
            perm       = torch.randperm(data.size(0), generator=shard_gen)
            data       = data[perm]

            # 5) Yield sample by sample
            for sample in data:
                yield {'input_ids': sample}

if __name__ == "__main__":
    # 1) Top-level reproducible seed
    TOP_SEED = 42
    torch.manual_seed(TOP_SEED)

    # 2) Prepare your domains
    base = '/home/shuyaoli/llm_data/converted_dataset'
    domain_dirs = {
        'book':        os.path.join(base, 'book'),
        'arxiv':       os.path.join(base, 'arxiv'),
        'stackexchange':os.path.join(base, 'stackexchange'),
        'wiki':        os.path.join(base, 'wiki'),
        'c4-rp':       os.path.join(base, 'c4-rp'),
        'cc':          os.path.join(base, 'cc'),
        'github':      os.path.join(base, 'github'),
    }
    init_weights = [0.1]*len(domain_dirs)

    # 3) Create dataset
    master_ds = StatefulShardedDataset(
        domain_dirs=domain_dirs,
        initial_weights=init_weights,
        chunk_size=4  # small for demo
    )

    # 4) Create a Generator for DataLoader to deterministically seed workers
    g = torch.Generator()
    g.manual_seed(TOP_SEED)

    # 5) Wrap in DataLoader
    loader = DataLoader(
        master_ds,
        batch_size=2,
        num_workers=9,
        generator=g,
        persistent_workers=True
    )

    # 6) Iterate and print the first few
    print("Starting iteration...")
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print("  input_ids shape:", batch['input_ids'].shape, batch['set'])
        # Optionally, reweight on the fly:
        if i == 3:
            print(" >> Updating weights to emphasize 'wiki' and 'arxiv'")
            new_w = [0.05, 0.4, 0.05, 0.4, 0.025, 0.025, 0.025]
            master_ds.update_weights(new_w)
        if i >= 5:
            print("\nDemo complete.")
            break

    eval_dir = "/home/shuyaoli/llm_data/converted_dataset/eval_merge"

    # 3) Instantiate dataset
    ds = EvaluationShardedDataset(data_dir=eval_dir)

    # 4) Create DataLoader (with worker‐controlled seeding via generator)
    loader = DataLoader(
        ds,
        batch_size=4,         # how many samples per batch
        num_workers=2,        # spawn 2 workers
        generator=g,          # ensures each worker has a reproducible torch.initial_seed()
        persistent_workers=True,
    )

    # 5) Iterate a few batches
    print("Starting evaluation iteration:")
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print("  input_ids.shape =", batch['input_ids'].shape)
        if i >= 3:
            print("\nDemo complete.")
            break