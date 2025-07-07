import os
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from typing import List, Dict, Generator, Optional
from glob import glob

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
    ):
        super().__init__()
        # 1. Store configurations - these are simple and safe to pickle
        self.domain_names  = list(domain_dirs)
        self.domain_shards : list[list[str]] = [
            glob(os.path.join(path, "*.npy")) for path in domain_dirs.values()
        ]
        self.chunk_size = chunk_size

        # 2. Create the shared weights array here. It's designed to be safely
        #    pickled and shared with worker processes.
        if initial_weights:
            assert len(initial_weights) == len(self.domain_names)
            weights = initial_weights
        else:
            weights = [1.0/len(self.domain_names)] * len(self.domain_names)
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.weights.share_memory_() 


        print(f"Initialized dataset with {len(self.domain_names)} domains.")
        for name, shards in zip(self.domain_names, self.domain_shards):
            print(f"  -> Found {len(shards)} shards for domain '{name}'")

    def update_weights(self, new_weights: list[float]):
        """Updates the shared weights array from the main process."""
        with torch.no_grad():
            self.weights[:] = torch.tensor(new_weights, dtype=torch.double)

    def __iter__(self):
        # pick a seed: DataLoader has already set torch.initial_seed()
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)
        rng = torch.Generator().manual_seed(seed)

        def make_domain_gen(
            name: str,
            shard_files: List[str],
        ) -> Generator[Dict[str, torch.Tensor | str], None, None]:
            while True:
                # shuffle shards
                idxs = torch.randperm(len(shard_files), generator=rng).tolist()
                for i in idxs:
                    data = np.load(shard_files[i])
                    np.random.shuffle(data)
                    for sample in data:
                        yield {
                            "input_ids": torch.from_numpy(sample).long(), 
                            # "set": name,
                        }

        domain_gens = [make_domain_gen(name, shards) for name, shards in zip(self.domain_names, self.domain_shards)]

        while True:
            probs = self.weights / float(self.weights.sum())
            picks = torch.multinomial(probs,
                                      self.chunk_size,
                                      replacement=True,
                                      generator=rng).tolist()
            for d in picks:
                yield next(domain_gens[d])

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
        num_workers=5,
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