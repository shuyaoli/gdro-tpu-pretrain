import os
import torch
import numpy as np
import multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

# Make sure to install mosaicml-streaming
# pip install mosaicml-streaming
from streaming import Stream, StreamingDataset

# ==============================================================================
# 1. YOUR CUSTOM DATASET (with the chunking optimization)
#    This class acts as the "sampler" or "meta-dataset".
# ==============================================================================

def worker_init_fn(worker_id):
    """Initializes each data loader worker to have its own data streams."""
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    # Each worker gets its own generator for each underlying data source.
    # The StreamingDataset objects in `dataset.sources` are designed to handle
    # this correctly. When iter() is called, they initialize properly for the worker.
    dataset.data_streams = [iter(source) for source in dataset.sources]

class StatefulStreamingDataset(IterableDataset):
    """
    An IterableDataset that samples from multiple data sources based on
    dynamically updatable weights managed in shared memory.
    """
    def __init__(self, sources, initial_weights, chunk_size=4096):
        self.sources = sources
        # Use a multiprocessing Array for weights to ensure they are shared
        # across all data loader worker processes.        
        self.weights = mp.Array('d', initial_weights)
        self.chunk_size = chunk_size
        self.data_streams = None

    def _get_weights(self):
        """Reads the current weights from the shared memory array."""
        return np.array(self.weights[:])

    def update_weights(self, new_weights: list[float]):
        """
        Updates the shared weights from the main process. 
        This method is called from the main process by a custom callback.
        """
        with self.weights.get_lock():
            for i in range(len(new_weights)):
                self.weights[i] = new_weights[i]
                
    def process_sample(self, raw_sample: dict) -> dict:
        """
        Decodes a single sample from raw bytes to a tensor dictionary based on
        the exact specifications from the data creation script.
        """
        # Use np.uint16 as the data type, which is confirmed by the source script.
        token_ids_np = np.frombuffer(raw_sample['tokens'], dtype=np.uint16)
    
        # Convert the NumPy array to a PyTorch tensor.
        # It's good practice to .copy() to avoid potential memory ownership issues.
        token_ids_tensor = torch.from_numpy(token_ids_np.copy())
        
        # The Hugging Face model and collator expect a dictionary containing
        # `input_ids` and `attention_mask`.
        final_sample = {
            'input_ids': token_ids_tensor,
            # The attention mask is a tensor of all 1s since the data is
            # pre-chunked to a fixed length with no padding.
            'attention_mask': torch.ones_like(token_ids_tensor),
            'set':raw_sample['set']
        }
        return final_sample

    def __iter__(self):
        """The core streaming logic for each worker, using chunked sampling."""
        # Initialize streams if they haven't been, relevant for num_workers=0
        if self.data_streams is None:
            self.data_streams = [iter(source) for source in self.sources]

        while True:
            # 1. Get weights and probabilities ONCE per chunk.
            current_weights = self._get_weights()
            probabilities = current_weights / np.sum(current_weights)
            
            # 2. Generate a large chunk of source indices at once.
            # This is vastly more efficient than calling it in a loop.
            source_indices = np.random.choice(
                len(self.sources), 
                size=self.chunk_size, 
                p=probabilities
            ) # generate a chunk of indices from 0 to len(sources)-1

            # 3. Iterate through the pre-sampled chunk and yield items.
            for source_idx in source_indices:
                try:
                    yield self.process_sample(next(self.data_streams[source_idx]))
                except StopIteration:
                    # A source was exhausted, restart it for continuous training
                    print(f"Worker {get_worker_info().id if get_worker_info() else 0}: Restarting stream {source_idx}.")
                    self.data_streams[source_idx] = iter(self.sources[source_idx])
                    yield self.process_sample(next(self.data_streams[source_idx]))


# ==============================================================================
# 2. MAIN SCRIPT TO INITIALIZE AND COMBINE EVERYTHING
# ==============================================================================
BATCH_SIZE = 2
if __name__ == "__main__":
    # Use spawn for multiprocessing to be safe across platforms
    mp.set_start_method("spawn", force=True)

    # --- Setup the Underlying Readers ---
    local_path = '/home/shuyaoli/llm_data/LLM-Shearing/for_prune'
    stream_names = [
        'book', 'arxiv', 'stackexchange', 'wiki', 'c4-rp', 'cc', 'github'
    ]

    print("Initializing sources...")
    # Create a list of sources, where each source is a StreamingDataset for one domain
    sources = []
    for name in stream_names:
        # Each StreamingDataset object is an independent iterable data source
        domain_dataset = StreamingDataset(
            local=os.path.join(local_path, name),
            shuffle=True, # Shuffle within this stream
            batch_size=BATCH_SIZE # We will handle batching in the final DataLoader
        )
        sources.append(domain_dataset)
    
    print(f"Created {len(sources)} data sources.")

    # --- Setup Your Custom Wrapper Dataset ---
    # Define the initial sampling weights for each stream
    # Make sure the length matches the number of streams!
    initial_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert len(initial_weights) == len(sources)

    # Instantiate your master dataset
    master_dataset = StatefulStreamingDataset(
        sources=sources,
        initial_weights=initial_weights,
        chunk_size=1
    )

    # --- Create the Final DataLoader ---
    # This DataLoader wraps YOUR dataset. Your dataset, in turn, wraps the Mosaic ones.
    final_dataloader = DataLoader(
        master_dataset,
        batch_size=BATCH_SIZE,  # Your final desired batch size
        num_workers=1,
        worker_init_fn=worker_init_fn,
        persistent_workers=True # Good practice for performance
    )

    # --- Demonstrate Usage ---
    print("\nStarting DataLoader iteration...")
    for i, batch in enumerate(final_dataloader):
        # `batch` is now a dictionary where each value is a tensor of `batch_size`
        print(f"Step {i}:")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  value: {batch['input_ids']}")
        print(f"  Attention mask: {batch['attention_mask']}")
        # You can access the 'set' to see which domain the data came from
        # Note: In a batch, you might get samples from multiple domains
        unique_sets, counts = np.unique(batch['set'], return_counts=True)
        print(f"  Domain counts in batch: {dict(zip(map(str, unique_sets), map(int, counts)))}")

        if i >= 3:
            print("\nDemonstration complete.")
            break