import os
import numpy as np
from streaming import StreamingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Configuration ---
# The path to your current MDS dataset
MDS_BASE_PATH = '/home/shuyaoli/llm_data/LLM-Shearing/for_prune'

# The path where you want to save the new, clean dataset
CONVERTED_BASE_PATH = '/home/shuyaoli/llm_data/converted_dataset'

# How many samples (each of 4096 tokens) to save in each .npy shard file.
# A larger number means fewer files but larger memory usage during conversion.
SAMPLES_PER_SHARD = 8192

# List of training domains to convert
STREAM_NAMES = ['book', 'arxiv', 'stackexchange', 'wiki', 'c4-rp', 'cc', 'github', 'eval_merge']

def write_shard(out_dir, shard_id, buffer):
    """Stacks a buffer of samples and saves to a .npy file."""
    if not buffer:
        return
    # Stack the list of 1D arrays into a single 2D array
    shard_data = np.stack(buffer, axis=0)
    # Define the shard filename
    shard_filename = os.path.join(out_dir, f'shard_{shard_id:05d}.npy')
    # Save the numpy array
    np.save(shard_filename, shard_data)
    print(f"  -> Saved {shard_filename} with shape {shard_data.shape}")

# --- Main Conversion Logic ---
if __name__ == "__main__":
    for domain_name in STREAM_NAMES:
        print(f"\nProcessing domain: {domain_name}...")
        
        mds_path = os.path.join(MDS_BASE_PATH, domain_name)
        if not os.path.exists(os.path.join(mds_path, 'index.json')):
            print(f"  Skipping '{domain_name}', no 'index.json' found.")
            continue

        # Create the output directory for this domain
        output_domain_dir = os.path.join(CONVERTED_BASE_PATH, domain_name)
        os.makedirs(output_domain_dir, exist_ok=True)
        
        # Use the StreamingDataset to read the MDS data
        # batch_size=1 is important here so we get individual samples
        mds_dataset = StreamingDataset(local=mds_path, shuffle=False, batch_size=1)
        
        # Use a DataLoader for faster reading with multiple workers
        loader = DataLoader(mds_dataset, batch_size=None, num_workers=8)

        sample_buffer = []
        shard_id = 0
        
        for raw_sample in tqdm(loader, desc=f"Converting {domain_name}"):
            # Decode the raw bytes into a numpy array of uint16
            token_ids_np = np.frombuffer(raw_sample['tokens'], dtype=np.uint16)
            sample_buffer.append(token_ids_np)

            # When the buffer is full, write a shard file
            if len(sample_buffer) >= SAMPLES_PER_SHARD:
                write_shard(output_domain_dir, shard_id, sample_buffer)
                sample_buffer = []
                shard_id += 1
        
        # Write any remaining samples in the buffer as the last shard
        write_shard(output_domain_dir, shard_id, sample_buffer)

    print("\n\nConversion Complete!")