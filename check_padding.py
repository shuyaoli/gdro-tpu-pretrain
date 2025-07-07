import os
import numpy as np
from streaming import StreamingDataset
from torch.utils.data import DataLoader
import sentencepiece as spm
import argparse
from tqdm import tqdm

def run_padding_check(mds_data_path: str, tokenizer_model_path: str, num_samples_to_check: int):
    """
    Reads samples from an MDS dataset and checks for the presence of a padding token.

    Args:
        mds_data_path (str): Path to the MDS stream directory (e.g., './LLM-Shearing/for_prune/book').
        tokenizer_model_path (str): Path to the 'tokenizer.model' file.
        num_samples_to_check (int): The number of samples to inspect.
    """
    print("--- Padding Verification Script ---")

    # 1. Load tokenizer model and get the padding token ID
    if not os.path.exists(tokenizer_model_path):
        print(f"Error: Tokenizer model not found at '{tokenizer_model_path}'")
        return

    print(f"Loading tokenizer from: {tokenizer_model_path}")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
    pad_id = sp.pad_id()
    
    # SentencePiece uses -1 if no pad_id is defined in the model.
    if pad_id == -1:
        print("Tokenizer model does not have a padding token defined. No padding is possible.")
        return
        
    print(f"Found Padding Token ID: {pad_id}")

    # 2. Load the MDS dataset stream
    if not os.path.exists(os.path.join(mds_data_path, 'index.json')):
        print(f"Error: MDS index.json not found in '{mds_data_path}'")
        return

    print(f"Loading MDS data from: {mds_data_path}")
    mds_dataset = StreamingDataset(local=mds_data_path, shuffle=False, batch_size=1)
    
    # Use a DataLoader for potentially faster reading
    loader = DataLoader(mds_dataset, batch_size=None, num_workers=4)

    # 3. Iterate, decode, and check for padding
    padding_found_count = 0
    samples_checked = 0

    print(f"\nChecking the first {num_samples_to_check} samples for padding...")
    for raw_sample in tqdm(loader, total=num_samples_to_check):
        if samples_checked >= num_samples_to_check:
            break

        # Decode the raw bytes into a numpy array of uint16
        token_ids_np = np.frombuffer(raw_sample['tokens'], dtype=np.uint16)
        
        # Check if the padding token ID exists in the array
        if pad_id in token_ids_np:
            padding_found_count += 1
        
        samples_checked += 1

    # 4. Report the final results
    print("\n--- Verification Complete ---")
    print(f"Total samples inspected: {samples_checked}")
    print(f"Samples containing the padding token (ID: {pad_id}): {padding_found_count}")

    if padding_found_count == 0:
        print("\nConclusion: NO PADDING DETECTED.")
        print("It is safe to create an attention_mask of all ones.")
    else:
        print("\nConclusion: PADDING DETECTED.")
        print("You will need to create a proper attention_mask based on the location of the padding tokens.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check for padding tokens in an MDS dataset.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True,
        help="Path to the MDS stream directory (e.g., './LLM-Shearing/for_prune/book')."
    )
    parser.add_argument(
        "--tokenizer_model", 
        type=str, 
        required=True, 
        help="Path to the 'tokenizer.model' file."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10000, 
        help="Number of samples to check."
    )
    args = parser.parse_args()

    run_padding_check(args.data_dir, args.tokenizer_model, args.num_samples)