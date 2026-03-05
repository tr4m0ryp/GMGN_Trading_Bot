#!/usr/bin/env python3
"""
Script to combine multiple CSV files into one, remove duplicates based on token_address,
and fix any misalignment issues.
"""

import pandas as pd
import glob
import os

def combine_csv_files(data_dir: str, output_file: str):
    """
    Combine all CSV files in the data directory into a single file,
    removing duplicates based on token_address.
    """
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "tokens_*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to combine:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Expected columns based on the data structure
    expected_columns = [
        'token_address', 
        'symbol', 
        'discovered_at_unix', 
        'discovered_age_sec', 
        'death_reason', 
        'chart_data_json'
    ]
    
    all_dataframes = []
    
    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")
        try:
            # Read CSV with proper handling
            df = pd.read_csv(csv_file, dtype=str, on_bad_lines='warn')
            
            print(f"  Original shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Check if columns match expected
            if list(df.columns) != expected_columns:
                print(f"  WARNING: Column mismatch detected!")
                print(f"  Expected: {expected_columns}")
                print(f"  Got: {list(df.columns)}")
                
                # Try to fix misalignment by forcing expected column names
                if len(df.columns) == len(expected_columns):
                    df.columns = expected_columns
                    print(f"  Fixed by reassigning column names")
            
            # Drop rows where token_address is null or empty
            df = df.dropna(subset=['token_address'])
            df = df[df['token_address'].str.strip() != '']
            
            print(f"  After cleaning: {df.shape}")
            all_dataframes.append(df)
            
        except Exception as e:
            print(f"  ERROR processing {csv_file}: {e}")
            continue
    
    if not all_dataframes:
        print("No data to combine!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nCombined shape (before dedup): {combined_df.shape}")
    
    # Remove duplicates based on token_address, keeping the first occurrence
    combined_df = combined_df.drop_duplicates(subset=['token_address'], keep='first')
    print(f"Combined shape (after dedup): {combined_df.shape}")
    
    # Final validation - remove any rows with misaligned data
    # Check if token_address looks valid (Solana addresses are ~44 chars, base58)
    def is_valid_token_address(addr):
        if pd.isna(addr):
            return False
        addr = str(addr).strip()
        # Solana addresses are typically 32-44 characters
        if len(addr) < 30 or len(addr) > 50:
            return False
        # Should contain only alphanumeric characters
        return addr.replace('_', '').isalnum()
    
    valid_mask = combined_df['token_address'].apply(is_valid_token_address)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"Removing {invalid_count} rows with invalid token addresses")
        combined_df = combined_df[valid_mask]
    
    print(f"Final shape: {combined_df.shape}")
    
    # Save the combined file
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined dataset to: {output_file}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total unique tokens: {len(combined_df)}")
    print(f"Death reason distribution:")
    print(combined_df['death_reason'].value_counts())

if __name__ == "__main__":
    data_dir = "/home/tr4moryp/script/gmgn_trading/ai_data/data"
    output_file = os.path.join(data_dir, "combined_tokens.csv")
    combine_csv_files(data_dir, output_file)
