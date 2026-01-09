import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import numpy as np
import os
import argparse


def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)


def filter_by_difficulty(df, dataset_name, difficulty):
    """Filter dataframe based on dataset name and difficulty level."""
    if difficulty == "all":
        return df
    
    if 'gsm8k' in dataset_name.lower():
        # GSM8K is considered easy
        return df if difficulty == "easy" else df.head(0)  # Return empty if not asking for easy
    
    elif 'openr1' in dataset_name.lower() or 'open-r1' in dataset_name.lower():
        # Open-R1 is considered hard
        return df if difficulty == "hard" else df.head(0)  # Return empty if not asking for hard
    
    elif 'math' in dataset_name.lower():
        # Math dataset has explicit difficulty levels
        if 'level' not in df.columns:
            print(f"Warning: 'level' column not found in {dataset_name}, skipping difficulty filtering")
            return df
        
        if difficulty == "easy":
            return df[df['level'].isin(['Level 1', 1])]
        elif difficulty == "medium":
            return df # all data in math dataset can be considered medium
        elif difficulty == "hard":
            return df[df['level'].isin(['Level 5', 5])]
    
    # Default: return the original dataframe
    return df


def merge_datasets(input_dirs, output_dir, difficulty="all", max_samples_per_source=10000):
    # Lists to store dataframes for train and test
    train_dfs = []
    test_dfs = []
    
    # Load all Parquet files from input directories
    for dir_path in input_dirs:
        dataset_name = os.path.basename(dir_path)
        train_path = os.path.join(dir_path, 'train.parquet')
        test_path = os.path.join(dir_path, 'test.parquet')
        
        if os.path.exists(train_path):
            print(f"Loading training data from {train_path}")
            train_df = pd.read_parquet(train_path)
            
            # Apply difficulty filtering
            filtered_train_df = filter_by_difficulty(train_df, dataset_name, difficulty)
            
            # Ensure number of samples doesn't exceed max_samples_per_source
            if len(filtered_train_df) > max_samples_per_source:
                filtered_train_df = filtered_train_df.sample(max_samples_per_source, random_state=42)
                print(f"  - Sampled {max_samples_per_source} from {len(filtered_train_df)} samples")
            
            if len(filtered_train_df) > 0:
                print(f"  - Kept {len(filtered_train_df)} samples (difficulty: {difficulty})")
                train_dfs.append(filtered_train_df)
            else:
                print(f"  - Skipped (difficulty {difficulty} not applicable)")
        else:
            print(f"Warning: No train.parquet found in {dir_path}")
        
        if os.path.exists(test_path):
            print(f"Loading test data from {test_path}")
            test_df = pd.read_parquet(test_path)
            
            # Apply difficulty filtering
            filtered_test_df = filter_by_difficulty(test_df, dataset_name, difficulty)
            
            # Ensure number of samples doesn't exceed max_samples_per_source
            if len(filtered_test_df) > max_samples_per_source:
                filtered_test_df = filtered_test_df.sample(max_samples_per_source, random_state=42)
                print(f"  - Sampled {max_samples_per_source} from {len(filtered_test_df)} samples")
            
            if len(filtered_test_df) > 0:
                print(f"  - Kept {len(filtered_test_df)} samples (difficulty: {difficulty})")
                test_dfs.append(filtered_test_df)
            else:
                print(f"  - Skipped (difficulty {difficulty} not applicable)")
        else:
            print(f"Warning: No test.parquet found in {dir_path}")
    
    # Merge train dataframes
    if train_dfs:
        print("Merging training data...")
        merged_train_df = pd.concat(train_dfs, ignore_index=True)
        print(f"Combined training set: {len(merged_train_df)} samples")
        
        # Re-arrange indexes in extra_info
        print("Re-arranging indexes in training data...")
        for i, row in enumerate(merged_train_df.itertuples()):
            if 'extra_info' in merged_train_df.columns:
                if isinstance(merged_train_df.at[i, 'extra_info'], dict):
                    extra_info = merged_train_df.at[i, 'extra_info'].copy()
                    extra_info['index'] = i
                    merged_train_df.at[i, 'extra_info'] = extra_info
    else:
        print("No training data found after filtering!")
        return
    
    # Merge test dataframes
    if test_dfs:
        print("Merging test data...")
        merged_test_df = pd.concat(test_dfs, ignore_index=True)
        print(f"Combined test set: {len(merged_test_df)} samples")
        
        # Re-arrange indexes in extra_info
        print("Re-arranging indexes in test data...")
        for i, row in enumerate(merged_test_df.itertuples()):
            if 'extra_info' in merged_test_df.columns:
                if isinstance(merged_test_df.at[i, 'extra_info'], dict):
                    extra_info = merged_test_df.at[i, 'extra_info'].copy()
                    extra_info['index'] = i
                    merged_test_df.at[i, 'extra_info'] = extra_info
    else:
        print("Warning: No test data found after filtering!")
        merged_test_df = None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    train_parquet_path = os.path.join(output_dir, 'train.parquet')
    train_json_path = os.path.join(output_dir, 'train.jsonl')
    
    # Write the train table to Parquet and JSON files
    print(f"Writing train data to {train_parquet_path}")
    pq.write_table(pa.Table.from_pandas(merged_train_df), train_parquet_path)
    parquet_to_json(train_parquet_path, train_json_path)
    
    # Write the test table to Parquet and JSON files if it exists
    if merged_test_df is not None and not merged_test_df.empty:
        test_parquet_path = os.path.join(output_dir, 'test.parquet')
        test_json_path = os.path.join(output_dir, 'test.jsonl')
        
        print(f"Writing test data to {test_parquet_path}")
        pq.write_table(pa.Table.from_pandas(merged_test_df), test_parquet_path)
        parquet_to_json(test_parquet_path, test_json_path)
    
    # Print statistics
    print(f"\nMerge complete!")
    print(f"Total training samples: {len(merged_train_df)}")
    if merged_test_df is not None and not merged_test_df.empty:
        print(f"Total test samples: {len(merged_test_df)}")
    else:
        print("No test samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge multiple parquet datasets')
    parser.add_argument('--input-dirs', '-i', nargs='+', required=True, 
                       help='List of input directories that contain train.parquet and test.parquet')
    parser.add_argument('--output-dir', '-o', required=True,
                       help='Output directory for merged dataset')
    parser.add_argument('--difficulty', '-d', default="all", choices=["easy", "medium", "hard", "all"],
                       help='Filter by difficulty level')
    
    args = parser.parse_args()
    
    args.output_dir += "_" + args.difficulty
    merge_datasets(args.input_dirs, args.output_dir, args.difficulty)

    # python examples/data_preprocess/merge_datasets.py -i data/gsm8k data/math data/openr1 -o data/math_combined -d hard