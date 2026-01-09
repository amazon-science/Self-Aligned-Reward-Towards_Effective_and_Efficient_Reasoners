import argparse
import os
import re
import pandas as pd
import datasets
import unicodedata
import ftfy  # For fixing Unicode issues automatically
import kagglehub

from verl.utils.hdfs_io import copy, makedirs


def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/aime")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--noformat", action="store_true")

    args = parser.parse_args()

    data_source = "aime"
    
    # Download latest version
    path = kagglehub.dataset_download("hemishveeraboina/aime-problem-set-1983-2024")
    dataset = pd.read_csv(path+"/AIME_Dataset_1983_2024.csv")  # Replace with your CSV file path
    dataset = datasets.Dataset.from_pandas(dataset)

    from verl.utils.global_tools import math_instruction, math_instruction_noformat
    instruction_following = math_instruction_noformat if args.noformat else math_instruction
    
    def process_fn(example, idx):
        question_raw = example["Question"]
        
        # Clean and normalize text
        question = question_raw + "\n" + instruction_following

        answer_raw = str(example["Answer"])


        trace = f'''I got the results from a goddess in my dream.\nAnswer: {answer_raw}'''

        data = {
            "data_source": data_source,
            "question": question,
            "answer": answer_raw,
            "solution": trace,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer_raw},
            "extra_info": {
                "split": "test",
                "index": idx,
                "id": example["ID"],
                "year": example["Year"],
                "problem_number": example["Problem Number"]
            },
        }
        return data

    # Map the processing function to the dataset
    test_dataset = dataset.map(process_fn, with_indices=True)
    
    # Define columns to keep
    columns_to_keep = ["data_source", "question", "answer", "solution", "ability", "reward_model", "extra_info"]
    test_dataset = test_dataset.select_columns(columns_to_keep)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet
    test_parquet_path = os.path.join(local_dir, "test.parquet")
    test_dataset.to_parquet(test_parquet_path)

    # Convert parquet to JSON
    test_json_path = os.path.join(local_dir, "test.jsonl")
    parquet_to_json(test_parquet_path, test_json_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)