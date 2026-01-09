# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import pandas as pd
import datasets
import unicodedata
import ftfy  # For fixing Unicode issues automatically

from verl.utils.hdfs_io import copy, makedirs


def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)


def remove_tool_calls(x: str):
    """
    Remove all content between << and >> from a string.
    
    Args:
        x (str): Input string potentially containing tool calls
        
    Returns:
        str: String with tool calls removed
    """
    if x is None:
        return x
    
    # Use regex to remove all content between << and >>
    return re.sub(r'<<.*?>>', '', x, flags=re.DOTALL)

def clean_text(text: str):
    """
    Clean and normalize text using standard libraries to handle Unicode issues.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Normalized and cleaned text
    """
    if text is None:
        return text
    
    # Use ftfy to fix common Unicode issues
    text = ftfy.fix_text(text)
    
    # Normalize Unicode (convert to canonical form)
    text = unicodedata.normalize('NFKC', text)
    
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/gsm_symbolic")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--noformat", action="store_true")

    args = parser.parse_args()

    data_source = "gsm8k_symbolic"

    dataset = datasets.load_dataset("apple/GSM-Symbolic", "p1")

    test_dataset = dataset["test"]

    from verl.utils.global_tools import math_instruction, math_instruction_noformat
    instruction_following = math_instruction_noformat if args.noformat else math_instruction
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            
            # Clean and normalize text
            question_raw = clean_text(question_raw)
            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            answer_raw = remove_tool_calls(answer_raw)
            answer_raw = clean_text(answer_raw)

            thought, answer = [x.strip() for x in answer_raw.split("####")]
            trace = f'''{thought}\nAnswer: {answer}'''

            data = {
                "data_source": data_source,
                "question": question,
                "answer": answer,
                "solution": trace,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn


    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    # Remove all original columns from the dataset
    columns_to_keep = ["data_source", "question", "answer", "solution", "ability", "reward_model", "extra_info"]
    test_dataset = test_dataset.select_columns(columns_to_keep)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)


    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Convert parquet files to json format
    test_parquet_path = os.path.join(local_dir, "test.parquet")

    test_json_path = os.path.join(local_dir, "test.jsonl")
    

    parquet_to_json(test_parquet_path, test_json_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

