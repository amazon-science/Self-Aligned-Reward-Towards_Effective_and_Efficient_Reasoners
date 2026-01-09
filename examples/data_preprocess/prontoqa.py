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
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os
import pandas as pd
import datasets
import re
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_dataset import last_boxed_only_string, remove_boxed

def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)



def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/prosqa")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--noformat", action="store_true")

    args = parser.parse_args()


    dataset = datasets.load_dataset("renma/ProntoQA", trust_remote_code=True)

    data_source = "prontoqa"
    test_dataset = dataset["validation"]

    from verl.utils.global_tools import math_instruction, math_instruction_noformat
    instruction_following = math_instruction_noformat if args.noformat else math_instruction
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("context") + " " + example.pop("question") + " " + instruction_following
            answer = "True" if example.pop("answer") == "A" else "False"
            trace = "The answer is " + answer + ". "

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

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)

    # Create training dataset from last 80% of original validation data
    original_dataset = datasets.load_dataset("renma/ProntoQA", trust_remote_code=True)["validation"]
    total_size = len(original_dataset)
    train_start_idx = int(total_size * 0.2)  # Start from 20% mark to get last 80%
    train_dataset = original_dataset.select(range(train_start_idx, total_size))
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Save training data
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    
    # Save test data
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

    # Convert parquet files to json format
    train_parquet_path = os.path.join(local_dir, "train.parquet")
    train_json_path = os.path.join(local_dir, "train.jsonl")
    test_parquet_path = os.path.join(local_dir, "test.parquet")
    test_json_path = os.path.join(local_dir, "test.jsonl")
    
    parquet_to_json(train_parquet_path, train_json_path)
    parquet_to_json(test_parquet_path, test_json_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

