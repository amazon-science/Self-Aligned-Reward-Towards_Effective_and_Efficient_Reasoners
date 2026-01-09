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


import argparse
import os
import pandas as pd
import datasets
import math_verify
import re
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_dataset import last_boxed_only_string, remove_boxed

def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)

def remove_think_section(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/openr1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--noformat", action="store_true")

    args = parser.parse_args()

    dataset = datasets.load_dataset("open-r1/OpenR1-Math-220k", "default")["train"]

    def filter_data(example):
        global cnt
        if example["question_type"] != "math-word-problem":
            return False
        if len(example['answer']) > 20:
            return False
        if all([not x for x in example["correctness_math_verify"]]):
            return False
        pattern = r'^[-0-9.()\\]*(\\frac{[1-9][0-9]*}{[1-9][0-9]*})?[-0-9.()\\]*$'
        if not re.match(pattern, example['answer']):
            return False
        if "http" in example["problem"] or "http" in example["solution"]:
            return False
        return True

    dataset = dataset.filter(filter_data)
    train_test_split = dataset.train_test_split(test_size=0.08, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    data_source = "open-r1"

    from verl.utils.global_tools import math_instruction, math_instruction_noformat
    instruction_following = math_instruction_noformat if args.noformat else math_instruction
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            answer = example.pop("answer")

            # r1_answers = list(zip(example.pop("generations"), example.pop("correctness_math_verify")))
            # for ans, correctness in r1_answers:
            #     if correctness == 1:
            #         thought = remove_boxed(ans)
            #         # thought = remove_think_section(thought)
            #         thought = '\n'.join([line for line in thought.split("\n") if line.strip()])
            #         break

            thought = example.pop("solution")
            lines = [line for line in thought.split("\n") if line.strip()]
            if lines and (answer in lines[0] or "answer" in lines[0].lower()):
                lines = lines[1:]
            if lines:
                thought = '\n'.join(lines)
            else:
                thought = f"I think the answer is {answer}."
            
            for key in list(example.keys()):
                if key not in ["problem", "solution", "answer", "problem_type", "source", "question_type"]:
                    example.pop(key, None)
            
            question = question + "\n" + instruction_following
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

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

    # Convert parquet files to json format
    train_parquet_path = os.path.join(local_dir, "train.parquet")
    test_parquet_path = os.path.join(local_dir, "test.parquet")
    train_json_path = os.path.join(local_dir, "train.jsonl")
    test_json_path = os.path.join(local_dir, "test.jsonl")
    
    parquet_to_json(train_parquet_path, train_json_path)
    parquet_to_json(test_parquet_path, test_json_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

