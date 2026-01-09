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

from collections import defaultdict

import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import random


def list_of_dicts_to_tensordict(data):
    keys = data[0].keys()
    stacked = {k: torch.tensor([d[k] for d in data]) for k in keys}
    return TensorDict(stacked, batch_size=[len(data)])


class NaiveRewardManager:
    """The reward manager."""
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", div="train", config=None) -> None:
        assert div in ['train', 'val']
        self.div = div
        self.config = config
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        if 0 < self.num_examine < 1:
            n_output = (int)(random.random() < self.num_examine)
        else:
            n_output = self.num_examine

        cur_step_ratio = data.meta_info.get("cur_step_ratio", 0)
        
        # First decode all prompts and responses
        decoded_data = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            decoded_data.append({
                "prompt_str": prompt_str,
                "solution_str": response_str,
                "response_ids": valid_response_ids.tolist(),
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
                "response_length": valid_response_length,
                "problem_index": data_item.non_tensor_batch["extra_info"]["index"],
                **data_item.batch
            })

        # First, judge correctness of all data instances
        for item in decoded_data:
            correctness, _ = self.compute_score(
                **item,
                config=self.config,
                cur_step_ratio=cur_step_ratio,
                force_simple=True,
                div=self.div,
            )
            item["correctness"] = correctness


        # Group responses by problem_index to create batch_info
        problem_to_responses = defaultdict(list)
        for i, item in enumerate(decoded_data):
            problem_index = item["problem_index"]
            problem_to_responses[problem_index].append({
                **item,
                "response_index": i
            })

        # Then compute rewards sequentially
        # This is not needed for validation, but we still do it for consistency
        for i, item in enumerate(decoded_data):
            problem_index = item["problem_index"]
            batch_info = problem_to_responses[problem_index]
            item["score"], item["details"] = self.compute_score(
                **item,
                config=self.config,
                cur_step_ratio=cur_step_ratio,
                batch_info=batch_info,
                div=self.div,
            )

        # log and save the results
        all_details = []
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        for i, item in enumerate(decoded_data):
            score = item["score"]
            details = item["details"]

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, item["response_length"] - 1] = reward
            all_details.append(details)

            if i < n_output:
                # already_print_data_sources[data_source] += 1
                print("[div]", self.div)
                print("[prompt]\n"+ item["prompt_str"])
                print("[response]\n" + item["solution_str"])
                print("[data_source]", item["data_source"])
                print("[ground_truth]", item["ground_truth"])
                print("[score]", score)
                for key, value in details.items():
                    print(f"[{key}]", value)
                    

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "reward_details": list_of_dicts_to_tensordict(all_details)
            }
        else:
            return reward_tensor
