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
# from . import gsm8k, math, prime_math, prime_code

from . import general_math_reward
from . import efficiency_reward
import random

def _default_compute_score(data_source, solution_str, ground_truth, config = None, force_simple=False, **kwargs):
    # Important: now we treat each reward separately, even those efficiency rewards!
    cur_step_ratio = kwargs.get("cur_step_ratio", 0)

    # consider alpha scheduling
    if config.trainer.alpha_scheduler.type == "linear":
        peak = config.trainer.alpha_scheduler.peak_pos
        bottom = config.trainer.alpha_scheduler.bottom
        increase = config.trainer.alpha_scheduler.increase
        decrease = config.trainer.alpha_scheduler.decrease

        if cur_step_ratio <= peak:
            if increase:
                # Linearly increase alpha from bottom to 1
                scheduler_term = bottom + (1.0 - bottom) * (cur_step_ratio / peak)
            else:
                # Keep alpha at 1 until peak
                scheduler_term = 1.0
        else:
            if decrease:
                # Linearly decrease alpha from 1 to bottom
                scheduler_term = 1.0 - (1.0 - bottom) * ((cur_step_ratio - peak) / (1.0 - peak))
            else:
                # Keep alpha at 1 after peak
                scheduler_term = 1.0

    elif config.trainer.alpha_scheduler.type == "constant":
        scheduler_term = 1.0
    else:
        raise NotImplementedError(f"Unknown alpha scheduler type: {config.trainer.alpha_scheduler.type}")


    rewards, factors = config.trainer.reward_types, config.trainer.reward_factors
    assert len(rewards) == len(factors), "Rewards and factors must have the same length"
    reward_breakdown = {}

    # first calculate basic correctness score
    if data_source in ["gsm8k", "math", "open-r1", "aime"]:
        basic_reward_cls = general_math_reward.compute_score
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
    basic_res = basic_reward_cls(solution_str, ground_truth, config = config, **kwargs)

    # Then calculate all rewards
    if kwargs.get("div", "train") == "val" or force_simple: # In validation, we don't consider any efficiency rewards
        res = basic_res
        reward_breakdown["rewards/base"] = basic_res
    else:
        res = 0
        # Process each reward type
        for reward, factor in zip(rewards, factors):
            if reward == "base":
                reward_value = basic_res
            elif reward == "O1-pruner":
                assert config.actor_rollout_ref.rollout.n > 1, "O1-pruner reward is only available for multi-sample rollout"
                reward_value = efficiency_reward.compute_score_o1p(solution_str, ground_truth, method = method, config = config, **kwargs)
            elif reward == "ER":
                assert config.actor_rollout_ref.rollout.n > 1, "ER reward is only available for multi-sample rollout"
                reward_value = efficiency_reward.compute_score_er(solution_str, ground_truth, method = method, config = config, **kwargs)
            elif reward == "ppl_qa":
                reward_value = efficiency_reward.compute_ppl_qa_log(solution_str, ground_truth, config = config, **kwargs)
            # elif reward == "token_overlap":
            #     assert config.actor_rollout_ref.rollout.n > 2, "need at least 3 samples for token overlap reward"
            #     reward_value = efficiency_reward.compute_token_overlap(solution_str, ground_truth, config = config, **kwargs)
            # elif reward == "token_overlap_plus":
            #     assert config.actor_rollout_ref.rollout.n > 2, "need at least 3 samples for token overlap reward"
            #     reward_value = efficiency_reward.compute_token_overlap_plus(solution_str, ground_truth, config = config, **kwargs)
            # elif reward == "token_lcs":
            #     assert config.actor_rollout_ref.rollout.n > 2, "need at least 3 samples for token lcs reward"
            #     reward_value = efficiency_reward.compute_token_lcs(solution_str, ground_truth, config = config, **kwargs)
            # elif reward == "fmv":
            #     assert config.actor_rollout_ref.rollout.n > 2, "need at least 3 samples for fmv reward"
            #     reward_value = efficiency_reward.compute_fmv(solution_str, ground_truth, config = config, **kwargs)
            else:
                raise NotImplementedError(f"Reward {reward} is not implemented")

            # Store individual reward value and weighted contribution
            reward_breakdown["rewards/" + reward] = float(reward_value) if not isinstance(reward_value, dict) else reward_value

            if reward == "base":
                res += reward_value * factor
            else:
                res += reward_value * factor * scheduler_term
                
    # Prepare return value
    try:
        if isinstance(res, dict):
            final_score = res
        elif isinstance(res, (int, float, bool)):
            final_score = float(res)
        else:
            final_score = float(res[0])
    except Exception as e:
        print(e)
        print(res, type(res))
        raise ValueError(f"Invalid return type from compute_score: {type(res)}. Expected int, float, bool, or dict.")

    return final_score, reward_breakdown
