import numpy as np
import torch
import numpy as np
import math



def compute_score_o1p(solution_str, ground_truth, method="strict", config = None, **kwargs):
    """The scoring function with O1-pruner style length penalty."""

    base_score = kwargs["correctness"] # This field must exist, refer to naive.py, is 0/1
    length = kwargs.get("response_length", 1).item()
    info_from_same_query = kwargs["batch_info"]

    ref_avg_length_lst = np.array([x["response_length"] for x in info_from_same_query], dtype=np.float32)

    average_length = np.mean(ref_avg_length_lst).item()
    length_penalty = (length - average_length) / length
    length_penalty = min(1, max(-1, length_penalty)) # clip to [-1, 1]
    return - length_penalty


def compute_score_er(solution_str, ground_truth, method="strict", config = None, **kwargs):
    """The scoring function with ER style length penalty."""
    base_score = kwargs["correctness"] # This field must exist, refer to naive.py
    if base_score < 1e-5:
        return 0

    def sigmoid(x):
        return float(1 / (1 + np.exp(-x)))

    length = kwargs.get("response_length", 1).item()
    info_from_same_query = kwargs["batch_info"]
    
    correct_info = [x for x in info_from_same_query if x.get("correctness", 0) > 0]
    
    if len(correct_info) <= 1:
        length_penalty = 0
    else:
        ref_avg_length_lst = np.array([x["response_length"] for x in correct_info], dtype=np.float32)
        
        average_length = np.mean(ref_avg_length_lst).item()
        std_length = np.std(ref_avg_length_lst).item()
        
        length_penalty = (sigmoid((length - average_length) / (std_length + 1e-5)) - 0.5) * 2
    
    return - base_score * length_penalty


def compute_ppl_qa_log(solution_str, ground_truth, config = None, **kwargs):
    """Compute PPL-QA log score."""
    ppl_qa, ppl_a = kwargs["perplexity"].item(), kwargs["perplexity_a"].item()

    if config.trainer.ppl_qa.use_log: # Use ppl or log(ppl) (cross-entropy)
        QA_term, A_term = np.log(ppl_qa), np.log(ppl_a)
    else:
        QA_term, A_term = ppl_qa, ppl_a

    if config.trainer.ppl_qa.clip_value > 1:
        QA_term = min(QA_term, config.trainer.ppl_qa.clip_value)
        A_term = min(A_term, config.trainer.ppl_qa.clip_value)
    
    if config.trainer.ppl_qa.use_comparison == False: # Formula: -QA
        assert config.trainer.ppl_qa.use_denominator == False
        return -QA_term + 5    # +5 doesn't affect the GRPO, just makes reward positive
        
    if config.trainer.ppl_qa.use_denominator:  # Formula: (A - QA) / A
        return max((A_term - QA_term) / (A_term + 1e-9), -1) # clip to [-1, 1] range
    else: # Formula: A - QA
        return A_term - QA_term




# def compute_token_overlap(solution_str, ground_truth, config = None, **kwargs):
#     """Compute token overlap (unique) score."""
    
#     info_from_same_query = kwargs["batch_info"]
#     cur_tokens = kwargs["response_ids"]
#     reference_tokens = [x["response_ids"] for x in info_from_same_query]
#     reference_tokens = [token_seq for token_seq in reference_tokens if token_seq != cur_tokens]  # Exclude the current tokens from reference tokens

#     # Create token sets for each reference answer
#     ref_token_sets = [set(tokens) for tokens in reference_tokens]
    
#     # Find tokens that appear in ALL reference answers
#     if len(reference_tokens) > 1:
#         tokens_in_all = set.intersection(*ref_token_sets) if ref_token_sets else set()
#     else:
#         tokens_in_all = set()
    
#     # Create combined reference token set (excluding tokens that appear in ALL answers)
#     ref_token_set = set(token for tokens in reference_tokens for token in tokens) - tokens_in_all
    
#     # Current tokens set (excluding tokens that appear in ALL answers)
#     cur_tokens_set = set(cur_tokens) - tokens_in_all

#     # Count how many of the current text's tokens appear in the reference set
#     overlap_count = sum(1 for token in cur_tokens_set if token in ref_token_set)

#     # Compute the proportion of overlapping tokens
#     overlap_ratio = overlap_count / len(cur_tokens_set) if cur_tokens_set else 0.0

#     return overlap_ratio


# from typing import List, Set
# def get_set_n_gram(input: List, config) -> Set:
#     n = config.n_gram
#     repeat_as_different = config.repeat_as_different
#     input_len = len(input)
#     result = set()
    
#     for i in range(1, n + 1):
#         occurrence_count = {}  # Track occurrences for each subsequence
#         for j in range(input_len - i + 1):
#             subsequence = tuple(input[j:j + i])
#             if repeat_as_different:
#                 # Count occurrences for this subsequence
#                 occurrence_count[subsequence] = occurrence_count.get(subsequence, -1) + 1
#                 # Add nested tuple with subsequence and occurrence time
#                 result.add((subsequence, occurrence_count[subsequence]))
#             else:
#                 # Add subsequence directly
#                 result.add(subsequence)
                
#     return result


# def calc_overall_ppl(ppls: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
#     """
#     Calculate overall perplexity from sequence-level perplexities and lengths (PyTorch version).

#     Args:
#         ppls (torch.Tensor): Tensor of shape (N,) with sequence-level perplexities.
#         lengths (torch.Tensor): Tensor of shape (N,) with corresponding sequence lengths.

#     Returns:
#         torch.Tensor: Scalar tensor representing the overall query-level perplexity.
#     """
#     # Ensure float64 precision for numerical stability
#     ppls = ppls.to(dtype=torch.float64)
#     lengths = lengths.to(dtype=torch.float64)

#     # Compute NLL_i = L_i * log(PPL_i)
#     nlls = lengths * torch.log(ppls)

#     # Total number of tokens
#     total_tokens = lengths.sum()

#     # Average token-level NLL
#     avg_nll = nlls.sum() / total_tokens

#     # Convert back to perplexity
#     ppl_q = torch.exp(avg_nll)

#     return ppl_q



# def compute_token_overlap_plus(solution_str, ground_truth, config = None, **kwargs):
#     """Compute token overlap plus score."""
#     info_from_same_query = kwargs["batch_info"]

#     # Filter: correct answers need at least 3, incorrect answers need at least 1
#     n_cor_ans = (int)(sum([x["correctness"] for x in info_from_same_query]) + 0.5)
#     if n_cor_ans == 0 or (n_cor_ans <= 1 and kwargs["correctness"] > 0.999):
#         return 0.0

#     reference_token_sets = [get_set_n_gram(x["response_ids"], config = config.trainer.overlap_plus) for x in info_from_same_query]
#     tokens_in_all = set.intersection(*reference_token_sets)

#     cur_tokens = kwargs["response_ids"]
#     cur_tokens_set = get_set_n_gram(cur_tokens, config = config.trainer.overlap_plus) - tokens_in_all

#     cor_tokens_sets = []
#     for x in info_from_same_query:
#         ref_tokens = x["response_ids"]
#         if ref_tokens == cur_tokens:
#             continue
#         if x["correctness"] > 0.999:
#             ref_tokens_set = get_set_n_gram(ref_tokens, config = config.trainer.overlap_plus) - tokens_in_all
#             cor_tokens_sets.append(ref_tokens_set)

#     if len(cor_tokens_sets) == 0:
#         return 0.0

#     def overlap(a: set, b: set):
#         if config.trainer.overlap_plus.use_jaccard:
#             return len(a & b) / len(a | b) if (a | b) else 0.0
#         else:
#             return len(a & b) / len(a) if a else 0.0

#     score = sum([overlap(cur_tokens_set, cor_tokens_set) for cor_tokens_set in cor_tokens_sets]) / len(cor_tokens_sets)

#     if config.trainer.overlap_plus.entropy_upweight:
#         ppls = [item["perplexity"] for item in info_from_same_query if "perplexity" in item]
#         lengths = [item["response_length"] for item in info_from_same_query if "response_length" in item]
#         # print(calc_overall_ppl(torch.tensor(ppls), torch.tensor(lengths)).item())
#         overall_ppl = calc_overall_ppl(torch.tensor(ppls), torch.tensor(lengths)).item()
#         return score * (1 + 1 / overall_ppl)  # Upweight by perplexity
#     else:
#         return score


# import rapidfuzz
# def compute_token_lcs(solution_str, ground_truth, config = None, **kwargs):
#     """Compute token lcs score."""
    
#     info_from_same_query = kwargs["batch_info"]
#     cur_tokens = kwargs["response_ids"]
#     reference_tokens = [x["response_ids"] for x in info_from_same_query]
#     reference_tokens = [token_seq for token_seq in reference_tokens if token_seq != cur_tokens]

#     def lcs_length(seq1, seq2):
#         return max(len(seq1), len(seq2)) - rapidfuzz.distance.LCSseq.distance(seq1, seq2)
    
#     # Calculate LCS ratio for each reference sequence
#     lcs_ratios = []
#     for ref_seq in reference_tokens:
#         lcs_len = lcs_length(cur_tokens, ref_seq)
#         # Calculate ratio as LCS length / current sequence length
#         ratio = lcs_len / len(cur_tokens) if len(cur_tokens) > 0 else 0.0
#         lcs_ratios.append(ratio)

#     # Return average LCS ratio
#     return sum(lcs_ratios) / len(lcs_ratios) if lcs_ratios else 0.0



# def compute_fmv(solution_str, ground_truth, config = None, **kwargs):
#     """Compute token overlap plus score."""
#     info_from_same_query = kwargs["batch_info"]

#     # Filter: correct answers need at least 3, incorrect answers need at least 1
#     n_cor_ans = (int)(sum([x["correctness"] for x in info_from_same_query]) + 0.5)
#     if n_cor_ans == 0 or (n_cor_ans <= 2 and kwargs["correctness"] > 0.999):
#         return 0.0

#     reference_token_sets = [get_set_n_gram(x["response_ids"], config = config.trainer.fmv) for x in info_from_same_query]
#     correct_token_sets = [get_set_n_gram(x["response_ids"], config = config.trainer.fmv) for x in info_from_same_query if x["correctness"] > 0.999]
#     tokens_in_all = set.intersection(*reference_token_sets)

#     cur_tokens = kwargs["response_ids"]
#     cur_tokens_set = get_set_n_gram(cur_tokens, config = config.trainer.fmv) - tokens_in_all

#     if len(correct_token_sets) == 0:
#         return 0.0
    
#     token_frequency = {}
#     for tokens in correct_token_sets:
#         for token in tokens:
#             if token not in tokens_in_all:
#                 token_frequency[token] = token_frequency.get(token, 0) + 1

#     def mapped_score(freq, total):
#         assert 0 <= freq <= total
#         if config.trainer.fmv.promote_incorrect and kwargs["correctness"] < 0.001:
#             score = freq - 0.5 # for incorrect answer, any overlap token will be rewarded
#             return score if score > 0 else score / config.trainer.fmv.neg_factor
#         else:
#             score = freq - total / 2
#             return score if score > 0 else score / config.trainer.fmv.neg_factor

#     # Compute the maximum possible score (upper bound)
#     max_possible_score = sum(
#         mapped_score(freq, n_cor_ans)
#         for freq in token_frequency.values()
#         if mapped_score(freq, n_cor_ans) > 0
#     )

#     if max_possible_score < 1e-5: # no positive tokens
#         return 0.0

#     # Compute the score of the current answer
#     current_score = sum(
#         mapped_score(token_frequency.get(token, 0), n_cor_ans)
#         for token in cur_tokens_set
#     )

#     return max(current_score / max_possible_score, 0)

