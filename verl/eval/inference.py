import argparse
import json
import os
from statistics import mean, stdev
import concurrent
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import SamplingParams, LLM
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from verl.utils.reward_score.general_math_reward import compute_score as math_compute_score
from verl.utils.global_tools import external_batch_inference, SYS_PROMPT



def load_jsonl(file_path):
    """
    Load data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file.
    
    Returns:
        list: List of dictionaries, each containing data from one line of the JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

FAST_SOLVING_PROMPT = '''You are a reasoning expert assistant. Given a question, you will use your reasoning skills to solve the problem efficiently and concisely. You will be given an easy problem, so please solve it quickly without any pause, check or reflection.'''

def logic_compute_score(question, model_answer, ground_truth, data_source):
    if data_source == "prontoqa":
        candidates = ["true", "false"]
        model_answer = model_answer.lower()
        ground_truth = ground_truth.lower()
    elif data_source == "prosqa":
        extracted = question.split("?")[0].split()
        candidates = [extracted[-3].lower(), extracted[-1].lower()]
        model_answer = model_answer.lower()
        ground_truth = ground_truth.lower()
    elif data_source == "LogicBench":
        candidates = ['A', 'B', 'C', 'D']
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    def last_candidate_appearing(text, candidates):
        import re
        
        # Split text into words by removing punctuation and splitting on whitespace
        words = re.findall(r'\b\w+\b', text)
        
        last_index = -1
        last_candidate = None

        for candidate in candidates:
            # Find the last occurrence of the candidate as a complete word
            for i in range(len(words) - 1, -1, -1):
                if words[i] == candidate:
                    if i > last_index:
                        last_index = i
                        last_candidate = candidate
                    break

        return last_candidate
    
    last_candidate = last_candidate_appearing(model_answer, candidates)
    if last_candidate is None:
        return 0.0
    return 1.0 if last_candidate == ground_truth else 0.0

        

import concurrent.futures
import sys

def eval_answers(answer_file, data_source, valid_threshold=3891, num_threads=32):
    """
    Evaluate model answers based on the data source sequentially.
    
    Args:
        answer_file (str): Path to the file containing model answers.
        data_source (str): Source of the data (e.g., 'gsm8k', 'mmlu', etc.).
        valid_threshold (int): Maximum token length for valid responses.
        num_threads (int): Unused parameter, kept for backward compatibility.
    
    Returns:
        None: Results are written to a stats file.
    """
    print(f"\nEvaluating answers for {data_source}...")
    
    # Load answers
    with open(answer_file, 'r') as f:
        answer_data = [json.loads(line) for line in f if line.strip()]
     
    stats_file = answer_file.replace('.jsonl', '_stats.json')
    results = []
    
    # Process entries sequentially
    try:
        for entry in tqdm(answer_data, desc="Evaluating"):
            try:
                question = entry["prompt"]["question"]
                ground_truth = (str)(entry.get('answer'))
                model_answer = entry.get('model_answer', [])
                # Handle both single answer (str) and multiple answers (list)
                model_answers = model_answer if isinstance(model_answer, list) else [model_answer]

                # Compute strict and flexible scores for each answer
                if data_source in ["prontoqa", "prosqa", "LogicBench"]:
                    strict_scores = [0] * len(model_answers)
                    flexible_scores = [logic_compute_score(question, ans, ground_truth, data_source) for ans in model_answers]
                else:
                    strict_scores = [math_compute_score(ans, ground_truth) for ans in model_answers]
                    flexible_scores = [math_compute_score(ans, ground_truth, method="flexible") for ans in model_answers]

                result = {
                    "id": entry.get("id"),
                    "question": entry["prompt"]["question"],
                    "strict_scores": strict_scores,
                    "flexible_scores": flexible_scores,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "answer_length": entry["answer_length"],
                    "scores": (strict_scores, flexible_scores)
                }
                results.append(result)
            except Exception as exc:
                raise RuntimeError(f"Error processing entry {entry.get('id', 'unknown')}: {exc}")
    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
        sys.exit(1)
    
    # Only proceed if we have results
    if not results:
        print("No results were processed successfully. Exiting.")
        sys.exit(1)
        
    # Extract scores from results
    strict_scores = [result["scores"][0] for result in results]  # List of lists
    flexible_scores = [result["scores"][1] for result in results]  # List of lists
    
    # Calculate hit@1 and hit@n
    strict_hit_at_1 = sum(1 for scores in strict_scores if scores[0] > 0.5) / len(strict_scores) if strict_scores else 0
    strict_hit_at_n = sum(1 for scores in strict_scores if any(score > 0.5 for score in scores)) / len(strict_scores) if strict_scores else 0
    flexible_hit_at_1 = sum(1 for scores in flexible_scores if scores[0] > 0.5) / len(flexible_scores) if flexible_scores else 0
    flexible_hit_at_n = sum(1 for scores in flexible_scores if any(score > 0.5 for score in scores)) / len(flexible_scores) if flexible_scores else 0
    

    if len(strict_scores[0]) > 1:
        avg_response_length = mean(
            val for x in results for val in x["answer_length"]
        ) if results else 0
    else:
        avg_response_length = mean(x["answer_length"] for x in results) if results else 0

    # Calculate avg_correct_response_length (for correct answers using flexible scoring)
    if len(flexible_scores[0]) > 1:
        # Multiple answers per question
        correct_lengths = []
        for i, scores in enumerate(flexible_scores):
            answer_lengths = results[i]["answer_length"]
            for j, score in enumerate(scores):
                if score > 0.5:
                    correct_lengths.append(answer_lengths[j])
        avg_correct_response_length = mean(correct_lengths) if correct_lengths else 0
    else:
        # Single answer per question
        correct_lengths = [
            results[i]["answer_length"] for i, scores in enumerate(flexible_scores)
            if scores[0] > 0.5
        ]
        avg_correct_response_length = mean(correct_lengths) if correct_lengths else 0

    # Calculate avg_valid_response_length (for answers with length < valid_threshold)
    if len(strict_scores[0]) > 1:
        # Multiple answers per question
        valid_lengths = []
        for result in results:
            answer_lengths = result["answer_length"]
            for length in answer_lengths:
                if length < valid_threshold:
                    valid_lengths.append(length)
        avg_valid_response_length = mean(valid_lengths) if valid_lengths else 0
    else:
        # Single answer per question
        valid_lengths = [
            result["answer_length"] for result in results
            if result["answer_length"] < valid_threshold
        ]
        avg_valid_response_length = mean(valid_lengths) if valid_lengths else 0

    # Remove scores from final results
    for result in results:
        result.pop("scores", None)
    
    # Calculate statistics
    stats = {
        "data_source": data_source,
        "num_samples": len(strict_scores),
        "strict": {
            "hit_at_1": strict_hit_at_1,
            "hit_at_n": strict_hit_at_n,
        },
        "flexible": {
            "hit_at_1": flexible_hit_at_1,
            "hit_at_n": flexible_hit_at_n,
        },
        "avg_response_length": avg_response_length,
        "avg_correct_response_length": avg_correct_response_length,
        "avg_valid_response_length": avg_valid_response_length,
    }
    
    # Write detailed results and statistics
    with open(stats_file, 'w') as f:
        json.dump({
            "stats": stats,
            "results": results
        }, f, indent=2)
    
    # Print summary statistics
    print(f"\nEvaluation Results for {data_source}:")
    print(f"Total samples: {stats['num_samples']}")
    print(f"Strict Hit@1: {stats['strict']['hit_at_1'] * 100:.4f}%")
    print(f"Strict Hit@n: {stats['strict']['hit_at_n'] * 100:.4f}%")
    print(f"Flexible Hit@1: {stats['flexible']['hit_at_1'] * 100:.4f}%")
    print(f"Flexible Hit@n: {stats['flexible']['hit_at_n'] * 100:.4f}%")
    print(f"Average response length: {stats['avg_response_length']:.2f} tokens")
    print(f"Average correct response length: {stats['avg_correct_response_length']:.2f} tokens")
    print(f"Average valid response length: {stats['avg_valid_response_length']:.2f} tokens")
    print(f"Detailed results saved to: {stats_file}")


def run_eval(args, raw_queries, answer_file):
    """
    Run evaluation with all parameters contained in args object
    """
    # Ensure temperature > 0 when n_rollouts > 1
    if args.n_rollouts > 1 and args.temperature <= 1e-5:
        raise ValueError("Temperature must be greater than 0 when n_rollouts > 1")

    # Check if skip is enabled and answer file exists
    if args.skip and os.path.exists(answer_file):
        print(f"Skip enabled and answer file found at {answer_file}. Skipping inference...")
        data_source = raw_queries[0]["data_source"]
        with open(answer_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('answer') is not None:
                    eval_answers(answer_file, data_source, args.valid_threshold)
                    return
        print("Answer file exists but doesn't contain evaluation data.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data_source = raw_queries[0]["data_source"]
    dialogs, listed_dialogs = [], []
    answers = []

    # Choose the appropriate system prompt based on the fast_solving flag
    if args.fast_solving:
        system_prompt = FAST_SOLVING_PROMPT
    else:
        system_prompt = SYS_PROMPT
    
    for idx, question in enumerate(raw_queries):
        qs = question[args.query_key]

        if tokenizer.chat_template:
            try:
                prompt_chat = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": qs}]
                prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
            except Exception as e: # No system role supported
                prompt_chat = [
                    {"role": "user", "content": f"{system_prompt}\n{qs}"}]
                prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_chat = f"System: {system_prompt}\nUser: {qs}\nAssistant: "
            prompt_chat_str = f"System: {system_prompt}\nUser: {qs}\nAssistant: "

        # print(prompt_chat)
        # print("|" + prompt_chat_str + "|")
        # assert False

        dialogs.append(prompt_chat_str)
        listed_dialogs.append(prompt_chat)

        if "answer" in question:
            answers.append(question[args.answer_key])


    # Inference
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_token,
        n=args.n_rollouts
    )
    if not args.api_model:
        print(f"Loading model from: {args.model_path}")
        model = LLM(
            model=args.model_path,
            dtype='bfloat16',
            tensor_parallel_size=args.num_gpus_per_model,
            gpu_memory_utilization=0.85,
            disable_sliding_window="Phi" in args.model_path,
            max_model_len=5120,
        )
        vllm_outputs_raw = model.generate(dialogs, sampling_params)
        vllm_outputs = [[attempt.text.strip() for attempt in output.outputs] for output in vllm_outputs_raw]
    else:   # TODO: Test This
        vllm_outputs = external_batch_inference(args.model_path, listed_dialogs, sampling_params, n_rollout=args.n_rollouts)
    

    # Dump answers
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "w") as fout:
        for idx, model_answers in enumerate(vllm_outputs):
            # Compute token length for each answer
            token_lengths = [len(tokenizer.tokenize(ans)) for ans in model_answers]
            ans_json = {
                "id": idx,
                "prompt": raw_queries[idx],
                "answer": answers[idx] if answers else None,
                "model_answer": model_answers if args.n_rollouts > 1 else model_answers[0],
                "answer_length": token_lengths if args.n_rollouts > 1 else token_lengths[0],
            }
            fout.write(json.dumps(ans_json, ensure_ascii=True) + "\n")

    if answers:
        eval_answers(answer_file, data_source, args.valid_threshold)

    # Clean up model to free GPU memory
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True,
                        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="model_answers",
                        help="The output answer directory.")
    parser.add_argument("--max-new-token", type=int, default=512,
                        help="The maximum number of new generated tokens.")
    parser.add_argument("--num-gpus-per-model", type=int, default=1,
                        help="The number of GPUs per model.")
    parser.add_argument("--n_rollouts", type=int, default=1,
                        help="Number of answers to generate per query.")
    parser.add_argument("--proportion", type=float, default=1.0,
                        help="The portion of data used.")
    parser.add_argument("--query-key", type=str, default="question",
                        help="The key for the query text in the input file.")
    parser.add_argument("--answer-key", type=str, default="solution",
                        help="The key for the answer text in the input file.")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fast_solving", action="store_true",
                        help="Use fast solving prompt.")
    parser.add_argument("--api_model", action="store_true", help="Use API model instead of local model.")
    parser.add_argument("--valid_threshold", type=int, default=3891)
    args = parser.parse_args()


    raw_queries = load_jsonl(args.input_file)
    if args.fast_solving:
        print(f"Fast solving mode enabled for {args.input_file} - using specialized reasoning prompts")
        output_path = args.output_dir.split("/")
        output_path[-2] += "_fast"
        args.output_dir = "/".join(output_path)

    answer_file = os.path.join(args.output_dir, "result.jsonl")

    print(f"Output to {answer_file}")

    assert 0 < args.proportion <= 1
    raw_queries = raw_queries[:int(len(raw_queries) * args.proportion)]

    run_eval(args, raw_queries, answer_file)