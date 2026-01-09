import re
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import os
import time
import random


# valid_answer_pattern = r'[-0-9.()\\]*(\\frac{[1-9][0-9]*}{[1-9][0-9]*})?[-0-9.()\\]*'


p_number = r'-?(?:\d+\.?\d*|\.\d+)'  # Handles integers and decimals
p_latex_fraction = r'-?\s*\\frac\s*{\s*' + p_number + r'\s*}\s*{\s*' + p_number + r'\s*}'

valid_answer_pattern = r'(?:' + '|'.join([
    p_latex_fraction,
    p_number,
]) + r')'



SYS_PROMPT = '''You are a reasoning expert assistant. Given a question, you will use your reasoning skills to solve the problem.'''

math_instruction = 'Please explain your reasoning process first. On the final line, output only your answer, a single number or fractional expression (e.g. 18, 0.5, \\frac{1}{2}), using the format of "Answer: xxx".'
math_instruction_noformat = 'Please explain your reasoning process before providing an answer.'


def extract_answer(s):
    matches = list(re.finditer(valid_answer_pattern, s, re.IGNORECASE))
    if not matches:
        return ""
    last_match = max(matches, key=lambda m: m.end())

    start = last_match.start()
    end = last_match.end()

    return s[start:end].strip()




def external_batch_inference(model, requests, sampling_params, n_rollout=1):
    params = {
        "temperature": sampling_params.temperature,
        "max_tokens": sampling_params.max_tokens,
    }

    # Check if temperature is properly set for multiple rollouts
    if n_rollout > 1:
        if params["temperature"] <= 1e-6:
            raise ValueError("Temperature must be positive when n_rollout > 1 to ensure response diversity")
    
    if model == "gpt-4o" or model == "gpt-4o-mini":
        active_client = OpenAI()
    else:
        import json
        nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
        active_client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = nvidia_api_key
            )
    n_threads = 5

    def get_completion(request_with_index, max_retries=5, backoff_base=5.0):
        idx, request = request_with_index
        assert isinstance(request, list) and all(isinstance(turn, dict) for turn in request), \
            "Format error. Should be a list of dictionaries"

        for attempt in range(max_retries):
            try:
                x = active_client.chat.completions.create(
                    model=model,
                    messages=request,
                    **params
                )
                return idx, x
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for request {idx}: {e}")
                if hasattr(e, "status_code") and e.status_code in [429, 500, 502, 503, 504]:
                    wait_time = backoff_base * (1.5 ** attempt) + random.uniform(0, 1) # wait time will increase exponentially
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Non-retriable error: {e}")
        raise RuntimeError(f"Failed after {max_retries} retries.")

    # Create expanded request list for multiple rollouts
    expanded_requests = []
    for i, request in enumerate(requests):
        for _ in range(n_rollout):
            expanded_requests.append((i, request))
    
    progress = len(expanded_requests) > 1
    with ThreadPoolExecutor(max_workers=min(len(expanded_requests), n_threads)) as executor:
        if progress:
            expanded_results = list(tqdm(
                executor.map(get_completion, expanded_requests),
                total=len(expanded_requests),
                desc=f"Inference (Parallel, Model: {model}, Rollouts: {n_rollout})"
            ))
        else:
            expanded_results = list(executor.map(get_completion, expanded_requests))

    # Group results by original request index
    grouped_results = {}
    for idx, result in expanded_results:
        if idx not in grouped_results:
            grouped_results[idx] = []
        grouped_results[idx].append(result.choices[0].message.content)
    
    # Convert back to a list in the original request order
    results = [grouped_results[i] for i in range(len(requests))]
    
    return results












if __name__ == "__main__":
    ans = "<<8*.25=2>> <<5*2=10.00>>"
    extracted = extract_answer(ans)
    print(extracted)

    if re.fullmatch(valid_answer_pattern, extracted):
        print("Valid answer format")
    else:
        print("Invalid answer format")