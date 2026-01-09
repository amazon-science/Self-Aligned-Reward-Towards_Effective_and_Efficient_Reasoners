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

import re
import numpy as np
from sympy import sympify, S


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        # find the last numerical expression
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward if there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.' or empty
            for ans in reversed(answer):
                if ans not in invalid_str:
                    final_answer = ans.replace(",", "").replace("$", "")
                    break
    return final_answer


def are_numbers_equivalent(num_str1, num_str2, tolerance=1e-7):
    """Check if two number strings are mathematically equivalent.

    Args:
        num_str1: First number string
        num_str2: Second number string
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        bool: True if numbers are equivalent, False otherwise
    """
    try:
        # Clean the strings
        clean_str1 = num_str1.replace(",", "").replace("$", "")
        clean_str2 = num_str2.replace(",", "").replace("$", "")

        # Try symbolic evaluation first for exact comparison
        try:
            # Convert to sympy objects for exact comparison
            sym1 = sympify(clean_str1)
            sym2 = sympify(clean_str2)
            if sym1 == sym2:
                return True
        except:
            pass  # Fall back to numerical comparison

        # Convert to float for numerical comparison
        float1 = float(clean_str1)
        float2 = float(clean_str2)

        # Use numpy for robust floating point comparison
        return np.isclose(float1, float2, rtol=tolerance, atol=tolerance)
    except:
        # If any conversion fails, they're not equivalent
        return False


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    assert method in ["strict", "flexible"]
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if method == "strict":
            # In strict mode, still require exact format match
            if answer == ground_truth:
                return score
            # But also check numeric equivalence as a fallback
            elif are_numbers_equivalent(answer, ground_truth):
                return format_score
            else:
                return format_score
        else:
            # In flexible mode, use numeric equivalence
            if are_numbers_equivalent(answer, ground_truth):
                return score
            else:
                return format_score





if __name__ == "__main__":
    # Test cases for are_numbers_equivalent function
    test_cases = [
        # Basic equivalence
        ("10", "10", True),
        ("10", "10.0", True),
        ("10.0", "10.00", True),
        ("10", "10.00001", False),  # Outside tolerance
        
        # Formatting variations
        ("10,000", "10000", True),
        ("$10,000", "10000", True),
        ("$10,000.00", "10000", True),
        ("10,000.50", "10000.5", True),
        
        # Negative numbers
        ("-10", "-10.0", True),
        ("-10,000", "-10000", True),
        ("-10", "10", False),
        
        # Very large numbers
        ("1000000000", "1,000,000,000", True),
        ("1e9", "1000000000", True),
        
        # Decimal precision
        ("0.1234567", "0.1234566", True),  # Within tolerance
        ("0.12345", "0.12346", False),  # Outside default tolerance
        
        # Edge cases
        ("0", "0.0", True),
        ("0", "-0", True),
        
        # Invalid inputs should return False
        ("abc", "123", False),
        ("10", "abc", False),
    ]
    
    print("Testing are_numbers_equivalent function:")
    for i, (num1, num2, expected) in enumerate(test_cases):
        result = are_numbers_equivalent(num1, num2)
        status = "PASS" if result == expected else "FAIL"
        print(f"Test {i+1}: {num1} == {num2} → Expected: {expected}, Got: {result} - {status}")
    
    # Testing with different tolerance values
    print("\nTesting with custom tolerance values:")
    # These should be equivalent with higher tolerance
    print(f"0.12345 == 0.12348 (tolerance=1e-4): {are_numbers_equivalent('0.12345', '0.12348', tolerance=1e-4)}")
    # These should NOT be equivalent with lower tolerance
    print(f"0.1234567 == 0.1234569 (tolerance=1e-8): {are_numbers_equivalent('0.1234567', '0.1234569', tolerance=1e-8)}")
    
    # Test complex cases
    complex_cases = [
        # Testing extract_solution with strict mode
        ("The answer is #### 42", "42", "strict", True),
        ("The answer is #### $42.00", "42", "strict", True),
        ("The answer is 42", "42", "strict", False),  # No "#### " format
        
        # Testing extract_solution with flexible mode
        ("The answer is 42", "42", "flexible", True),
        ("The calculation is 10 + 20 + 12 = 42", "42", "flexible", True),
        ("First I got 10, then 20, and finally 42.", "42", "flexible", True),
        ("I calculated $4,200.00 divided by 100 to get 42", "42", "flexible", True),
    ]
    
    print("\nTesting compute_score function:")
    for i, (solution, ground_truth, method, should_score) in enumerate(complex_cases):
        score = compute_score(solution, ground_truth, method=method)
        expected = 1.0 if should_score else 0.0
        status = "PASS" if score == expected else "FAIL"
        print(f"Test {i+1}: {method} mode, solution: \"{solution}\", ground_truth: \"{ground_truth}\" → Got: {score}, Expected: {expected} - {status}")