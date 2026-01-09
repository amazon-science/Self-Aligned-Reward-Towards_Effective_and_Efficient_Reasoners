import re
import numpy as np
from sympy import sympify, S
from verl.utils.global_tools import extract_answer, valid_answer_pattern
from math_verify import parse, verify
from sympy.parsing.latex import parse_latex
import numpy as np
import os
from contextlib import redirect_stdout, redirect_stderr

def are_numbers_equivalent(num_str1, num_str2):
    """Check if two number strings are mathematically equivalent.

    Args:
        num_str1: First number string
        num_str2: Second number string
        tolerance: Numerical tolerance for floating point comparison
    """
    try:
        # Clean the strings
        num_str1 = num_str1.replace(",", "").replace(" ", "").replace("$", "").strip()
        num_str2 = num_str2.replace(",", "").replace(" ", "").replace("$", "").strip()

        if num_str1 == num_str2:
            return True

        parsed_expr1 = parse(num_str1)
        parsed_expr2 = parse(num_str2)
        return verify(parsed_expr1, parsed_expr2)

    except:
        return False




def compute_score(solution_str, ground_truth, **kwargs):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if "correctness" in kwargs: # already calculated
        return kwargs["correctness"]

    answer = extract_answer(solution_str)
    if (answer == "") or (not re.fullmatch(valid_answer_pattern, answer)):
        return 0

    return (int)(are_numbers_equivalent(answer, ground_truth))



if __name__ == "__main__":
    # Example usage

    solution_str = "To determine if the student's partial solution will lead to the correct answer, let's analyze the given information and the approach the student has taken.\n\nThe student correctly identifies that to find the intersection of \\(y = h(2x)\\) and \\(y = 2j(x)\\), we need to find \\(x\\) such that \\(h(2x) = 2j(x)\\). The student also correctly lists the points of intersection of \\(y = h(x)\\) and \\(y = j(x)\\) as \\((2, 2)\\), \\((4, 6)\\), \\((6, 12)\\), and \\((8, 12)\\).\n\nNow, let's check each point to see if it satisfies the condition \\(h(2x) = 2j(x)\\):\n\n1. For \\((2, 2)\\):\n   - \\(h(2 \\cdot 1) = h(2) = 2\\)\n   - \\(2j(1) = 2 \\cdot 2 = 4\\)\n   - \\(2 \\neq 4\\), so \\((1, 2)\\) is not a solution.\n\n2. For \\((4, 6)\\):\n   - \\(h(2 \\cdot 2) = h(4) = 6\\)\n   - \\(2j(2) = 2 \\cdot 6 = 12\\)\n   - \\(6 \\neq 12\\), so \\((2, 6)\\) is not a solution.\n\n3. For \\((6, 12)\\):\n   - \\(h(2 \\cdot 3) = h(6) = 12\\)\n   - \\(2j(3) = 2 \\cdot 6 = 12\\)\n   - \\(12 = 12\\), so \\((3, 12)\\) is a solution.\n\n4. For \\((8, 12)\\):\n   - \\(h(2 \\cdot 4) = h(8) = 12\\)\n   - \\(2j(4) = 2 \\cdot 6 = 12\\)\n   - \\(12 = 12\\), so \\((4, 12)\\) is a solution.\n\nThe student correctly identified that \\((3, 12)\\) and \\((4, 12)\\) are points where the conditions are satisfied. However, the student's solution is incomplete as it does not specify which of these points is the correct one to use for the sum of the coordinates.\n\nGiven that the correct answer is 16, and the only point that fits the condition and leads to the correct sum is \\((4, 12)\\) (since \\(4 + 12 = 16\\)), the student's partial solution will indeed lead to the correct answer if they correctly identify \\((4, 12)\\).\n\nTherefore, the answer is 1."
    ground_truth = 1.0
    method = "flexible"

    result = compute_score(solution_str, ground_truth)
    print(f"Score: {result}")
