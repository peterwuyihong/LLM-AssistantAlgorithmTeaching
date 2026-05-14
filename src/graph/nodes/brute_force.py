"""
Brute force nodes: generate and validate brute force solution code.

generate_bf_node: Calls LLM to produce brute force code.
validate_bf_node: Runs the code against sample test cases.
"""

import os
import subprocess
import tempfile

from src.graph.state import ProblemState
from src.llm.codegen import generate_code
from src.llm.compare import compare_output, format_mismatch
from src.config import CODE_EXECUTION_TIMEOUT


def generate_bf_node(state: ProblemState) -> dict:
    """Generate brute force solution code via LLM.

    On first attempt, generates from scratch. On retry, includes
    error feedback from the previous attempt.

    Args:
        state: Current graph state with problem context and optional bf_error.

    Returns:
        Dict with updated bf_attempt, brute_force_code, and log.
    """
    prompt = f"""你是一位算法竞赛专家。请针对以下问题，编写一个逻辑完全正确的【暴力搜索 (Brute Force)】解法。
不必担心时间复杂度或空间效率，首要目标是确保算法在逻辑上无懈可击，能完美通过小规模数据。

### Question Details
- **Problem Description**: {state['question']}
- **Input Format**: {state['input_format']}
- **Output Format**: {state['output_format']}
- **Data Constraints**: {state['data_range']}
"""

    if state.get("bf_error"):
        prompt += f"""---
### Error Correction
注意：你之前生成的程序在测试样例上运行失败了。
**错误反馈**:
{state['bf_error']}

请仔细检查逻辑，修正边界条件或解题思路中的偏差，重新给出一份可运行且正确的代码。
"""

    prompt += "\n请直接给出代码实现（包含必要的注释）："

    attempt = state.get("bf_attempt", 0) + 1
    code = generate_code(prompt)

    return {
        "bf_attempt": attempt,
        "brute_force_code": code,
        "bf_error": None,  # Reset error on new generation
        "stage": "brute_force",
        "log": [f"[BF] Attempt {attempt}: Generated brute force code ({len(code)} chars)"],
    }


def validate_bf_node(state: ProblemState) -> dict:
    """Validate brute force code by running it against sample test cases.

    Writes code to a temp file, executes with each sample input,
    and compares output against expected answers.

    Args:
        state: Current graph state with brute_force_code and sample_test_cases.

    Returns:
        Dict with bf_error (None if all passed, error message if any failed).
    """
    code = state.get("brute_force_code") or ""
    if not code:
        return {
            "bf_error": "No brute force code generated.",
            "log": ["[BF] FAILED: No code to validate"],
        }

    test_cases = state.get("sample_test_cases", [])
    ignore_format = state.get("ignore_format", False)
    timeout = CODE_EXECUTION_TIMEOUT

    # Write code to a temp file for execution
    tmp_dir = tempfile.mkdtemp()
    bf_path = os.path.join(tmp_dir, "brute_force.py")
    with open(bf_path, "w", encoding="utf-8") as f:
        f.write(code)

    for i, tc in enumerate(test_cases):
        test_input = tc["input"]
        expected = tc["expected_output"]

        try:
            result = subprocess.run(
                ["python", bf_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                err_msg = (
                    f"Test case {i + 1} failed (Runtime Error).\n"
                    f"Input:\n{test_input}\n"
                    f"Error:\n{result.stderr[:500]}"
                )
                return {"bf_error": err_msg, "log": [f"[BF] RE on sample case {i + 1}"]}

            output = result.stdout
            if not compare_output(output, expected, ignore_format):
                err_msg = format_mismatch(
                    test_input, expected.rstrip(), output.rstrip(), i + 1
                )
                return {"bf_error": err_msg, "log": [f"[BF] WA on sample case {i + 1}"]}

        except subprocess.TimeoutExpired:
            err_msg = (
                f"Time limit exceeded ({timeout}s) for test case {i + 1}.\n"
                f"Input:\n{test_input}"
            )
            return {"bf_error": err_msg, "log": [f"[BF] TLE on sample case {i + 1}"]}

    # All test cases passed
    return {
        "bf_error": None,
        "log": [f"[BF] All {len(test_cases)} sample cases passed"],
    }
