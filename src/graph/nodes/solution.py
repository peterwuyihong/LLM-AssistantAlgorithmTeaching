"""
Solution nodes: generate and validate the optimized solution code.

generate_sol_node: Calls LLM to produce an optimized algorithm solution.
validate_sol_node: Runs the solution against both sample AND generated test cases.
"""

import os
import subprocess
import tempfile

from src.graph.state import ProblemState
from src.llm.codegen import generate_code
from src.llm.compare import compare_output, format_mismatch
from src.config import SOLUTION_TIMEOUT


def generate_sol_node(state: ProblemState) -> dict:
    """Generate optimized solution code via LLM.

    On first attempt, generates from scratch using the brute force
    code as inspiration. On retry, includes error feedback.

    Args:
        state: Current graph state with problem context, brute_force_code,
               and optional sol_error for retry feedback.

    Returns:
        Dict with updated sol_attempt, solution_code, and log.
    """
    brute_force_code = state.get("brute_force_code", "")

    prompt = f"""你是一位算法竞赛专家。请针对以下问题，编写一个高效的【优化解法】。
你可以参考下面的暴力解法来理解题意和解题思路，但需要使用更优的算法和数据结构来降低时间复杂度。

### Question Details
- **Problem Description**: {state['question']}
- **Input Format**: {state['input_format']}
- **Output Format**: {state['output_format']}
- **Data Constraints**: {state['data_range']}

### Brute Force Code (For Reference)
{brute_force_code}
"""

    if state.get("sol_error"):
        prompt += f"""---
### Error Correction
注意：你之前生成的优化解法在对拍测试中运行失败了。
**错误反馈**:
{state['sol_error']}

请仔细分析错误原因，可能是：
1. 算法逻辑本身有误（贪心策略不正确、动态规划状态定义错误等）
2. 边界条件处理不当（空输入、单个元素、最大值等）
3. 数据类型溢出（整数范围超出、取模处理错误等）
4. 特殊情况未考虑（重复元素、全0数据、极端大小等）

请修正代码并重新提交。
"""

    prompt += "\n请直接给出代码实现："

    attempt = state.get("sol_attempt", 0) + 1
    code = generate_code(prompt)

    return {
        "sol_attempt": attempt,
        "solution_code": code,
        "sol_error": None,
        "stage": "solution",
        "log": [f"[SOL] Attempt {attempt}: Generated solution code ({len(code)} chars)"],
    }


def validate_sol_node(state: ProblemState) -> dict:
    """Validate the optimized solution against all sample and generated test cases.

    Tests the solution against:
    1. Sample test cases from the problem folder.
    2. Generated test cases (from data maker + brute force cross-validation).

    Args:
        state: Current graph state with solution_code, sample_test_cases,
               and generated_test_cases.

    Returns:
        Dict with sol_error (None if all passed, error message if any failed).
    """
    code = state.get("solution_code") or ""
    if not code:
        return {
            "sol_error": "No solution code generated.",
            "log": ["[SOL] FAILED: No code to validate"],
        }

    ignore_format = state.get("ignore_format", False)
    timeout = SOLUTION_TIMEOUT

    tmp_dir = tempfile.mkdtemp()
    sol_path = os.path.join(tmp_dir, "solution.py")
    with open(sol_path, "w", encoding="utf-8") as f:
        f.write(code)

    all_test_cases = []
    sample_cases = state.get("sample_test_cases", [])
    generated_cases = state.get("generated_test_cases", [])

    # Add sample test cases with labels
    for tc in sample_cases:
        all_test_cases.append(("sample", tc))

    # Add generated test cases with labels
    for tc in generated_cases:
        all_test_cases.append(("generated", tc))

    for i, (label, tc) in enumerate(all_test_cases):
        test_input = tc["input"]
        expected = tc["expected_output"]

        try:
            result = subprocess.run(
                ["python", sol_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                err_msg = (
                    f"Test case {i + 1} ({label}) failed (Runtime Error).\n"
                    f"Input:\n{test_input[:500]}\n"
                    f"Error:\n{result.stderr[:500]}"
                )
                return {"sol_error": err_msg, "log": [f"[SOL] RE on {label} case {i + 1}"]}

            output = result.stdout
            if not compare_output(output, expected, ignore_format):
                err_msg = format_mismatch(
                    test_input[:500], expected.rstrip(), output.rstrip(), i + 1
                )
                prefix = f"({label})"
                return {
                    "sol_error": f"{prefix} {err_msg}",
                    "log": [f"[SOL] WA on {label} case {i + 1}"],
                }

        except subprocess.TimeoutExpired:
            err_msg = (
                f"Test case {i + 1} ({label}) failed (Time Limit Exceeded, {timeout}s).\n"
                f"Input:\n{test_input[:500]}"
            )
            return {"sol_error": err_msg, "log": [f"[SOL] TLE on {label} case {i + 1}"]}

    # All test cases passed
    return {
        "sol_error": None,
        "log": [f"[SOL] All {len(all_test_cases)} test cases passed "
                f"({len(sample_cases)} sample + {len(generated_cases)} generated)"],
    }
