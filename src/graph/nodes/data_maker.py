"""
Data maker nodes: generate and validate test data generator code.

generate_dm_node: Calls LLM to produce a data maker (data generator) script.
validate_dm_node: Runs the data maker and verifies output by piping through brute force.
"""

import os
import subprocess
import tempfile

from src.graph.state import ProblemState
from src.llm.codegen import generate_code
from src.config import DATA_MAKER_TIMEOUT, BRUTE_FORCE_TIMEOUT


def generate_dm_node(state: ProblemState) -> dict:
    """Generate data maker code via LLM.

    The data maker should generate random test data that conforms
    to the problem's input format, at a scale small enough for
    the brute force solution to process within the time limit.

    Args:
        state: Current graph state with problem context, brute_force_code,
               and optional dm_error for retry feedback.

    Returns:
        Dict with updated dm_attempt, data_maker_code, and log.
    """
    brute_force_code = state.get("brute_force_code", "")

    prompt = f"""你是一位资深的算法竞赛专家。
请编写一个Python数据生成器(data maker)，用于生成符合题目要求的随机测试输入数据。

### Requirements
1. **Scale**: 数据规模必须适中，确保【暴力解法】能在 1 秒内运行得出结果。例如，如果题目允许n<=100000，数据生成器应生成n<=20左右的小规模数据。
2. **Format**: 必须严格遵守题目的输入格式（空格、换行、数据范围）。
3. **Output**: 程序运行时直接将生成的数据输出到stdout。**严禁**包含任何解释、代码块标记或说明文字。
4. **Randomness**: 每次运行应生成不同的随机数据，使用random模块，并以random.randint等保证多样性。
5. **Edge Cases**: 适当包含边界情况，如最小值、最大值、全0等特殊情况。

### Context
- **Question**: {state['question']}
- **Input Format**: {state['input_format']}
- **Output Format**: {state['output_format']}
- **Data Constraints**: {state['data_range']}
- **Brute Force Code (Reference for data scale)**:
{brute_force_code}
"""

    if state.get("dm_error"):
        prompt += f"""---
### Critical Fix
上一次生成的数据生成器存在以下问题，请务必修正：
{state['dm_error']}
"""

    prompt += "\n请直接给出代码实现："

    attempt = state.get("dm_attempt", 0) + 1
    code = generate_code(prompt)

    return {
        "dm_attempt": attempt,
        "data_maker_code": code,
        "dm_error": None,
        "stage": "data_maker",
        "log": [f"[DM] Attempt {attempt}: Generated data maker code ({len(code)} chars)"],
    }


def validate_dm_node(state: ProblemState) -> dict:
    """Validate data maker code by running it and piping output through brute force.

    Checks:
    1. Data maker runs without errors and produces output.
    2. Brute force can process the generated data within the time limit.
    3. The generated data conforms to the expected format (no crash in brute force).

    Args:
        state: Current graph state with data_maker_code and brute_force_code.

    Returns:
        Dict with dm_error (None if validation passed, error message if failed).
    """
    data_maker_code = state.get("data_maker_code") or ""
    brute_force_code = state.get("brute_force_code") or ""

    if not data_maker_code:
        return {"dm_error": "No data maker code generated.", "log": ["[DM] FAILED: No code"]}

    if not brute_force_code:
        return {"dm_error": "No brute force code available.", "log": ["[DM] FAILED: No BF code"]}

    tmp_dir = tempfile.mkdtemp()
    dm_path = os.path.join(tmp_dir, "data.py")
    bf_path = os.path.join(tmp_dir, "brute_force.py")

    with open(dm_path, "w", encoding="utf-8") as f:
        f.write(data_maker_code)
    with open(bf_path, "w", encoding="utf-8") as f:
        f.write(brute_force_code)

    # Step 1: Run data maker and capture output
    try:
        dm_result = subprocess.run(
            ["python", dm_path],
            capture_output=True,
            text=True,
            timeout=DATA_MAKER_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {
            "dm_error": f"Data maker timed out ({DATA_MAKER_TIMEOUT}s).\nCode:\n{data_maker_code}",
            "log": ["[DM] FAILED: Data maker TLE"],
        }

    if dm_result.returncode != 0:
        return {
            "dm_error": f"Data maker crashed with error:\n{dm_result.stderr[:500]}\nCode:\n{data_maker_code}",
            "log": ["[DM] FAILED: Data maker RE"],
        }

    generated_data = dm_result.stdout
    if not generated_data.strip():
        return {
            "dm_error": f"Data maker produced empty output.\nCode:\n{data_maker_code}",
            "log": ["[DM] FAILED: Empty output"],
        }

    # Step 2: Run brute force on generated data to verify it's processable
    try:
        bf_result = subprocess.run(
            ["python", bf_path],
            input=generated_data,
            capture_output=True,
            text=True,
            timeout=BRUTE_FORCE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {
            "dm_error": (
                f"Data scale too large — brute force timed out ({BRUTE_FORCE_TIMEOUT}s).\n"
                f"Please reduce the data scale.\nCode:\n{data_maker_code}"
            ),
            "log": ["[DM] FAILED: BF TLE on generated data (scale too large)"],
        }

    if bf_result.returncode != 0:
        return {
            "dm_error": (
                f"Generated data format is incorrect — brute force crashed.\n"
                f"BF Error:\n{bf_result.stderr[:500]}\n"
                f"Generated data:\n{generated_data[:300]}\n"
                f"Code:\n{data_maker_code}"
            ),
            "log": ["[DM] FAILED: BF RE on generated data (format error)"],
        }

    # Both passed — data maker produces valid, small-scale data
    return {
        "dm_error": None,
        "log": ["[DM] Data maker validated successfully"],
    }
