"""
Test generation node: runs the data maker multiple times,
piping each output through brute force to produce generated test cases.
"""

import os
import subprocess
import tempfile

from src.graph.state import ProblemState
from src.config import DATA_MAKER_TIMEOUT, BRUTE_FORCE_TIMEOUT, NUM_GENERATED_TESTS


def generate_tests_node(state: ProblemState) -> dict:
    """Generate test cases by running the data maker N times.

    For each iteration:
    1. Run data.py to produce random test input.
    2. Pipe that input through brute_force.py to get the expected answer.
    3. Store the {input, expected_output} pair.

    Args:
        state: Current graph state with data_maker_code and brute_force_code.

    Returns:
        Dict with generated_test_cases list and log.
    """
    data_maker_code = state.get("data_maker_code") or ""
    brute_force_code = state.get("brute_force_code") or ""
    num_tests = state.get("num_generated_tests", NUM_GENERATED_TESTS)

    tmp_dir = tempfile.mkdtemp()
    dm_path = os.path.join(tmp_dir, "data.py")
    bf_path = os.path.join(tmp_dir, "brute_force.py")

    with open(dm_path, "w", encoding="utf-8") as f:
        f.write(data_maker_code)
    with open(bf_path, "w", encoding="utf-8") as f:
        f.write(brute_force_code)

    generated_test_cases = []
    errors = []

    for i in range(num_tests):
        try:
            # Run data maker to generate input
            dm_result = subprocess.run(
                ["python", dm_path],
                capture_output=True,
                text=True,
                timeout=DATA_MAKER_TIMEOUT,
            )

            if dm_result.returncode != 0:
                errors.append(f"Run {i + 1}: Data maker RE: {dm_result.stderr[:200]}")
                continue

            test_input = dm_result.stdout

            # Run brute force to get expected output
            bf_result = subprocess.run(
                ["python", bf_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=BRUTE_FORCE_TIMEOUT,
            )

            if bf_result.returncode != 0:
                errors.append(f"Run {i + 1}: Brute force RE: {bf_result.stderr[:200]}")
                continue

            generated_test_cases.append({
                "input": test_input,
                "expected_output": bf_result.stdout,
            })

        except subprocess.TimeoutExpired:
            errors.append(f"Run {i + 1}: Timeout (data maker or brute force too slow)")
            continue
        except Exception as e:
            errors.append(f"Run {i + 1}: Unexpected error: {e}")
            continue

    log_messages = [f"[TEST] Generated {len(generated_test_cases)}/{num_tests} test cases"]
    if errors:
        log_messages.append(f"[TEST] {len(errors)} errors: {errors[0]}")

    return {
        "generated_test_cases": generated_test_cases,
        "log": log_messages,
    }
