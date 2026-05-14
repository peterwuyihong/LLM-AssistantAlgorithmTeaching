"""
Load problem node: reads question.json and sample test cases
from the problem folder into the graph state.
"""

import json
import os
from src.graph.state import ProblemState


def load_problem_node(state: ProblemState) -> dict:
    """Load problem description and sample test cases into state.

    Reads question.json from the problem folder, then scans
    the sample/ subdirectory for .in/.ans file pairs.

    Args:
        state: Current graph state with 'folder' set.

    Returns:
        Dict with problem context fields and sample_test_cases.
    """
    folder = state["folder"]

    # Read question.json
    question_path = os.path.join(folder, "question.json")
    with open(question_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question = data.get("question", "")
    input_format = data.get("input_format", "")
    output_format = data.get("output_format", "")
    data_range = data.get("data_range", "")

    # Collect sample test cases from sample/ subdirectory
    sample_test_cases = []
    sample_dir = os.path.join(folder, "sample")

    if os.path.isdir(sample_dir):
        # Find all .in files and pair them with .ans files
        in_files = sorted(
            [f for f in os.listdir(sample_dir) if f.endswith(".in")]
        )
        for in_file in in_files:
            base_name = in_file[:-3]  # strip .in
            ans_file = base_name + ".ans"
            in_path = os.path.join(sample_dir, in_file)
            ans_path = os.path.join(sample_dir, ans_file)

            if os.path.exists(ans_path):
                with open(in_path, "r", encoding="utf-8") as inf:
                    test_input = inf.read()
                with open(ans_path, "r", encoding="utf-8") as ansf:
                    expected_output = ansf.read()
                sample_test_cases.append({
                    "input": test_input,
                    "expected_output": expected_output,
                })

    return {
        "folder": folder,
        "question": question,
        "input_format": input_format,
        "output_format": output_format,
        "data_range": data_range,
        "sample_test_cases": sample_test_cases,
        "log": [f"[LOAD] Loaded problem from {folder}, "
                f"{len(sample_test_cases)} sample test cases found"],
    }
