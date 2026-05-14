"""
Graph builder: constructs and compiles the LangGraph StateGraph
for the algorithm competition problem-solving workflow.

Graph topology:
    START → load_problem → generate_bf → validate_bf
    ├─ (bf_error & retries left) → generate_bf [RETRY LOOP]
    ├─ (bf_error & exhausted) → END [FAIL]
    └─ (passed) → generate_dm → validate_dm
        ├─ (dm_error & retries left) → generate_dm [RETRY LOOP]
        ├─ (dm_error & exhausted) → END [FAIL]
        └─ (passed) → generate_tests → generate_sol → validate_sol
            ├─ (sol_error & retries left) → generate_sol [RETRY LOOP]
            └─ (passed or exhausted) → END

Persistence:
    After graph execution, all generated code, results, and logs
    are saved into the problem folder:
    - brute_force.py   (generated brute force code)
    - data.py          (generated data maker code)
    - solution.py      (generated optimized solution code)
    - result.json      (structured result: status, attempts, errors, timestamps)
    - execution_log.txt (human-readable execution log)
"""

import json
import os
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from src.graph.state import ProblemState
from src.graph.nodes.load_problem import load_problem_node
from src.graph.nodes.brute_force import generate_bf_node, validate_bf_node
from src.graph.nodes.data_maker import generate_dm_node, validate_dm_node
from src.graph.nodes.test_gen import generate_tests_node
from src.graph.nodes.solution import generate_sol_node, validate_sol_node
from src.graph.routing import (
    route_after_bf_validate,
    route_after_dm_validate,
    route_after_sol_validate,
    route_after_end,
)


def build_solve_graph():
    """Build and compile the problem-solving LangGraph.

    Returns:
        A compiled LangGraph ready for invocation.
    """
    builder = StateGraph(ProblemState)

    # Add all nodes
    builder.add_node("load_problem", load_problem_node)
    builder.add_node("generate_bf", generate_bf_node)
    builder.add_node("validate_bf", validate_bf_node)
    builder.add_node("generate_dm", generate_dm_node)
    builder.add_node("validate_dm", validate_dm_node)
    builder.add_node("generate_tests", generate_tests_node)
    builder.add_node("generate_sol", generate_sol_node)
    builder.add_node("validate_sol", validate_sol_node)

    # Linear edges
    builder.add_edge(START, "load_problem")
    builder.add_edge("load_problem", "generate_bf")
    builder.add_edge("generate_bf", "validate_bf")
    builder.add_edge("generate_dm", "validate_dm")
    builder.add_edge("validate_dm", "generate_tests")
    builder.add_edge("generate_tests", "generate_sol")
    builder.add_edge("generate_sol", "validate_sol")

    # Conditional edges for retry loops
    builder.add_conditional_edges(
        "validate_bf",
        route_after_bf_validate,
        {
            "generate_bf": "generate_bf",
            "generate_dm": "generate_dm",
            "__end__": END,
        },
    )

    builder.add_conditional_edges(
        "validate_dm",
        route_after_dm_validate,
        {
            "generate_dm": "generate_dm",
            "generate_tests": "generate_tests",
            "__end__": END,
        },
    )

    builder.add_conditional_edges(
        "validate_sol",
        route_after_sol_validate,
        {
            "generate_sol": "generate_sol",
            "__end__": END,
        },
    )

    return builder.compile()


def save_results_to_folder(result: dict, folder: str) -> str:
    """Save all generated code, result metadata, and execution log
    into the problem folder.

    Files written:
    - {folder}/brute_force.py   — brute force code (or empty if not generated)
    - {folder}/data.py          — data maker code (or empty if not generated)
    - {folder}/solution.py      — optimized solution code (or empty if not generated)
    - {folder}/result.json      — structured result with status/attempts/errors
    - {folder}/execution_log.txt — human-readable log trail

    Args:
        result: Final graph state dict.
        folder: Path to the problem folder.

    Returns:
        Path to the saved result.json.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # 1. Save generated code files
    bf_code = result.get("brute_force_code") or ""
    dm_code = result.get("data_maker_code") or ""
    sol_code = result.get("solution_code") or ""

    _write_file(os.path.join(folder, "brute_force.py"), bf_code)
    _write_file(os.path.join(folder, "data.py"), dm_code)
    _write_file(os.path.join(folder, "solution.py"), sol_code)

    # 2. Save result.json
    result_data = {
        "timestamp": timestamp,
        "folder": folder,
        "result": result.get("result", "unknown"),
        "result_detail": result.get("result_detail", ""),
        "brute_force": {
            "attempt_count": result.get("bf_attempt", 0),
            "passed": result.get("bf_error") is None and result.get("bf_attempt", 0) > 0,
            "last_error": result.get("bf_error"),
            "code_saved": "brute_force.py",
        },
        "data_maker": {
            "attempt_count": result.get("dm_attempt", 0),
            "passed": result.get("dm_error") is None and result.get("dm_attempt", 0) > 0,
            "last_error": result.get("dm_error"),
            "code_saved": "data.py",
        },
        "solution": {
            "attempt_count": result.get("sol_attempt", 0),
            "passed": result.get("sol_error") is None and result.get("sol_attempt", 0) > 0,
            "last_error": result.get("sol_error"),
            "code_saved": "solution.py",
        },
        "test_info": {
            "sample_test_count": len(result.get("sample_test_cases", [])),
            "generated_test_count": len(result.get("generated_test_cases", [])),
        },
    }

    result_json_path = os.path.join(folder, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    # 3. Save execution_log.txt
    log_path = os.path.join(folder, "execution_log.txt")
    log_lines = [
        f"{'='*60}",
        f"  Problem Solving Execution Log",
        f"  Folder: {folder}",
        f"  Timestamp: {timestamp}",
        f"  Result: {result.get('result', 'unknown').upper()}",
        f"{'='*60}",
        "",
    ]
    for entry in result.get("log", []):
        log_lines.append(f"  {entry}")
    log_lines.append("")
    log_lines.append(f"{'='*60}")
    log_lines.append(f"  Summary")
    log_lines.append(f"{'='*60}")
    log_lines.append(f"  Brute Force: {result.get('bf_attempt', 0)} attempts"
                     f"  |  {'PASSED' if result.get('bf_error') is None and result.get('bf_attempt', 0) > 0 else 'FAILED'}")
    log_lines.append(f"  Data Maker:  {result.get('dm_attempt', 0)} attempts"
                     f"  |  {'PASSED' if result.get('dm_error') is None and result.get('dm_attempt', 0) > 0 else 'FAILED'}")
    log_lines.append(f"  Solution:    {result.get('sol_attempt', 0)} attempts"
                     f"  |  {'PASSED' if result.get('sol_error') is None and result.get('sol_attempt', 0) > 0 else 'FAILED'}")
    log_lines.append(f"  Sample Tests:    {len(result.get('sample_test_cases', []))}")
    log_lines.append(f"  Generated Tests: {len(result.get('generated_test_cases', []))}")
    log_lines.append(f"{'='*60}")

    _write_file(log_path, "\n".join(log_lines))

    return result_json_path


def _write_file(path: str, content: str) -> None:
    """Write content to a file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def invoke_solve(folder: str, ignore_format: bool = False, max_retries: int = 3,
                  num_generated_tests: int = 20, save: bool = True,
                  log_callback=None) -> dict:
    """Invoke the solve graph for a single problem.

    Args:
        folder: Path to the problem folder containing question.json and sample/.
        ignore_format: Whether to ignore non-alphanumeric characters in output comparison.
        max_retries: Maximum number of retries per stage.
        num_generated_tests: Number of test cases to generate.
        save: If True, save results/code/logs to the problem folder after execution.
        log_callback: Optional callable invoked with new log entries after each node.
            Signature: log_callback(new_entries: list[str])

    Returns:
        Final graph state dict with all results.
    """
    graph = build_solve_graph()

    initial_state = {
        "folder": folder,
        "question": "",
        "input_format": "",
        "output_format": "",
        "data_range": "",
        "ignore_format": ignore_format,
        "max_retries": max_retries,
        "num_generated_tests": num_generated_tests,
        "brute_force_code": None,
        "data_maker_code": None,
        "solution_code": None,
        "sample_test_cases": [],
        "generated_test_cases": [],
        "bf_attempt": 0,
        "bf_error": None,
        "dm_attempt": 0,
        "dm_error": None,
        "sol_attempt": 0,
        "sol_error": None,
        "stage": "init",
        "result": "",
        "result_detail": None,
        "log": [],
    }

    # Use graph.stream() for live progress updates
    final_state = dict(initial_state)
    try:
        for event in graph.stream(initial_state):  # type: ignore[arg-type]
            # event is dict like {"node_name": {updated_state_fields}}
            for node_name, node_output in event.items():
                final_state.update(node_output)
                final_state["stage"] = node_name
                # Push new log entries to callback
                if log_callback and node_output.get("log"):
                    try:
                        log_callback(node_output["log"])
                    except Exception:
                        pass  # Don't let callback errors break the solver
    except Exception:
        # Fallback: if streaming fails for any reason, try invoke
        result = graph.invoke(initial_state)  # type: ignore[arg-type]
        final_state.update(result)

    # Determine final result
    final_status = route_after_end(final_state)  # type: ignore[arg-type]
    final_state["result"] = final_status

    if final_status == "success":
        final_state["result_detail"] = "Solution passed all tests (sample + generated)."
    else:
        # Determine which stage failed
        if final_state.get("bf_error") and final_state.get("bf_attempt", 0) >= max_retries:
            final_state["result_detail"] = f"Brute force failed after {final_state['bf_attempt']} attempts. Last error: {final_state['bf_error'][:200]}"
        elif final_state.get("dm_error") and final_state.get("dm_attempt", 0) >= max_retries:
            final_state["result_detail"] = f"Data maker failed after {final_state['dm_attempt']} attempts. Last error: {final_state['dm_error'][:200]}"
        elif final_state.get("sol_error") and final_state.get("sol_attempt", 0) >= max_retries:
            final_state["result_detail"] = f"Solution failed after {final_state['sol_attempt']} attempts. Last error: {final_state['sol_error'][:200]}"
        else:
            final_state["result_detail"] = "Failed for unknown reason."

    # Save results to disk
    if save:
        result_json_path = save_results_to_folder(final_state, folder)
        final_state["_saved_result_json"] = result_json_path

    return final_state
