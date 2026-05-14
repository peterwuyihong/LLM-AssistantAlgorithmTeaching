"""
CLI entry point for the LangGraph-based Algorithm Competition Problem Solver.

Usage:
    python -m src.runner -solve <problem_folder> [-f] [--force] [--max-retries N] [--num-tests N]
    python -m src.runner -apps <problem_id> [--split train|test] [-f] [--force] [--max-retries N]
    python -m src.runner -apps-batch <count> [--split train|test] [-f] [--force] [--start-from N]
    python -m src.runner -convert-apps [--split train|test] [--limit N]

Skip logic:
    By default, problems that already have a result.json in their folder are SKIPPED.
    Use --force to rerun already-solved problems.

All results are persisted:
    - Per-problem: {problem_folder}/brute_force.py, data.py, solution.py, result.json, execution_log.txt
    - Batch:       results/batch_{timestamp}/summary.json + per-problem result.json

Examples:
    python -m src.runner -solve reverse_pair
    python -m src.runner -solve reverse_pair --force --max-retries 5
    python -m src.runner -apps 0000 --split train
    python -m src.runner -apps 0000 --split train --force
    python -m src.runner -apps-batch 100 --split train
    python -m src.runner -apps-batch 100 --split train --force
"""

import argparse
import json
import os
import sys
from datetime import datetime

from src.graph.builder import invoke_solve, save_results_to_folder
from src.config import MAX_RETRIES, NUM_GENERATED_TESTS, APPS_TRAIN_DIR, APPS_TEST_DIR

# Default batch results directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _is_already_solved(folder: str) -> bool:
    """Check if a problem has already been solved by looking for result.json.

    Args:
        folder: Path to the problem folder.

    Returns:
        True if result.json exists in the folder.
    """
    return os.path.isfile(os.path.join(folder, "result.json"))


def _load_existing_result(folder: str) -> dict:
    """Load an existing result.json from a problem folder.

    Args:
        folder: Path to the problem folder.

    Returns:
        Parsed result dict, or empty dict if file doesn't exist or is invalid.
    """
    result_path = os.path.join(folder, "result.json")
    if not os.path.isfile(result_path):
        return {}
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def solve_single(folder: str, ignore_format: bool = False,
                 max_retries: int = MAX_RETRIES,
                 num_tests: int = NUM_GENERATED_TESTS,
                 save: bool = True,
                 force: bool = False) -> dict:
    """Solve a single problem and print results.

    Args:
        folder: Path to the problem folder.
        ignore_format: Whether to ignore non-alphanumeric chars in comparison.
        max_retries: Maximum retry attempts per stage.
        num_tests: Number of test cases to generate.
        save: Whether to save results to disk (default True).
        force: If True, rerun even if result.json already exists.

    Returns:
        Final state dict.
    """
    # Skip check: if already solved and not forced, load existing result
    if not force and _is_already_solved(folder):
        existing = _load_existing_result(folder)
        if existing:
            status = existing.get("result", "unknown")
            print(f"\n[SKIP] {folder} — already solved ({status})")
            print(f"  Use --force to rerun.")
            # Return a dict shaped like a fresh result so callers don't break
            return {
                "result": existing.get("result", "unknown"),
                "result_detail": existing.get("result_detail", ""),
                "bf_attempt": existing.get("brute_force", {}).get("attempt_count", 0),
                "bf_error": existing.get("brute_force", {}).get("last_error"),
                "dm_attempt": existing.get("data_maker", {}).get("attempt_count", 0),
                "dm_error": existing.get("data_maker", {}).get("last_error"),
                "sol_attempt": existing.get("solution", {}).get("attempt_count", 0),
                "sol_error": existing.get("solution", {}).get("last_error"),
                "log": ["[SKIP] Already solved, loaded from result.json"],
                "_skipped": True,
            }

    print(f"\n{'='*60}")
    print(f"Solving: {folder}")
    print(f"{'='*60}")

    result = invoke_solve(
        folder=folder,
        ignore_format=ignore_format,
        max_retries=max_retries,
        num_generated_tests=num_tests,
        save=save,
    )

    # Print logs
    print(f"\n--- Execution Log ---")
    for log_entry in result.get("log", []):
        print(f" {log_entry}")

    # Print final result
    status = result.get("result", "unknown")
    detail = result.get("result_detail", "")

    if status == "success":
        print(f"\n\033[92m[PASS] {detail}\033[0m")
    else:
        print(f"\n\033[91m[FAIL] {detail}\033[0m")

    # Print saved file locations
    if save:
        saved_path = result.get("_saved_result_json", "")
        if saved_path:
            print(f"\n--- Saved Files ---")
            print(f" Result:      {saved_path}")
            print(f" BF Code:     {os.path.join(folder, 'brute_force.py')}")
            print(f" Data Maker:  {os.path.join(folder, 'data.py')}")
            print(f" Solution:    {os.path.join(folder, 'solution.py')}")
            print(f" Exec Log:    {os.path.join(folder, 'execution_log.txt')}")

    return result


def solve_apps_problem(problem_id: str, split: str = "train",
                       ignore_format: bool = False,
                       max_retries: int = MAX_RETRIES,
                       force: bool = False) -> dict:
    """Solve an APPS dataset problem by converting it first, then solving.

    Args:
        problem_id: The APPS problem ID (e.g., "0000", "0001").
        split: "train" or "test".
        ignore_format: Whether to ignore format in comparison.
        max_retries: Maximum retry attempts per stage.
        force: If True, rerun even if result.json already exists.

    Returns:
        Final state dict.
    """
    from src.converters.apps_converter import convert_apps_problem

    # Determine paths
    if split == "train":
        apps_dir = APPS_TRAIN_DIR
    else:
        apps_dir = APPS_TEST_DIR

    apps_problem_dir = os.path.join(apps_dir, problem_id)
    if not os.path.isdir(apps_problem_dir):
        print(f"APPS problem directory not found: {apps_problem_dir}")
        return {}

    # Convert to canonical format
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    converted_dir = os.path.join(project_root, "APPS_converted", split, problem_id)

    # Skip check before converting/solving
    if not force and _is_already_solved(converted_dir):
        existing = _load_existing_result(converted_dir)
        status = existing.get("result", "unknown") if existing else "unknown"
        print(f"[SKIP] APPS {split}/{problem_id} — already solved ({status})")
        return {
            "result": existing.get("result", "unknown") if existing else "unknown",
            "result_detail": existing.get("result_detail", "") if existing else "",
            "bf_attempt": existing.get("brute_force", {}).get("attempt_count", 0) if existing else 0,
            "bf_error": existing.get("brute_force", {}).get("last_error") if existing else None,
            "dm_attempt": existing.get("data_maker", {}).get("attempt_count", 0) if existing else 0,
            "dm_error": existing.get("data_maker", {}).get("last_error") if existing else None,
            "sol_attempt": existing.get("solution", {}).get("attempt_count", 0) if existing else 0,
            "sol_error": existing.get("solution", {}).get("last_error") if existing else None,
            "log": ["[SKIP] Already solved, loaded from result.json"],
            "_skipped": True,
        }

    print(f"Converting APPS problem {split}/{problem_id}...")
    try:
        convert_apps_problem(apps_problem_dir, converted_dir)
    except Exception as e:
        print(f"Failed to convert APPS problem: {e}")
        return {}

    # Solve the converted problem
    return solve_single(
        folder=converted_dir,
        ignore_format=ignore_format,
        max_retries=max_retries,
        force=force,
    )


def solve_apps_batch(count: int, split: str = "train",
                     start_from: int = 0,
                     ignore_format: bool = False,
                     max_retries: int = MAX_RETRIES,
                     force: bool = False,
                     results_dir: str = DEFAULT_RESULTS_DIR) -> list[dict]:
    """Solve a batch of APPS dataset problems.

    By default, skips problems that already have result.json.
    Use force=True to rerun all problems.

    Results are saved to:
    - Per-problem: {problem_folder}/result.json, execution_log.txt, etc.
    - Batch summary: {results_dir}/batch_{timestamp}/summary.json

    Args:
        count: Number of problems to solve.
        split: "train" or "test".
        start_from: Starting problem index.
        ignore_format: Whether to ignore format in comparison.
        max_retries: Maximum retry attempts per stage.
        force: If True, rerun already-solved problems. If False, skip them.
        results_dir: Directory for batch result summaries.

    Returns:
        List of result dicts.
    """
    if split == "train":
        apps_dir = APPS_TRAIN_DIR
    else:
        apps_dir = APPS_TEST_DIR

    if not os.path.isdir(apps_dir):
        print(f"APPS directory not found: {apps_dir}")
        return []

    # Get sorted problem directories
    problem_dirs = sorted(
        [d for d in os.listdir(apps_dir)
         if os.path.isdir(os.path.join(apps_dir, d))]
    )

    # Select range
    selected = problem_dirs[start_from:start_from + count]

    # Pre-check which problems are already solved (for progress reporting)
    if not force:
        already_solved = []
        need_to_run = []
        for pid in selected:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            converted_dir = os.path.join(project_root, "APPS_converted", split, pid)
            if _is_already_solved(converted_dir):
                already_solved.append(pid)
            else:
                need_to_run.append(pid)
        print(f"\n{'='*60}")
        print(f"Batch: {len(selected)} problems selected")
        print(f"  Already solved (will skip): {len(already_solved)}")
        print(f"  Need to run:                 {len(need_to_run)}")
        print(f"  Use --force to rerun all")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Batch: {len(selected)} problems selected (--force: rerun all)")
        print(f"{'='*60}")

    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    per_problem_summaries = []

    batch_start_time = datetime.now()
    batch_dir_name = f"batch_{batch_start_time.strftime('%Y%m%d_%H%M%S')}"
    batch_dir = os.path.join(results_dir, batch_dir_name)

    for idx, problem_id in enumerate(selected, 1):
        print(f"\n--- [{idx}/{len(selected)}] {split}/{problem_id} ---")

        result = solve_apps_problem(
            problem_id=problem_id,
            split=split,
            ignore_format=ignore_format,
            max_retries=max_retries,
            force=force,
        )
        results.append(result)

        is_skipped = result.get("_skipped", False)
        is_success = result.get("result") == "success"

        if is_skipped:
            skip_count += 1
            if is_success:
                success_count += 1
            else:
                fail_count += 1
        elif is_success:
            success_count += 1
        else:
            fail_count += 1

        # Collect per-problem summary for batch JSON
        per_problem_summaries.append({
            "problem_id": problem_id,
            "split": split,
            "result": result.get("result", "unknown"),
            "result_detail": result.get("result_detail", ""),
            "skipped": is_skipped,
            "bf_attempts": result.get("bf_attempt", 0),
            "bf_passed": result.get("bf_error") is None and result.get("bf_attempt", 0) > 0,
            "dm_attempts": result.get("dm_attempt", 0),
            "dm_passed": result.get("dm_error") is None and result.get("dm_attempt", 0) > 0,
            "sol_attempts": result.get("sol_attempt", 0),
            "sol_passed": result.get("sol_error") is None and result.get("sol_attempt", 0) > 0,
            "saved_result_json": result.get("_saved_result_json", ""),
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch Summary: {len(selected)} problems")
    print(f"  Success:  {success_count}")
    print(f"  Fail:     {fail_count}")
    print(f"  Skipped:  {skip_count}")
    print(f"  Newly run: {len(selected) - skip_count}")
    print(f"  Pass rate: {success_count / len(selected) * 100:.1f}%" if selected else "N/A")
    print(f"{'='*60}")

    # Save batch summary
    os.makedirs(batch_dir, exist_ok=True)
    batch_summary = {
        "batch_id": batch_dir_name,
        "timestamp": batch_start_time.isoformat(timespec="seconds"),
        "config": {
            "split": split,
            "count": count,
            "start_from": start_from,
            "max_retries": max_retries,
            "ignore_format": ignore_format,
            "force": force,
        },
        "summary": {
            "total": len(selected),
            "success": success_count,
            "fail": fail_count,
            "skipped": skip_count,
            "newly_run": len(selected) - skip_count,
            "pass_rate": f"{success_count / len(selected) * 100:.1f}%" if selected else "N/A",
        },
        "problems": per_problem_summaries,
    }

    summary_path = os.path.join(batch_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Batch summary saved to: {summary_path}")

    # Save batch execution log
    batch_log_lines = [
        f"{'='*60}",
        f"  Batch Execution Log",
        f"  Batch ID: {batch_dir_name}",
        f"  Timestamp: {batch_start_time.isoformat()}",
        f"  Total: {len(selected)}  |  Success: {success_count}  |  Fail: {fail_count}  |  Skipped: {skip_count}",
        f"{'='*60}",
        "",
    ]
    for p in per_problem_summaries:
        if p["skipped"]:
            status_icon = "SKIP"
        elif p["result"] == "success":
            status_icon = "PASS"
        else:
            status_icon = "FAIL"
        batch_log_lines.append(
            f"  [{status_icon}] {p['split']}/{p['problem_id']}  "
            f"BF:{p['bf_attempts']}  DM:{p['dm_attempts']}  SOL:{p['sol_attempts']}  "
            f"| {p['result_detail'][:80]}"
        )
    batch_log_lines.append("")
    batch_log_lines.append(f"{'='*60}")

    batch_log_path = os.path.join(batch_dir, "batch_log.txt")
    with open(batch_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(batch_log_lines))
    print(f"  Batch log saved to: {batch_log_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph-based Algorithm Competition Problem Solver"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # -solve command: solve a single problem folder
    solve_parser = subparsers.add_parser("-solve", help="Solve a single problem folder")
    solve_parser.add_argument("folder", type=str, help="Path to problem folder")
    solve_parser.add_argument("-f", action="store_true", dest="ignore_format",
                              help="Ignore non-alphanumeric chars in output comparison")
    solve_parser.add_argument("--force", action="store_true",
                              help="Force rerun even if result.json already exists")
    solve_parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                              help=f"Max retries per stage (default: {MAX_RETRIES})")
    solve_parser.add_argument("--num-tests", type=int, default=NUM_GENERATED_TESTS,
                              help=f"Number of test cases to generate (default: {NUM_GENERATED_TESTS})")

    # -apps command: solve a single APPS problem
    apps_parser = subparsers.add_parser("-apps", help="Solve a single APPS dataset problem")
    apps_parser.add_argument("problem_id", type=str, help="APPS problem ID (e.g., 0000)")
    apps_parser.add_argument("--split", type=str, default="train",
                             choices=["train", "test"], help="Dataset split (default: train)")
    apps_parser.add_argument("-f", action="store_true", dest="ignore_format",
                             help="Ignore non-alphanumeric chars in output comparison")
    apps_parser.add_argument("--force", action="store_true",
                             help="Force rerun even if result.json already exists")
    apps_parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                             help=f"Max retries per stage (default: {MAX_RETRIES})")

    # -apps-batch command: solve a batch of APPS problems
    batch_parser = subparsers.add_parser("-apps-batch", help="Solve a batch of APPS problems")
    batch_parser.add_argument("count", type=int, help="Number of problems to solve")
    batch_parser.add_argument("--split", type=str, default="train",
                              choices=["train", "test"], help="Dataset split (default: train)")
    batch_parser.add_argument("--start-from", type=int, default=0,
                              help="Starting problem index (default: 0)")
    batch_parser.add_argument("-f", action="store_true", dest="ignore_format",
                              help="Ignore non-alphanumeric chars in output comparison")
    batch_parser.add_argument("--force", action="store_true",
                              help="Force rerun all problems, even if already solved")
    batch_parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                              help=f"Max retries per stage (default: {MAX_RETRIES})")
    batch_parser.add_argument("--output-dir", type=str, default=DEFAULT_RESULTS_DIR,
                              help=f"Directory for batch result summaries (default: {DEFAULT_RESULTS_DIR})")

    # -convert-apps command: just convert APPS dataset
    convert_parser = subparsers.add_parser("-convert-apps", help="Convert APPS dataset to canonical format")
    convert_parser.add_argument("--split", type=str, default="train",
                                choices=["train", "test"], help="Dataset split (default: train)")
    convert_parser.add_argument("--limit", type=int, default=10,
                                help="Max problems to convert (0 for all, default: 10)")

    args = parser.parse_args()

    if args.command == "-solve":
        solve_single(
            folder=args.folder,
            ignore_format=args.ignore_format,
            max_retries=args.max_retries,
            num_tests=args.num_tests,
            force=args.force,
        )

    elif args.command == "-apps":
        solve_apps_problem(
            problem_id=args.problem_id,
            split=args.split,
            ignore_format=args.ignore_format,
            max_retries=args.max_retries,
            force=args.force,
        )

    elif args.command == "-apps-batch":
        solve_apps_batch(
            count=args.count,
            split=args.split,
            start_from=args.start_from,
            ignore_format=args.ignore_format,
            max_retries=args.max_retries,
            force=args.force,
            results_dir=args.output_dir,
        )

    elif args.command == "-convert-apps":
        from src.converters.apps_converter import convert_apps_dataset
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        apps_root = os.path.join(project_root, "APPS")
        output_root = os.path.join(project_root, "APPS_converted")
        convert_apps_dataset(apps_root, output_root, split=args.split, limit=args.limit)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
