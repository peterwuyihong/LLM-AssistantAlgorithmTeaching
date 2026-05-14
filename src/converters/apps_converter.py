"""
APPS dataset converter: converts APPS dataset problems into
the canonical format used by the problem-solving graph.

APPS dataset structure per problem:
    - question.txt: Problem description with -----Input----- / -----Output----- sections
    - input_output.json: {"inputs": [...], "outputs": [...]}
    - solutions.json: List of reference solution code strings
    - metadata.json: {"difficulty": "...", "url": "..."}

Canonical format (our internal format):
    - question.json: {"question", "input_format", "output_format", "data_range"}
    - sample/1.in, sample/1.ans, ...: Sample test case pairs
"""

import json
import os
import re


def parse_question_txt(question_txt: str) -> dict:
    """Parse APPS question.txt to extract structured fields.

    APPS question.txt contains the problem description with
    sections delimited by markers like -----Input-----,
    -----Output-----, -----Example-----, -----Note-----.

    Args:
        question_txt: Full text content of question.txt.

    Returns:
        Dict with keys: question, input_format, output_format, data_range.
    """
    # Split by section markers like -----Input-----, -----Output-----, etc.
    # The markers are lines of 5+ dashes surrounding a keyword
    section_pattern = r"-----\s*(\w+)\s*-----"

    # Find all section boundaries
    splits = list(re.finditer(section_pattern, question_txt))

    if not splits:
        # No section markers found — use the entire text as question
        return {
            "question": question_txt.strip(),
            "input_format": "See problem description.",
            "output_format": "See problem description.",
            "data_range": "See problem description for constraints.",
        }

    # Extract sections based on markers
    sections = {}
    # First part (before any marker) is the question/description
    sections["question"] = question_txt[:splits[0].start()].strip()

    for i, match in enumerate(splits):
        section_name = match.group(1).lower()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(question_txt)
        sections[section_name] = question_txt[start:end].strip()

    # Build structured output
    question = sections.get("question", question_txt)
    input_format = sections.get("input", "See problem description.")
    output_format = sections.get("output", "See problem description.")

    # Try to extract data range from the input section or question
    data_range = ""
    # Look for lines containing constraint patterns
    constraint_lines = []
    for line in question_txt.split("\n"):
        if re.search(r"≤|<=|≤.*≤|10\^|10\^\{|\$\\le\$|\\leq|\\le ", line):
            constraint_lines.append(line.strip())
    if constraint_lines:
        data_range = "\n".join(constraint_lines[:5])

    if not data_range:
        data_range = "See problem description for constraints."

    # Include example in question for clarity (helps LLM understand)
    example_section = sections.get("example", sections.get("examples", ""))
    if example_section:
        question = f"{question}\n\n-----Example-----\n{example_section}"

    return {
        "question": question,
        "input_format": input_format,
        "output_format": output_format,
        "data_range": data_range,
    }


def convert_apps_problem(apps_dir: str, output_dir: str) -> str:
    """Convert a single APPS problem directory to canonical format.

    Reads question.txt and input_output.json from the APPS directory,
    parses them, and writes question.json + sample/ test files.

    Args:
        apps_dir: Path to the APPS problem directory (e.g., "APPS/train/0000").
        output_dir: Path to the output directory for converted files.

    Returns:
        Path to the output directory.
    """
    # Read question.txt
    question_path = os.path.join(apps_dir, "question.txt")
    with open(question_path, "r", encoding="utf-8") as f:
        question_txt = f.read()

    # Parse into structured fields
    parsed = parse_question_txt(question_txt)

    # Read input_output.json
    io_path = os.path.join(apps_dir, "input_output.json")
    with open(io_path, "r", encoding="utf-8") as f:
        io_data = json.load(f)

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    sample_dir = os.path.join(output_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    # Write question.json
    with open(os.path.join(output_dir, "question.json"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=4, ensure_ascii=False)

    # Write sample test case files
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])

    for i, (inp, out) in enumerate(zip(inputs, outputs), start=1):
        with open(os.path.join(sample_dir, f"{i}.in"), "w", encoding="utf-8") as f:
            f.write(inp)
        with open(os.path.join(sample_dir, f"{i}.ans"), "w", encoding="utf-8") as f:
            f.write(out)

    # Copy metadata if exists
    metadata_path = os.path.join(apps_dir, "metadata.json")
    if os.path.exists(metadata_path):
        import shutil
        shutil.copy2(metadata_path, os.path.join(output_dir, "metadata.json"))

    return output_dir


def convert_apps_dataset(apps_root: str, output_root: str,
                         split: str = "train", limit: int = 0) -> list[str]:
    """Convert APPS dataset problems to canonical format.

    Args:
        apps_root: Root path of the APPS dataset (containing train/ and test/).
        output_root: Root output path for converted problems.
        split: "train" or "test".
        limit: Maximum number of problems to convert (0 for all).

    Returns:
        List of converted output directory paths.
    """
    apps_split_dir = os.path.join(apps_root, split)
    output_split_dir = os.path.join(output_root, split)

    if not os.path.isdir(apps_split_dir):
        print(f"APPS split directory not found: {apps_split_dir}")
        return []

    # Get sorted problem directories
    problem_dirs = sorted(
        [d for d in os.listdir(apps_split_dir)
         if os.path.isdir(os.path.join(apps_split_dir, d))]
    )

    if limit > 0:
        problem_dirs = problem_dirs[:limit]

    converted = []
    for problem_id in problem_dirs:
        apps_dir = os.path.join(apps_split_dir, problem_id)
        out_dir = os.path.join(output_split_dir, problem_id)

        try:
            convert_apps_problem(apps_dir, out_dir)
            converted.append(out_dir)
            print(f"  Converted {split}/{problem_id}")
        except Exception as e:
            print(f"  Failed to convert {split}/{problem_id}: {e}")

    return converted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert APPS dataset to canonical format")
    parser.add_argument("--apps-root", type=str, default=None,
                        help="Root path of APPS dataset (default: auto-detect)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Output root path (default: ./APPS_converted)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Dataset split to convert")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max problems to convert (0 for all, default: 10)")
    args = parser.parse_args()

    # Auto-detect APPS root
    if args.apps_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.apps_root = os.path.join(project_root, "APPS")

    if args.output_root is None:
        args.output_root = os.path.join(args.apps_root, "..", "APPS_converted")

    print(f"Converting APPS {args.split} dataset...")
    print(f"  Source: {args.apps_root}")
    print(f"  Output: {args.output_root}")

    result = convert_apps_dataset(
        args.apps_root,
        args.output_root,
        split=args.split,
        limit=args.limit,
    )

    print(f"\nConverted {len(result)} problems.")
