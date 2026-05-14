"""
Output comparison utilities for algorithm problem validation.
Supports three comparison modes:
1. Exact comparison
2. Ignore-format comparison (only alphanumeric characters)
3. Normalized comparison (strip whitespace, compare token-by-token)
"""

import re


def compare_output(output: str, expected: str, ignore_format: bool = False) -> bool:
    """Compare program output against expected output.

    Comparison modes:
    - ignore_format=True: Compare only alphanumeric/hyphen tokens (most lenient).
    - ignore_format=False (default): Token-by-token comparison — splits both
      outputs by whitespace and compares the resulting token lists. This handles
      differences in blank lines, trailing spaces, and extra whitespace gracefully.

    Args:
        output: The actual output from the program.
        expected: The expected output.
        ignore_format: If True, compare only alphanumeric/hyphen characters.

    Returns:
        True if outputs match, False otherwise.
    """
    if ignore_format:
        out_cmp = re.findall(r"[A-Za-z0-9\-]", output)
        ans_cmp = re.findall(r"[A-Za-z0-9\-]", expected)
        return out_cmp == ans_cmp
    else:
        # Token-by-token comparison: split by whitespace and compare lists.
        # This is robust against blank line differences, trailing spaces,
        # and extra whitespace — common issues in APPS dataset.
        out_tokens = output.split()
        ans_tokens = expected.split()
        return out_tokens == ans_tokens


def _normalize_output(text: str) -> str:
    """Normalize output text for comparison.

    Strips trailing whitespace from each line, removes trailing
    empty lines, and normalizes line endings. Also normalizes
    consecutive blank lines to single blank lines.

    Args:
        text: Raw output text.

    Returns:
        Normalized text string.
    """
    lines = text.split('\n')
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in lines]
    # Remove trailing empty lines
    while lines and lines[-1] == '':
        lines.pop()
    # Normalize consecutive blank lines to single blank line
    normalized = []
    prev_blank = False
    for line in lines:
        if line == '':
            if not prev_blank:
                normalized.append(line)
            prev_blank = True
        else:
            normalized.append(line)
            prev_blank = False
    return '\n'.join(normalized)


def format_mismatch(test_input: str, expected: str, actual: str, case_index: int) -> str:
    """Format a test case mismatch into a readable error message.

    Args:
        test_input: The test case input.
        expected: The expected output.
        actual: The actual output produced.
        case_index: The test case number (1-indexed).

    Returns:
        Formatted error message string.
    """
    return (
        f"Test case {case_index} failed.\n"
        f"Input:\n{test_input}\n"
        f"Expected output:\n{expected}\n"
        f"Your output:\n{actual}"
    )
