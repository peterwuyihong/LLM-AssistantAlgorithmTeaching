"""
LangGraph-based Algorithm Competition Problem Solver
State definition for the problem-solving graph.
"""

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class TestCase(TypedDict):
    """A single test case with input and expected output."""
    input: str
    expected_output: str


class ProblemState(TypedDict):
    """State schema for the algorithm competition problem-solving graph.

    This state flows through all nodes in the LangGraph, carrying problem
    context, generated code, test results, and retry/error information.
    """

    # --- Problem context (set once by load_problem, read-only after init) ---
    folder: str                           # Problem folder path
    question: str                         # Problem description
    input_format: str                     # Input format specification
    output_format: str                    # Output format specification
    data_range: str                       # Data constraints
    ignore_format: bool                   # Whether to ignore non-alphanumeric chars in comparison
    max_retries: int                      # Max retry attempts per stage
    num_generated_tests: int              # Number of test cases to generate (default 20)

    # --- Code artifacts (overwritten on each attempt) ---
    brute_force_code: Optional[str]       # Generated brute force code
    data_maker_code: Optional[str]        # Generated data maker code
    solution_code: Optional[str]          # Generated optimized solution code

    # --- Test data (in-memory, avoids filesystem coupling) ---
    sample_test_cases: list               # list[TestCase] from problem folder or APPS
    generated_test_cases: list            # list[TestCase] from data maker + brute force

    # --- Retry/error accumulation per stage ---
    bf_attempt: int                       # Brute force attempt count
    bf_error: Optional[str]               # Last brute force error message
    dm_attempt: int                       # Data maker attempt count
    dm_error: Optional[str]               # Last data maker error message
    sol_attempt: int                      # Solution attempt count
    sol_error: Optional[str]              # Last solution error message

    # --- Overall result ---
    stage: str                            # Current stage name
    result: str                           # "success" or "fail"
    result_detail: Optional[str]          # Detailed result message

    # --- Log trail (accumulates across all nodes) ---
    log: Annotated[list[str], operator.add]  # Progress messages from each node
