"""
Routing functions for conditional edges in the problem-solving graph.

Each routing function inspects the state after a validation node
and decides whether to retry (loop back to generation) or
proceed forward (to next stage or END).
"""

from typing import Literal

from src.graph.state import ProblemState


def route_after_bf_validate(state: ProblemState) -> Literal["generate_bf", "generate_dm", "__end__"]:
    """Route after brute force validation.

    - If bf_error is None (passed): proceed to data maker generation.
    - If bf_error exists but under max_retries: retry brute force generation.
    - If bf_error exists and exhausted retries: end with failure.

    Args:
        state: Current graph state after validate_bf.

    Returns:
        Next node name: "generate_bf", "generate_dm", or "__end__".
    """
    max_retries = state.get("max_retries", 3)

    if state.get("bf_error") is None:
        # Brute force passed — move to data maker
        return "generate_dm"

    if state.get("bf_attempt", 0) < max_retries:
        # Still have retries left
        return "generate_bf"

    # Exhausted retries — fail
    return "__end__"


def route_after_dm_validate(state: ProblemState) -> Literal["generate_dm", "generate_tests", "__end__"]:
    """Route after data maker validation.

    - If dm_error is None (passed): proceed to test generation.
    - If dm_error exists but under max_retries: retry data maker generation.
    - If dm_error exists and exhausted retries: end with failure.

    Args:
        state: Current graph state after validate_dm.

    Returns:
        Next node name: "generate_dm", "generate_tests", or "__end__".
    """
    max_retries = state.get("max_retries", 3)

    if state.get("dm_error") is None:
        # Data maker passed — move to test generation
        return "generate_tests"

    if state.get("dm_attempt", 0) < max_retries:
        # Still have retries left
        return "generate_dm"

    # Exhausted retries — fail
    return "__end__"


def route_after_sol_validate(state: ProblemState) -> Literal["generate_sol", "__end__"]:
    """Route after solution validation.

    - If sol_error is None (passed): end with success.
    - If sol_error exists but under max_retries: retry solution generation.
    - If sol_error exists and exhausted retries: end with failure.

    Args:
        state: Current graph state after validate_sol.

    Returns:
        Next node name: "generate_sol" or "__end__".
    """
    max_retries = state.get("max_retries", 3)

    if state.get("sol_error") is None:
        # Solution passed all tests — success
        return "__end__"

    if state.get("sol_attempt", 0) < max_retries:
        # Still have retries left
        return "generate_sol"

    # Exhausted retries — fail
    return "__end__"


def route_after_end(state: ProblemState) -> str:
    """Determine the final result status based on state at END.

    Called by the result-determination logic to set the final
    result and result_detail fields.

    Args:
        state: Final graph state.

    Returns:
        "success" or "fail" string.
    """
    # If solution has no error, it's a success
    if state.get("sol_error") is None and state.get("solution_code"):
        return "success"

    # Otherwise, determine which stage failed
    if state.get("bf_error") and state.get("bf_attempt", 0) >= state.get("max_retries", 3):
        return "fail"
    if state.get("dm_error") and state.get("dm_attempt", 0) >= state.get("max_retries", 3):
        return "fail"
    if state.get("sol_error") and state.get("sol_attempt", 0) >= state.get("max_retries", 3):
        return "fail"

    return "fail"
