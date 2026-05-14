"""
LLM code generation utilities.
Wraps LangChain ChatOpenAI for code generation with Python code extraction.
"""

import re
from typing import cast, Any
from langchain_openai import ChatOpenAI
from src.config import (
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    get_llm_config,
)


def get_llm() -> ChatOpenAI:
    """Get a configured ChatOpenAI instance using the current runtime config."""
    cfg = get_llm_config()
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cast(Any, cfg["api_key"]),
        temperature=cfg["temperature"],
        max_retries=2,
        max_tokens=cfg["max_tokens"],
    )


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response text.

    Handles multiple formats:
    1. ```python ... ``` code blocks
    2. ``` ... ``` code blocks (assumed Python)
    3. Raw code if no code blocks found

    Args:
        text: The LLM response text containing code.

    Returns:
        Extracted Python code string.
    """
    # Try to extract from ```python ... ``` blocks first
    matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fallback: try generic ``` ... ``` blocks
    matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Last resort: return the whole text stripped
    return text.strip()


def generate_code(prompt: str) -> str:
    """Call LLM to generate code from a prompt.

    Args:
        prompt: The full prompt string for code generation.

    Returns:
        Extracted Python code string.
    """
    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        # Handle list content by joining text parts
        content = " ".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return extract_python_code(str(content))
