"""
LLM-based translation for problem statements (English -> Chinese).
Uses the configured ChatOpenAI instance to translate algorithm
competition problem descriptions.
"""

from langchain_openai import ChatOpenAI
from src.config import get_llm_config

TRANSLATION_SYSTEM_PROMPT = """You are a professional translator specializing in algorithm competition problems. 
Translate the following problem statement from English to Chinese. 

Rules:
1. Keep all mathematical notation, variable names, and code snippets in English.
2. Keep proper nouns (like "Polycarp", "Codeforces") unchanged.
3. Translate the problem description naturally, as if it were originally written in Chinese.
4. Preserve the structure: Input, Output, Example/Note sections should use Chinese headers.
5. Keep the input/output format descriptions accurate.
6. Do NOT add any explanation or commentary — only the translation.
7. Use standard Chinese algorithm competition terminology (e.g., "time limit" -> "时间限制", "memory limit" -> "内存限制").
"""


def translate_problem_statement(text: str, source_lang: str = "en", target_lang: str = "zh") -> str:
    """Translate a problem statement using the LLM.

    Args:
        text: The problem statement text to translate.
        source_lang: Source language code (default: en).
        target_lang: Target language code (default: zh).

    Returns:
        Translated text string.
    """
    if not text or not text.strip():
        return ""

    cfg = get_llm_config()
    llm = ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=0.3,  # Lower temperature for more consistent translation
        max_retries=2,
        max_tokens=cfg["max_tokens"],
    )

    prompt = f"{TRANSLATION_SYSTEM_PROMPT}\n\n---\n\n{text}"

    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                for part in content
            )
        return str(content).strip()
    except Exception as e:
        raise RuntimeError(f"Translation failed: {e}") from e
