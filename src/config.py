"""
Global configuration for the LLM-Assisted Algorithm Teaching system.

LLM settings (model, base_url, api_key) are mutable at runtime
via set_llm_config() / get_llm_config() so the web UI can update them.
"""

import os

# --- Mutable LLM Configuration ---
# Initialised from environment variables, can be changed at runtime.

_llm_config: dict = {
    "model": os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
    "base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "4096")),
    "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7")),
}

# Keep module-level constants for backward compatibility (read from _llm_config)
LLM_MODEL = _llm_config["model"]
LLM_BASE_URL = _llm_config["base_url"]
LLM_API_KEY = _llm_config["api_key"]
LLM_MAX_TOKENS = _llm_config["max_tokens"]
LLM_TEMPERATURE = _llm_config["temperature"]


def get_llm_config() -> dict:
    """Return a copy of the current LLM configuration."""
    return dict(_llm_config)


def set_llm_config(**kwargs) -> dict:
    """Update LLM configuration fields and sync module-level constants.

    Accepted keys: model, base_url, api_key, max_tokens, temperature.
    Returns the updated config dict.
    """
    global LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_MAX_TOKENS, LLM_TEMPERATURE

    allowed = {"model", "base_url", "api_key", "max_tokens", "temperature"}
    for key, value in kwargs.items():
        if key in allowed and value is not None:
            _llm_config[key] = value

    # Sync module-level constants
    LLM_MODEL = _llm_config["model"]
    LLM_BASE_URL = _llm_config["base_url"]
    LLM_API_KEY = _llm_config["api_key"]
    LLM_MAX_TOKENS = _llm_config["max_tokens"]
    LLM_TEMPERATURE = _llm_config["temperature"]

    return get_llm_config()


# --- Execution Configuration ---
MAX_RETRIES = 3  # Default max retries per stage
CODE_EXECUTION_TIMEOUT = 3  # Timeout in seconds for code execution
DATA_MAKER_TIMEOUT = 3  # Timeout in seconds for data maker
BRUTE_FORCE_TIMEOUT = 3  # Timeout for brute force on generated data
SOLUTION_TIMEOUT = 3  # Timeout for solution code
NUM_GENERATED_TESTS = 20  # Number of test cases to generate

# --- APPS Dataset Configuration ---
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__))))
APPS_TRAIN_DIR = os.path.join(_PROJECT_ROOT, "APPS", "train")
APPS_TEST_DIR = os.path.join(_PROJECT_ROOT, "APPS", "test")
