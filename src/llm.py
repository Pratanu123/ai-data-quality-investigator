"""OpenRouter chat-model configuration."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"


class OpenRouterConfigurationError(RuntimeError):
    """Raised when OpenRouter credentials are not configured."""


def is_openrouter_configured() -> bool:
    """Check whether a local OpenRouter API key is available."""
    load_dotenv()
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def getllm() -> ChatOpenAI:
    """Create a LangChain chat model using OpenRouter's free-model router."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterConfigurationError(
            "OPENROUTER_API_KEY is not set. Add it to the project .env file."
        )

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
        or DEFAULT_OPENROUTER_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        timeout=60,
        max_retries=0,
        default_headers={"X-Title": "Amazon Product Data Quality"},
    )
