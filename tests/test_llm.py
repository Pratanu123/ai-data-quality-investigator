import pytest

from src.llm import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    OpenRouterConfigurationError,
    getllm,
)


def test_getllm_requires_openrouter_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    with pytest.raises(OpenRouterConfigurationError, match="OPENROUTER_API_KEY"):
        getllm()


def test_getllm_uses_openrouter_free_model_by_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    llm = getllm()

    assert llm.model_name == DEFAULT_OPENROUTER_MODEL
    assert llm.openai_api_base == OPENROUTER_BASE_URL
    assert llm.max_retries == 0
