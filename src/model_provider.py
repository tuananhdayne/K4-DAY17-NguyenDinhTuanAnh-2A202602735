from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower().replace("-", "_")
    provider = {"anthorpic": "anthropic", "google": "gemini", "open_router": "openrouter"}.get(provider, provider)
    if provider not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value}")
    return provider


def build_chat_model(config: ProviderConfig):
    """Import provider SDKs only when a live model is requested."""
    provider = normalize_provider(config.provider)
    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)
    from langchain_openrouter import ChatOpenRouter

    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return ChatOpenRouter(**kwargs)
