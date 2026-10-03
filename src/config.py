from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    keys = {
        "openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    model = ProviderConfig(
        provider=provider,
        model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=os.getenv(keys[provider]) if provider in keys else None,
        base_url=os.getenv("CUSTOM_BASE_URL" if provider == "custom" else "OLLAMA_BASE_URL") if provider in {"custom", "ollama"} else None,
    )
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_key = keys.get(judge_provider)
    judge = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", model.model_name),
        temperature=0,
        api_key=os.getenv("JUDGE_API_KEY") or (os.getenv(judge_key) if judge_key else None),
        base_url=os.getenv("JUDGE_BASE_URL") or model.base_url,
    )
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200"))
    keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))
    if threshold < 1 or keep < 1:
        raise ValueError("Compact threshold and keep count must be positive")
    return LabConfig(root, root / "data", state_dir, threshold, keep, model, judge)
