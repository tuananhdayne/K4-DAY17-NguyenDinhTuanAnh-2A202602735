from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Session memory only, indexed exclusively by thread id."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(thread_id, message) if self.langchain_agent is None else self._reply_live(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _record(self, thread_id: str, message: str, answer: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        answer_tokens = estimate_tokens(answer)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += answer_tokens
        session.messages.append({"role": "assistant", "content": answer})
        return {"answer": answer, "agent_tokens": answer_tokens, "prompt_tokens": prompt_tokens}

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        answer = "Mình chưa có thông tin đó trong phiên này." if "?" in message or "nhắc lại" in message.lower() else "Mình đã ghi nhận trong phiên này."
        return self._record(thread_id, message, answer)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        messages = session.messages + [{"role": "user", "content": message}]
        result = self.langchain_agent.invoke(messages)
        answer = str(result.content)
        return self._record(thread_id, message, answer)

    def _maybe_build_langchain_agent(self):
        if self.force_offline or (not self.config.model.api_key and self.config.model.provider != "ollama"):
            return None
        return build_chat_model(self.config.model)
