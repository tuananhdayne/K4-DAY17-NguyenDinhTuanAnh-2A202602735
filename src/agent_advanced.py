from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Session context plus persistent profile and bounded compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(user_id, thread_id, message) if self.langchain_agent is None else self._reply_live(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _prepare(self, user_id: str, thread_id: str, message: str) -> int:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        return self._estimate_prompt_context_tokens(user_id, thread_id)

    def _finish(self, thread_id: str, answer: str, prompt_tokens: int) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", answer)
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + answer_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {"answer": answer, "agent_tokens": answer_tokens, "prompt_tokens": prompt_tokens}

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        prompt_tokens = self._prepare(user_id, thread_id, message)
        answer = self._offline_response(user_id, thread_id, message)
        return self._finish(thread_id, answer, prompt_tokens)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        prompt_tokens = self._prepare(user_id, thread_id, message)
        context = self.compact_memory.context(thread_id)
        profile = self.profile_store.read_text(user_id)
        messages = [{"role": "system", "content": f"User profile:\n{profile}\nEarlier summary:\n{context['summary']}"}]
        messages += list(context["messages"])
        result = self.langchain_agent.invoke(messages)
        return self._finish(thread_id, str(result.content), prompt_tokens)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return (estimate_tokens(self.profile_store.read_text(user_id))
                + estimate_tokens(str(context["summary"]))
                + sum(estimate_tokens(m["content"]) for m in context["messages"]))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        lower = message.lower()
        is_recall = "?" in message or any(cue in lower for cue in ("nhắc lại", "tên gì", "ở đâu", "nghề gì", "nghề nghiệp", "yêu thích", "style trả lời", "tóm tắt ngắn về mình", "đâu mới là"))
        if not is_recall:
            return "Mình đã ghi nhận thông tin bạn chia sẻ."
        facts = self.profile_store.facts(user_id)
        keys = []
        if "tên" in lower or "dũngct" in lower or "về mình" in lower:
            keys.append("name")
        if any(term in lower for term in ("ở đâu", "nơi ở", "hiện đang ở", "huế", "hà nội", "đà nẵng")):
            keys.append("location")
        if any(term in lower for term in ("nghề", "công việc", "product manager")):
            keys.append("profession")
        if any(term in lower for term in ("đồ uống", "uống")):
            keys.append("favorite_drink")
        if any(term in lower for term in ("món ăn", "ăn yêu thích")):
            keys.append("favorite_food")
        if any(term in lower for term in ("nuôi", "con gì")):
            keys.append("pet")
        if any(term in lower for term in ("style", "trả lời", "kiểu trả lời")):
            keys.append("response_style")
        if any(term in lower for term in ("mối quan tâm", "kỹ thuật", "quan tâm chính")):
            keys.append("interests")
        if "về mình" in lower:
            keys += ["profession", "location", "interests"]
        labels = {"name": "Tên", "location": "Nơi ở hiện tại", "profession": "Nghề hiện tại", "favorite_drink": "Đồ uống yêu thích", "favorite_food": "Món ăn yêu thích", "pet": "Thú cưng", "response_style": "Style trả lời", "interests": "Mối quan tâm"}
        parts = [f"{labels[k]}: {facts[k]}" for k in dict.fromkeys(keys) if k in facts]
        return "; ".join(parts) + "." if parts else "Mình chưa có thông tin đó."

    def _maybe_build_langchain_agent(self):
        if self.force_offline or (not self.config.model.api_key and self.config.model.provider != "ollama"):
            return None
        return build_chat_model(self.config.model)
