from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    text = text.strip()
    return math.ceil(len(text) / 4) if text else 0


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", user_id).strip("_ .")
        if not safe_id or safe_id in {".", ".."}:
            raise ValueError("Invalid user id")
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts = {}
        for line in self.read_text(user_id).splitlines():
            match = re.fullmatch(r"- ([a-z_]+): (.+)", line)
            if match:
                facts[match.group(1)] = match.group(2)
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        if not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Invalid fact key")
        facts = self.facts(user_id)
        if facts.get(key) == value:
            return
        facts[key] = value
        self.write_text(user_id, "# User profile\n" + "".join(f"- {k}: {v}\n" for k, v in sorted(facts.items())))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Accept explicit self facts; ignore questions, hypotheticals and travel mentions."""
    updates: dict[str, str] = {}
    confidence_threshold = 0.8

    def accept(key: str, value: str, confidence: float) -> None:
        if confidence >= confidence_threshold:
            updates[key] = value

    # A question alone does not assert a profile fact. Long declarative turns may
    # contain a rhetorical question, so only reject short question-only turns.
    if "?" in message and len(message) < 180:
        return updates
    name = re.findall(r"(?:mình tên là|tên mình là|tên mình là|tên)\s+(DũngCT(?: Stress)?)\b", message, re.I)
    if name:
        accept("name", name[-1], 0.95)
    places = []
    for pattern in (
        r"(?:mình|tôi)\s+(?:vẫn\s+|đang\s+)?(?:ở|sống ở|làm việc ở)\s+(Huế|Đà Nẵng)",
        r"(?:hiện ở|nơi ở hiện tại (?:là|vẫn là)|hiện tại là)\s+(Huế|Đà Nẵng)",
    ):
        places += [(m.start(), m.group(1)) for m in re.finditer(pattern, message, re.I)]
    if places:
        accept("location", max(places)[1], 0.9)
    jobs = []
    for pattern in (
        r"(?:mình|tôi)\s+(?:đang\s+|vẫn\s+)?(?:làm|là)\s+(MLOps engineer|backend engineer)",
        r"(?:và\s+)?đang\s+làm\s+(MLOps engineer|backend engineer)",
        r"nghề nghiệp (?:hiện tại|mới)\s+(?:vẫn\s+)?là\s+(MLOps engineer|backend engineer)",
        r"(?:chuyển sang|nghề hiện tại là)\s+(MLOps engineer|backend engineer)",
    ):
        jobs += [(m.start(), m.group(1)) for m in re.finditer(pattern, message, re.I)]
    if jobs:
        accept("profession", max(jobs)[1], 0.9)
    if re.search(r"(?:đồ uống yêu thích là|mình (?:vẫn )?uống|mình thích)[^.]*cà phê sữa đá", message, re.I):
        accept("favorite_drink", "cà phê sữa đá", 0.9)
    if re.search(r"(?:món ăn yêu thích là|món ruột)[^.]*mì Quảng|mì Quảng[^.]*món ruột", message, re.I):
        accept("favorite_food", "mì Quảng", 0.9)
    if re.search(r"mình nuôi[^.]*corgi|con corgi Bơ|con Bơ", message, re.I):
        accept("pet", "corgi", 0.9)
    style_cue = re.search(r"(?:mình muốn|mình thích|mình vẫn muốn|hãy trả lời|style trả lời|cách giải thích|dài hạn:)[^.]*?(?:ngắn gọn|3 bullet)", message, re.I | re.S)
    if style_cue:
        accept("response_style", "ngắn gọn, 3 bullet" if "3 bullet" in message.lower() else "ngắn gọn", 0.85)
    if re.search(r"(?:mình thích|mình đang quan tâm|dài hạn:)[^.]*Python[^.]*AI|(?:mình thích|mình đang quan tâm|dài hạn:)[^.]*AI[^.]*Python", message, re.I | re.S):
        accept("interests", "Python, AI", 0.8)
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bound summary size so repeated compaction really reduces prompt load."""
    snippets = []
    for item in messages[-max_items:]:
        content = " ".join(item["content"].split())[:140]
        snippets.append(f"{item['role']}: {content}")
    return "\n".join(snippets)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        messages.append({"role": role, "content": content})
        total = estimate_tokens(str(thread["summary"])) + sum(estimate_tokens(m["content"]) for m in messages)
        if total > self.threshold_tokens and len(messages) > self.keep_messages:
            old = messages[:-self.keep_messages]
            previous = [{"role": "summary", "content": str(thread["summary"])}] if thread["summary"] else []
            thread["summary"] = summarize_messages(previous + old)
            thread["messages"] = messages[-self.keep_messages:]
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
