from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    model = ProviderConfig("openai", "unused", 0)
    return LabConfig(tmp_path, tmp_path / "data", tmp_path / "state", 180, 4, model, model)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("user") == "# User profile\n"
    path = store.write_text("user", "# User profile\n- location: Huế\n")
    assert path.exists() and store.file_size("user") > 0
    assert store.edit_text("user", "Huế", "Đà Nẵng")
    assert store.facts("user")["location"] == "Đà Nẵng"
    assert not store.edit_text("user", "Hà Nội", "Huế")
    assert store.path_for("../bad").is_relative_to(store.root_dir)


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(100, 4)
    for _ in range(8):
        manager.append("thread", "user", "Một đoạn dài để kiểm tra compact memory. " * 6)
    context = manager.context("thread")
    assert manager.compaction_count("thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= 4


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    fact = "Chào bạn, mình tên là DũngCT. Mình đang ở Huế."
    baseline.reply("user", "first", fact)
    advanced.reply("user", "first", fact)
    question = "Mình tên gì và hiện tại mình ở đâu?"
    assert "DũngCT" in advanced.reply("user", "second", question)["answer"]
    assert "Huế" in advanced.reply("user", "third", question)["answer"]
    assert "DũngCT" not in baseline.reply("user", "second", question)["answer"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for _ in range(16):
        message = "Mình đang so sánh memory và token cost trong một hội thoại rất dài. " * 15
        baseline.reply("user", "long", message)
        advanced.reply("user", "long", message)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")


def test_confidence_filter_and_correction(tmp_path: Path) -> None:
    assert extract_profile_updates("Mình tên gì?") == {}
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    agent.reply("user", "first", "Mình đang ở Huế và đang làm MLOps engineer.")
    agent.reply("user", "first", "Mình đang ở Đà Nẵng. Hà Nội chỉ là nơi mình bay ra họp. Có lúc mình đùa là product manager.")
    facts = agent.profile_store.facts("user")
    assert facts["location"] == "Đà Nẵng"
    assert facts["profession"] == "MLOps engineer"
    assert "Hà Nội" not in agent.profile_store.read_text("user")
    assert "product manager" not in agent.profile_store.read_text("user")
