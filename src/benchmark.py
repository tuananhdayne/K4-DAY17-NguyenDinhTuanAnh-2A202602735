from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    matched = sum(part.casefold() in answer.casefold() for part in expected)
    return 0.0 if matched == 0 else 1.0 if matched == len(expected) else 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    coverage = sum(part.casefold() in answer.casefold() for part in expected) / len(expected) if expected else 1.0
    return round(0.2 + 0.8 * coverage, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    tokens = prompts = compactions = 0
    recalls: list[float] = []
    quality: list[float] = []
    users = {c["user_id"] for c in conversations}
    before = {user: agent.memory_file_size(user) if isinstance(agent, AdvancedAgent) else 0 for user in users}
    for conversation in conversations:
        user = conversation["user_id"]
        thread = f"{conversation['id']}:main"
        for message in conversation["turns"]:
            result = agent.reply(user, thread, message)
            tokens += result["agent_tokens"]
            prompts += result["prompt_tokens"]
        compactions += agent.compaction_count(thread)
        for index, item in enumerate(conversation["recall_questions"]):
            # Each recall question starts a fresh thread, so another question
            # cannot leak its own answer into the baseline's session memory.
            recall_thread = f"{conversation['id']}:recall:{index}"
            result = agent.reply(user, recall_thread, item["question"])
            tokens += result["agent_tokens"]
            prompts += result["prompt_tokens"]
            expected = item["expected_contains"]
            recalls.append(recall_points(result["answer"], expected))
            quality.append(heuristic_quality(result["answer"], expected))
    growth = sum((agent.memory_file_size(user) if isinstance(agent, AdvancedAgent) else 0) - before[user] for user in users)
    return BenchmarkRow(agent_name, tokens, prompts,
                        round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
                        round(sum(quality) / len(quality), 3) if quality else 0.0,
                        growth, compactions)


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed), f"{row.recall_score:.3f}", f"{row.response_quality:.3f}", str(row.memory_growth_bytes), str(row.compactions)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    for title, filename in (("Standard Benchmark", "conversations.json"), ("Long-Context Stress Benchmark", "advanced_long_context.json")):
        conversations = load_conversations(config.data_dir / filename)
        # Fresh isolated profiles make repeat runs deterministic without
        # deleting a user's existing state/ directory.
        with tempfile.TemporaryDirectory(prefix="benchmark-", dir=config.state_dir) as work:
            baseline_config = replace(config, state_dir=Path(work) / "baseline")
            advanced_config = replace(config, state_dir=Path(work) / "advanced")
            rows = [
                run_agent_benchmark("Baseline", BaselineAgent(baseline_config, force_offline=True), conversations, baseline_config),
                run_agent_benchmark("Advanced", AdvancedAgent(advanced_config, force_offline=True), conversations, advanced_config),
            ]
        print(f"\n## {title}\n{format_rows(rows)}")


if __name__ == "__main__":
    main()
