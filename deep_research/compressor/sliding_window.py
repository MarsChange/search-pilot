from __future__ import annotations

from typing import Any


class SlidingWindowCompressor:
    def __init__(
        self,
        max_tokens: int = 12000,
        char_per_token: float = 3.5,
        min_recent_turns: int = 3,
    ) -> None:
        self.max_tokens = max_tokens
        self.char_per_token = char_per_token
        self.min_recent_turns = min_recent_turns
        self._stats: dict[str, Any] = {}

    def compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        max_chars = int(self.max_tokens * self.char_per_token)
        before = self._count_chars(messages)
        if before <= max_chars:
            self._stats = {"truncated": False, "before_chars": before, "after_chars": before}
            return messages
        system = [message for message in messages if message.get("role") == "system"]
        others = [message for message in messages if message.get("role") != "system"]
        kept = list(others)
        removed = 0
        while len(kept) > self.min_recent_turns and self._count_chars(system + kept) > max_chars:
            kept.pop(0)
            removed += 1
        if kept and self._count_chars(system + kept) > max_chars:
            last = dict(kept[-1])
            content = str(last.get("content", ""))
            keep_chars = max(500, max_chars - self._count_chars(system + kept[:-1]) - 100)
            last["content"] = content[:keep_chars] + "\n[CONTENT_TRUNCATED]"
            kept[-1] = last
        after = self._count_chars(system + kept)
        self._stats = {
            "truncated": True,
            "before_chars": before,
            "after_chars": after,
            "removed_turns": removed,
        }
        return system + kept

    def get_stats(self) -> dict[str, Any]:
        return dict(self._stats)

    @staticmethod
    def _count_chars(messages: list[dict[str, Any]]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)
