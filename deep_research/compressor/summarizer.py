from __future__ import annotations

from typing import Any


class LLMSummarizer:
    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    async def summarize_document(self, text: str, query: str, max_chars: int = 1600) -> str:
        if self.llm is None:
            return text[:max_chars]
        prompt = (
            "请在不引入新事实的前提下压缩以下材料，保留与查询相关的关键事实、数字、日期和来源。\n"
            f"查询：{query}\n\n材料：\n{text[:8000]}"
        )
        try:
            return await self.llm.complete(
                [
                    {"role": "system", "content": "你是上下文压缩助手，只输出压缩后的事实摘要。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
        except Exception:
            return text[:max_chars]

    async def summarize_documents(self, texts: list[str], query: str, max_chars: int = 2400) -> str:
        joined = "\n\n".join(texts)
        return await self.summarize_document(joined, query, max_chars=max_chars)
