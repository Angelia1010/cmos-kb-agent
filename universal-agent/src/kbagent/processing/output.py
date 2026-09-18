"""Processing 最终知识结果到下游 Chunk 契约的边界适配。"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Mapping

from ..shared.knowledge_processing.models import ProcessedKnowledge
from ..shared.knowledge_processing.richtext import sanitize_title_text
from ..shared.models import Chunk


def top3_to_processed_chunks(
    candidates: Sequence[ProcessedKnowledge],
    source_chunks: Mapping[str, Chunk],
) -> list[Chunk]:
    """按 chunk_id 关联原 Chunk，仅替换 Markdown 内容和输出白名单 extra。"""
    chunks: list[Chunk] = []
    for candidate in candidates:
        if not candidate.chunk_id:
            raise ValueError("Top3 候选缺少 chunk_id，无法关联原 Retrieval Chunk")
        source_chunk = source_chunks.get(candidate.chunk_id)
        if source_chunk is None:
            raise ValueError(f"找不到 chunk_id={candidate.chunk_id} 对应的原 Retrieval Chunk")
        chunks.append(replace(
            source_chunk,
            doc_title=sanitize_title_text(candidate.name),
            content=candidate.content_md,
            extra={
                "processing": {
                    "rerank_rank": candidate.rerank_rank,
                },
            },
        ))
    return chunks
