"""Processing 最终知识结果到下游 Chunk 契约的边界适配。"""
from __future__ import annotations

import copy
from collections.abc import Sequence

from ..shared.knowledge_processing.models import ProcessedKnowledge
from ..shared.models import Chunk


def top3_to_processed_chunks(
    candidates: Sequence[ProcessedKnowledge],
) -> list[Chunk]:
    """按现有 Top3 顺序将一条知识映射为一个完整 Markdown Chunk。"""
    chunks: list[Chunk] = []
    for candidate in candidates:
        if not candidate.knowledge_id:
            raise ValueError("Top3 候选缺少 knowledge_id，无法构造 processed Chunk")

        score_missing = candidate.retrieval_score is None
        chunks.append(Chunk(
            chunk_id=candidate.knowledge_id,
            doc_id=candidate.knowledge_id,
            doc_title=candidate.name,
            content=candidate.content_md,
            category="",
            updated_at="",
            score=0.0 if score_missing else candidate.retrieval_score,
            extra={
                "processing": {
                    "retrieval_rank": candidate.retrieval_rank,
                    "rerank_rank": candidate.rerank_rank,
                    "included_atom_count": candidate.included_atom_count,
                    "matched_atom_ids": copy.deepcopy(candidate.matched_atom_ids),
                    "source_routes": copy.deepcopy(candidate.source_routes),
                    "knowledge_type": candidate.knowledge_type,
                    "template_id": candidate.template_id,
                    "score_missing": score_missing,
                },
            },
        ))
    return chunks
