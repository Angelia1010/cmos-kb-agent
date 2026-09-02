"""Processing 服务边界：绑定 Workspace、调用固定编排器并构造安全响应。"""
from __future__ import annotations

import copy
from typing import Any

from kbagent.processing.agent import KnowledgeProcessingOrchestrator
from kbagent.shared.knowledge_processing.models import (
    ProcessingMeta,
    ProcessingWarning,
)
from kbagent.shared.models import Chunk
from kbagent.shared.workspace import RunWorkspace, workspace_scope

from .models import (
    ProcessedChunk,
    ProcessingRequest,
    ProcessingResponseObject,
    ProcessingStats,
    ProcessingWarningItem,
    TopCandidate,
)


_SAFE_WARNING_MESSAGES = {
    "rerank_model_error": "模型重排失败，已按现有规则降级",
    "rerank_timeout": "模型重排超时，已按现有规则降级",
    "rerank_invalid_json": "模型重排结果格式无效，已按现有规则降级",
    "rerank_invalid_id": "模型重排返回了无效编号，已忽略",
    "rerank_duplicate_id": "模型重排返回了重复编号，已去重",
    "rerank_incomplete": "模型重排结果不足，已按现有规则补位",
    "rerank_insufficient_candidates": "有效候选不足 3 条，已返回全部",
}


def _safe_warning(warning: ProcessingWarning) -> ProcessingWarningItem:
    return ProcessingWarningItem(
        code=warning.code,
        message=_SAFE_WARNING_MESSAGES.get(warning.code, "Processing 产生告警，请根据 code 和 field 排查"),
        source_index=warning.source_index,
        knowledge_id=warning.knowledge_id,
        field=warning.field,
    )


def _stats(meta: Any) -> ProcessingStats:
    if not isinstance(meta, ProcessingMeta):
        return ProcessingStats()
    return ProcessingStats(**meta.to_dict())


async def run_processing_request(
    request: ProcessingRequest,
    *,
    model: Any,
    request_id: str,
) -> ProcessingResponseObject:
    """执行一次请求；不共享 Workspace，也不返回 raw/metadata 等内部字段。"""
    ws = RunWorkspace(
        query=request.query,
        data={
            "retrieval_query": request.retrieval_query,
            "processing_context": copy.deepcopy(request.processing_context.model_dump()),
            "knowledge_candidates": copy.deepcopy(request.candidates),
        },
    )
    with workspace_scope(ws):
        top3 = await KnowledgeProcessingOrchestrator(model).run()
        meta = _stats(ws.data.get("processing_meta"))
        warnings = [
            _safe_warning(item)
            for item in ws.data.get("processing_warnings", [])
            if isinstance(item, ProcessingWarning)
        ]
        processed_chunks = ws.data.get("processed_chunks")
        if not isinstance(processed_chunks, list) or not all(
            isinstance(item, Chunk) for item in processed_chunks
        ):
            raise RuntimeError("Processing 未产生有效的 processed_chunks 工作区产物")

    top_rows = [
        TopCandidate(
            knowledge_id=item.knowledge_id or "",
            knowledge_name=item.name,
            retrieval_rank=item.retrieval_rank,
            retrieval_score=item.retrieval_score,
            rerank_rank=item.rerank_rank,
            content_md=item.content_md,
            included_atom_count=item.included_atom_count,
        )
        for item in top3
    ]
    chunk_rows = [
        ProcessedChunk(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            doc_title=item.doc_title,
            content=item.content,
            category=item.category,
            position=copy.deepcopy(item.position),
            version=item.version,
            updated_at=item.updated_at,
            score=item.score,
            source_chunk_ids=copy.deepcopy(item.source_chunk_ids),
            extra=copy.deepcopy(item.extra),
        )
        for item in processed_chunks
    ]
    if not top_rows:
        outcome = "no_valid_candidates"
    elif meta.degraded:
        outcome = "degraded"
    else:
        outcome = "success"

    return ProcessingResponseObject(
        request_id=request_id,
        trace_id=ws.tracer.trace_id,
        outcome=outcome,
        degraded=meta.degraded,
        elapsed_ms=ws.tracer.elapsed_ms(),
        top3_candidates=top_rows,
        processed_chunks=chunk_rows,
        processing_meta=meta,
        warnings=warnings,
    )
