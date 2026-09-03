"""Processing HTTP 请求与响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProcessingContextInput(BaseModel):
    """标准 Processing 请求上下文；字段缺失时由现有 Adapter 应用默认行为。"""

    model_config = ConfigDict(extra="forbid")

    region_id: str | None = None
    region_name: str | None = None
    channel_code: str | None = None
    request_time: str | None = None
    audience: str | None = None
    customer_type: str | None = None


class ProcessingRequest(BaseModel):
    """独立 Processing 服务的标准输入。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    processing_context: ProcessingContextInput
    candidates: list[dict[str, Any]]


class TopCandidate(BaseModel):
    knowledge_id: str
    knowledge_name: str
    retrieval_rank: int
    retrieval_score: float | None = None
    rerank_rank: int | None = None
    content_md: str
    included_atom_count: int


class ProcessedChunk(BaseModel):
    """共享 Chunk 契约的 HTTP 白名单表示。"""

    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    category: str
    position: dict[str, Any]
    version: str
    updated_at: str
    score: float
    source_chunk_ids: list[str]
    extra: dict[str, Any]


class ProcessingWarningItem(BaseModel):
    code: str
    message: str
    source_index: int | None = None
    knowledge_id: str | None = None
    field: str | None = None


class ProcessingStats(BaseModel):
    input_count: int = 0
    normalized_count: int = 0
    filtered_count: int = 0
    processed_count: int = 0
    rerank_eligible_count: int = 0
    top_count: int = 0
    warning_count: int = 0
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    stage_order: list[str] = Field(default_factory=list)


class ProcessingResponseObject(BaseModel):
    request_id: str
    trace_id: str
    model_mode: Literal["scripted", "llm"] = "scripted"
    outcome: Literal["success", "no_valid_candidates", "degraded"]
    degraded: bool
    elapsed_ms: int
    top3_candidates: list[TopCandidate]
    processed_chunks: list[ProcessedChunk]
    processing_meta: ProcessingStats
    warnings: list[ProcessingWarningItem]


class ProcessingResponse(BaseModel):
    rtnCode: str = "0"
    rtnMsg: str = "success"
    object: ProcessingResponseObject


class ProcessingErrorResponse(BaseModel):
    rtnCode: str
    rtnMsg: str
    request_id: str
    object: None = None


RTN_BAD_REQUEST = "40001"
RTN_INTERNAL = "50001"
RTN_TIMEOUT = "50002"
