"""Retrieval HTTP 请求与响应模型。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalRequest(BaseModel):
    """独立 Retrieval 服务的标准输入。

    region_code 支持省份名或区号(如 福建/591/000),缺省 "000" 表示全国。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    region_code: str = "000"


class RetrievalChunk(BaseModel):
    """共享 Chunk 契约的 HTTP 白名单表示。

    与 Processing 服务的 ProcessedChunk 同源,仅暴露检索阶段产出的字段,
    不包含 Processing 后才有的 rerank_rank 等字段。
    """

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


class RetrievalResponseObject(BaseModel):
    request_id: str
    trace_id: str
    outcome: Literal["success", "no_results", "degraded"]
    degraded: bool
    recalled_count: int
    elapsed_ms: int
    region_code: str
    keywords: list[str] = Field(default_factory=list)
    chunks: list[RetrievalChunk]


class RetrievalResponse(BaseModel):
    rtnCode: str = "0"
    rtnMsg: str = "success"
    object: RetrievalResponseObject


class RetrievalErrorResponse(BaseModel):
    rtnCode: str
    rtnMsg: str
    request_id: str
    object: None = None


RTN_BAD_REQUEST = "40001"
RTN_INTERNAL = "50001"
RTN_TIMEOUT = "50002"
