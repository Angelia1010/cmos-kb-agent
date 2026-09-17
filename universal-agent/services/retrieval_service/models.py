# -*- coding: utf-8 -*-
"""retrieval 服务契约 — 请求/响应 Pydantic 模型。

请求:轻量检索契约(对照 kbagent_service 的灵犀对话格式,此处无对话上下文)
  query / region_code

响应:灵犀返回信封
  rtnCode / rtnMsg / object
  object 内容面向纯检索场景(召回明细 + 降级标记),不含答案生成字段。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── 请求 ────────────────────────────────────────────────────────────────────

class RetrievalRequest(BaseModel):
    """独立 Retrieval 服务的标准输入。"""
    query: str = Field(min_length=1, description="检索query")
    region_code: str = Field(
        default="000",
        description="区域编码,支持省份名或区号(如 福建/591/000),缺省 000 表示全国")
    # 多余字段直接拒绝,避免调用方拼错字段名被静默忽略
    model_config = ConfigDict(extra="forbid")


# ── 响应 ────────────────────────────────────────────────────────────────────

class RetrievalChunk(BaseModel):
    """召回片段 — 共享 Chunk 契约的 HTTP 白名单表示。

    与 Processing 服务的 ProcessedChunk 同源,仅暴露检索阶段产出的字段,
    不包含 Processing 后才有的 rerank_rank 等字段。
    """
    chunk_id: str = Field(description="知识片段ID")
    doc_id: str = Field(description="所属文档ID")
    doc_title: str = Field(description="文档标题")
    content: str = Field(description="原文内容")
    category: str = Field(description="知识分类")
    position: dict[str, Any] = Field(description="在文档中的位置")
    version: str = Field(description="知识版本")
    updated_at: str = Field(description="知识更新日期")
    score: float = Field(description="召回得分")
    source_chunk_ids: list[str] = Field(description="溯源片段ID列表")
    extra: dict[str, Any] = Field(description="扩展字段")


class RetrievalResponseObject(BaseModel):
    """object 层 — 纯检索业务载荷。"""
    request_id: str = Field(description="回传请求ID(优先取 X-Request-ID 头,缺省服务端生成)")
    trace_id: str = Field(description="检索智能体内部trace ID")
    outcome: Literal["success", "no_results", "degraded"] = Field(
        description="结果:success 正常召回;no_results 零召回;degraded 走兜底降级路径")
    degraded: bool = Field(description="是否降级兜底结果;true 时未经一体化流水线,请人工核实")
    recalled_count: int = Field(description="召回片段数")
    elapsed_ms: int = Field(description="端到端耗时(毫秒)")
    region_code: str = Field(description="本次检索使用的区域编码")
    keywords: list[str] = Field(default_factory=list, description="检索关键词(槽位提取结果)")
    chunks: list[RetrievalChunk] = Field(description="召回片段列表")


class RetrievalResponse(BaseModel):
    """灵犀返回信封。"""
    rtnCode: str = Field(default="0", description="返回码:0成功,非0见错误码表")
    rtnMsg: str = Field(default="success", description="返回消息")
    object: RetrievalResponseObject


# ── 错误码 ──────────────────────────────────────────────────────────────────

RTN_OK = "0"                # 成功(含降级结果,降级通过 object.outcome/degraded 表达)
RTN_BAD_REQUEST = "40001"   # 参数校验失败(缺 query / 字段类型错误 / 多余字段)
RTN_INTERNAL = "50001"      # 服务内部未预期异常
RTN_TIMEOUT = "50002"       # 端到端处理超时


def error_body(code: str, msg: str) -> dict:
    """错误响应体:object 为空对象(契约要求 object 必含)。"""
    return {"rtnCode": code, "rtnMsg": msg, "object": {}}
