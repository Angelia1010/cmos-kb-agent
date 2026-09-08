# -*- coding: utf-8 -*-
"""answer 测试服务契约 — 请求/响应 Pydantic 模型。

请求:直接携带 query + 知识片段 chunks(即检索/处理阶段的输出),
     只跑 kbagent.answer 子智能体(片段精选 → LLM 组织答案 → 逐句锚定校验)。

响应:与 kbagent_service 一致的返回信封
  rtnCode / rtnMsg / object
  object 面向答案生成环节(业务说明 + 办理建议 + 逐句锚定明细 + 知识溯源 + 全链路 trace)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── 请求 ────────────────────────────────────────────────────────────────────

class ChunkIn(BaseModel):
    """知识片段(对应内部 kbagent.shared.models.Chunk)。

    通常来自检索+处理阶段的输出;测试时可手工构造。
    """
    chunkId: str = Field(min_length=1, description="知识片段ID,答案引用的锚点")
    docId: str = Field(min_length=1, description="所属文档ID(同文档最多精选 2 个片段)")
    docTitle: str = Field(min_length=1, description="文档标题")
    content: str = Field(min_length=1, description="片段正文")
    category: str = Field(default="", description="业务类目,如 套餐/资费")
    position: Dict[str, Any] = Field(default_factory=dict, description="位置信息(可选)")
    version: str = Field(default="v1.0", description="文档版本")
    updatedAt: str = Field(default="", description="知识更新日期,格式 yyyy-MM-dd;空/非法按疑似过旧处理")
    score: float = Field(default=0.0, description="检索得分(按输入顺序精选,得分仅作展示参考)")


class AnswerParams(BaseModel):
    """请求 params 层。"""
    appId: str = Field(min_length=1, description="调用方应用ID")
    requestId: str = Field(min_length=1, description="请求ID")
    sessionId: str = Field(default="", description="对话ID(可选,仅回传)")
    query: str = Field(min_length=1, description="用户问题")
    chunks: List[ChunkIn] = Field(min_length=1, description="候选知识片段(建议已排序,取前 4 个精选)")


class AnswerRequest(BaseModel):
    params: AnswerParams


# ── 响应 ────────────────────────────────────────────────────────────────────

class SentenceItem(BaseModel):
    """答案句子(锚定校验后保留的句子)。"""
    text: str = Field(description="句子文本")
    citations: List[str] = Field(description="引用的知识片段ID")
    hardFact: bool = Field(description="是否硬事实(资费/办理条件等)")
    anchored: bool = Field(description="锚定校验是否通过;false 且未被删除时带 note")
    note: str = Field(description="备注;软性表述锚定失败时为『建议核实』")


class SourceItem(BaseModel):
    """知识来源。chunkId 全链路可溯源。"""
    chunkId: str = Field(description="知识片段ID")
    docTitle: str = Field(description="文档标题")
    snippet: str = Field(description="原文摘录")
    updatedAt: str = Field(description="知识更新日期")
    stale: bool = Field(description="是否疑似过旧(超溯源天数或日期非法)")


class AnswerObject(BaseModel):
    """object 层 — 答案生成业务载荷。"""
    requestId: str = Field(description="回传请求ID")
    sessionId: str = Field(description="回传对话ID")
    traceId: str = Field(description="答案子智能体内部 trace ID")
    requestArrivedTime: str = Field(description="收到请求时间,格式 yyyy-MM-dd HH:mm:ss.SSS")
    elapsedMs: int = Field(description="答案生成耗时(毫秒,含全部锚定校验)")
    businessExplanation: str = Field(description="业务说明")
    handlingSuggestion: str = Field(description="办理建议")
    renderedText: str = Field(description="完整答案文本(含知识来源),可直接展示")
    sentences: List[SentenceItem] = Field(description="保留的答案句子及锚定明细")
    sources: List[SourceItem] = Field(description="知识来源列表")
    trace: Optional[Dict[str, Any]] = Field(
        default=None, description="全链路 trace(badcase 回放);测试服务默认携带")


class AnswerResponse(BaseModel):
    """返回信封。"""
    rtnCode: str = Field(description="返回码:0成功,非0见错误码表")
    rtnMsg: str = Field(description="返回消息")
    object: AnswerObject


# ── 错误码 ──────────────────────────────────────────────────────────────────

RTN_OK = "0"                # 成功
RTN_BAD_REQUEST = "40001"   # 参数校验失败 / appId 不允许
RTN_INTERNAL = "50001"      # 服务内部未预期异常(含大模型网关错误)
RTN_TIMEOUT = "50002"       # 答案生成超时


def error_body(code: str, msg: str) -> dict:
    """错误响应体:object 为空对象(契约要求 object 必含)。"""
    return {"rtnCode": code, "rtnMsg": msg, "object": {}}
