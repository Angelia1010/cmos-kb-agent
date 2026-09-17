# -*- coding: utf-8 -*-
"""answer 测试服务契约 — 请求/响应 Pydantic 模型。

请求:直接携带 query + 知识片段 chunks(即检索/处理阶段的输出),
     只跑 kbagent.answer 子智能体(片段定位+精选 → LLM 组织话术 → 批量一致性校验)。

响应:与 kbagent_service 一致的返回信封
  rtnCode / rtnMsg / object
  object 面向答案生成环节(话术 + 办理建议 + 可用性 + 引用文档相关度/关键片段/原文 + 全链路 trace)。
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

class SourceItem(BaseModel):
    """引用文档。chunkId 全链路可溯源。"""
    chunkId: str = Field(description="知识片段ID")
    docId: str = Field(default="", description="所属文档ID")
    docTitle: str = Field(description="文档标题")
    relevance: int = Field(default=0, description="相关度 0-100,最相关一篇=100")
    keyFragment: str = Field(
        default="", description="该文档最能回答用户问题的原文逐字片段(可能为空)")
    content: str = Field(default="", description="整篇文档原文")
    updatedAt: str = Field(description="知识更新日期")
    stale: bool = Field(description="是否疑似过旧(超溯源天数或日期非法)")


class UsabilityInfo(BaseModel):
    """坐席视角的话术可用性判定(LLM 自评 + 确定性规则纠偏)。"""
    level: str = Field(default="", description="directly_usable / verify_first / not_usable")
    reasons: List[str] = Field(default_factory=list, description="判定依据")
    uncovered: List[str] = Field(default_factory=list, description="未覆盖方面")


class AnswerObject(BaseModel):
    """object 层 — 答案生成业务载荷。"""
    requestId: str = Field(description="回传请求ID")
    sessionId: str = Field(description="回传对话ID")
    traceId: str = Field(description="答案子智能体内部 trace ID")
    requestArrivedTime: str = Field(description="收到请求时间,格式 yyyy-MM-dd HH:mm:ss.SSS")
    elapsedMs: int = Field(description="答案生成耗时(毫秒,含一致性校验)")
    script: str = Field(default="", description="可直接念给用户的口语化话术")
    handlingSuggestion: str = Field(default="", description="办理建议")
    usability: Optional[UsabilityInfo] = Field(default=None, description="话术可用性判定")
    sources: List[SourceItem] = Field(description="引用文档列表(按相关度降序)")
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
