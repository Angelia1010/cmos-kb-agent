# -*- coding: utf-8 -*-
"""灵犀平台契约 — 请求/响应 Pydantic 模型。

请求:灵犀对话格式
  params.appId / requestId / sessionId / userInfo / extInfo / conversations

响应:灵犀返回信封
  rtnCode / rtnMsg / object
  object 内容面向知识库检索问答场景(答案 + 知识溯源 + 降级标记)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── 请求 ────────────────────────────────────────────────────────────────────

class Conversation(BaseModel):
    """单条对话。conversations[0] 为最早一条;只含机器人(0)与用户(1)。"""
    role: Literal[0, 1] = Field(description="对话角色:0机器人 1用户")
    content: str = Field(min_length=1, description="对话内容(纯文本)")


class UserInfo(BaseModel):
    """用户信息。"""
    phone: str = Field(min_length=1, description="手机号(加密)")
    province: str = Field(min_length=1, description="省份")
    location: Optional[str] = Field(default=None, description="位置")
    areaCode: Optional[str] = Field(default=None, description="市县编码")


class Device(BaseModel):
    """设备信息(全部可选)。"""
    netType: Optional[str] = Field(default=None, description="网络")
    osType: Optional[str] = Field(default=None, description="系统")
    appVersion: Optional[str] = Field(default=None, description="app版本")


class Scene(BaseModel):
    """场景信息(全部可选)。"""
    brand: Optional[str] = Field(default=None, description="星级")
    initPageUrl: Optional[str] = Field(default=None, description="初始页面")
    latestPageUrl: Optional[str] = Field(default=None, description="当前页面")
    pageType: Optional[Literal[1, 4, 9, -1]] = Field(
        default=None, description="页面类型:1 H5,4 客户端,9 小程序,-1 未知")
    pageTitle: Optional[str] = Field(default=None, description="页面标题")


class ExtInfo(BaseModel):
    """其他信息。对象本身必填,内部字段全部可选。"""
    device: Optional[Device] = None
    scene: Optional[Scene] = None


class AskParams(BaseModel):
    """灵犀请求 params 层。"""
    appId: str = Field(min_length=1, description="灵犀应用ID")
    requestId: str = Field(min_length=1, description="请求ID")
    sessionId: str = Field(min_length=1, description="对话ID,唯一标识一通对话")
    userInfo: UserInfo
    extInfo: ExtInfo = Field(default_factory=ExtInfo)
    conversations: List[Conversation] = Field(min_length=1)


class AskRequest(BaseModel):
    params: AskParams


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
    stale: bool = Field(description="是否疑似过旧(超溯源天数)")


class UsabilityInfo(BaseModel):
    """坐席视角的话术可用性判定(LLM 自评 + 确定性规则纠偏)。"""
    level: str = Field(
        default="",
        description="directly_usable可直接使用 / verify_first核实后使用 / not_usable不可用转人工")
    reasons: List[str] = Field(default_factory=list, description="判定依据")
    uncovered: List[str] = Field(
        default_factory=list, description="用户问题中知识库未覆盖的方面")


class AnswerObject(BaseModel):
    """object 层 — 知识库检索问答业务载荷。"""
    requestId: str = Field(description="回传请求ID")
    sessionId: str = Field(description="回传对话ID")
    traceId: str = Field(description="知识库智能体内部trace ID")
    requestArrivedTime: str = Field(description="收到请求时间,格式 yyyy-MM-dd HH:mm:ss.SSS")
    degraded: bool = Field(description="是否降级兜底结果;true 时未经加工,请人工核实")
    elapsedMs: int = Field(description="端到端耗时(毫秒)")
    # ---- 坐席直接可用的两段内容 ----
    script: str = Field(default="", description="可直接念给用户的口语化话术")
    handlingSuggestion: str = Field(default="", description="办理建议")
    usability: Optional[UsabilityInfo] = Field(
        default=None, description="话术可用性判定")
    sources: List[SourceItem] = Field(description="引用文档列表(按相关度降序)")
    # 智能体内部执行 trace 事件列表(ts_ms/stage/event/payload),
    # 供前端演示页渲染"检索处理过程"时间线;
    # KB_SERVICE_EXPOSE_TRACE=0 时为空列表,灵犀老调用方不受影响
    processTrace: List[Dict[str, Any]] = Field(
        default_factory=list, description="智能体执行过程事件列表(调试透出)")


class AskResponse(BaseModel):
    """灵犀返回信封。"""
    rtnCode: str = Field(description="返回码:0成功,非0见错误码表")
    rtnMsg: str = Field(description="返回消息")
    object: AnswerObject


# ── 错误码 ──────────────────────────────────────────────────────────────────

RTN_OK = "0"                # 成功(含降级结果,降级通过 object.degraded 表达)
RTN_BAD_REQUEST = "40001"   # 参数校验失败 / appId 不允许 / 无有效用户消息
RTN_INTERNAL = "50001"      # 服务内部未预期异常
RTN_TIMEOUT = "50002"       # 端到端处理超时


def error_body(code: str, msg: str,
               trace_events: Optional[List[Dict[str, Any]]] = None) -> dict:
    """错误响应体:object 为空对象(契约要求 object 必含)。

    trace_events 非空时附带智能体已执行的 trace 事件,
    便于前端在超时/内部错误时也能展示"执行到哪一步了"。
    """
    obj: Dict[str, Any] = {}
    if trace_events:
        obj["processTrace"] = trace_events
    return {"rtnCode": code, "rtnMsg": msg, "object": obj}
