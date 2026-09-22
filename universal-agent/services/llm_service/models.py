# -*- coding: utf-8 -*-
"""llm_397b 服务契约 — 请求/响应 Pydantic 模型。

请求:只携带用户问题 query。
响应:与本项目其他服务一致的返回信封
  rtnCode / rtnMsg / object
  object 为大模型裸预测载荷(预测文本 + 模型名 + 耗时)。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── 请求 ────────────────────────────────────────────────────────────────────

class LlmRequest(BaseModel):
    """POST /llm_397b_api 请求体。"""
    query: str = Field(min_length=1, description="用户问题")


# ── 响应 ──────────────────────────────────────────────────────────────────

class LlmObject(BaseModel):
    """object 层 — 大模型裸预测载荷。"""
    requestId: str = Field(default="", description="服务端生成的请求ID(uuid4 hex),用于日志排查")
    answer: str = Field(default="", description="大模型预测 token 文本")
    model: str = Field(default="", description="实际调用的模型名")
    elapsedMs: int = Field(default=0, description="本次调用耗时(毫秒)")


class LlmResponse(BaseModel):
    """返回信封。"""
    rtnCode: str = Field(description="返回码:0成功,非0见错误码表")
    rtnMsg: str = Field(description="返回消息")
    object: LlmObject


# ── 错误码 ──────────────────────────────────────────────────────────────────

RTN_OK = "0"                # 成功
RTN_BAD_REQUEST = "40001"   # 参数校验失败
RTN_INTERNAL = "50001"      # 服务内部未预期异常(含大模型网关错误)
RTN_TIMEOUT = "50002"       # 大模型调用超时


def error_body(code: str, msg: str) -> dict:
    """错误响应体:object 为空载荷(契约要求 object 必含)。"""
    return {"rtnCode": code, "rtnMsg": msg, "object": {}}
