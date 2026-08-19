"""Token 用量统计中间件，支持可选的预算硬限制。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from uniagent.middleware.base import Middleware
from uniagent.state.thread_state import ThreadState

logger = logging.getLogger(__name__)


class TokenUsageMiddleware(Middleware):
    """收集模型响应中的 token 用量统计数据。

    在每次模型调用后检查 ``AIMessage.response_metadata`` 或
    ``usage_metadata``，并将计数器累加到 ``state["token_usage"]`` 中。

    与循环引擎配合使用时，还会暴露 ``TokenBudgetHook``，
    用于将用量数据同步到预算系统以实现硬限制。
    """

    name = "token_usage"

    def __init__(self, *, budget: Any | None = None) -> None:
        self._total_prompt: int = 0
        self._total_completion: int = 0
        self._total_calls: int = 0
        self._budget = budget  # 可选的预算引用，用于硬限制
        # C4: 记录上次处理到的消息位置，避免 O(n²) 重复计算
        self._last_msg_count: int = 0

    async def after_agent(self, state: ThreadState) -> ThreadState | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        new_prompt = 0
        new_completion = 0
        counted = 0

        # C4: 只统计新增消息，避免重复累加
        for msg in messages[self._last_msg_count:]:
            if not isinstance(msg, AIMessage):
                continue
            usage = _extract_usage(msg)
            if usage:
                new_prompt += usage.get("prompt_tokens", 0)
                new_completion += usage.get("completion_tokens", 0)
                counted += 1

        self._last_msg_count = len(messages)

        if counted == 0:
            return None

        self._total_prompt += new_prompt
        self._total_completion += new_completion
        self._total_calls += counted

        usage_update = {
            "prompt_tokens": self._total_prompt,
            "completion_tokens": self._total_completion,
            "total_tokens": self._total_prompt + self._total_completion,
            "model_calls": self._total_calls,
        }

        # 若预算可用，同步数据（硬限制）
        if self._budget is not None:
            self._budget.record_tokens(usage_update["total_tokens"])

        logger.debug("Token 用量：%s", usage_update)
        return {"token_usage": usage_update}  # type: ignore[return-value]

    def loop_hooks(self) -> list[Any]:
        """暴露用于循环层预算同步的 TokenBudgetHook。"""
        if self._budget is not None:
            from uniagent.runtime.hooks import TokenBudgetHook
            return [TokenBudgetHook(self._budget)]
        return []


def _extract_usage(msg: AIMessage) -> dict[str, int] | None:
    """从 AIMessage 中提取用量字典。"""
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        um = msg.usage_metadata
        return {
            "prompt_tokens": getattr(um, "input_tokens", 0),
            "completion_tokens": getattr(um, "output_tokens", 0),
        }
    meta = getattr(msg, "response_metadata", None)
    if meta and "token_usage" in meta:
        return meta["token_usage"]
    return None
