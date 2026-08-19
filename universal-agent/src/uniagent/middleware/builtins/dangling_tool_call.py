"""检测并修补悬空工具调用（有 tool_calls 但无对应 ToolMessage 的 AIMessage）。"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from uniagent.middleware.base import Middleware
from uniagent.state.thread_state import ThreadState

logger = logging.getLogger(__name__)


class DanglingToolCallMiddleware(Middleware):
    """当 AIMessage 存在无对应 ToolMessage 的 tool_calls 时修补消息列表。

    此操作可防止因对话历史中缺少工具结果而引发的 LLM API 错误
    （常见于中断或部分执行后的场景）。
    """

    name = "dangling_tool_call"

    async def before_agent(self, state: ThreadState) -> ThreadState | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        patched = _patch_dangling(messages)
        if patched is not messages:
            return {"messages": patched}  # type: ignore[return-value]
        return None


def _patch_dangling(messages: list[Any]) -> list[Any]:
    """扫描消息列表，为孤立的 tool_calls 插入合成 ToolMessage。"""
    # 收集所有已有对应 ToolMessage 的 tool_call ID
    answered: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            answered.add(msg.tool_call_id)

    # 找出孤立的 tool_calls
    orphans: list[tuple[int, list[dict[str, Any]]]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            missing = [
                tc for tc in msg.tool_calls if tc.get("id") not in answered
            ]
            if missing:
                orphans.append((i, missing))

    if not orphans:
        return messages

    logger.warning(
        "正在修补 %d 个悬空工具调用，涉及 %d 条消息。",
        sum(len(tcs) for _, tcs in orphans),
        len(orphans),
    )

    # 构建新消息列表，在 AI 消息之后插入合成 ToolMessage
    result: list[Any] = []
    orphan_map = dict(orphans)
    for i, msg in enumerate(messages):
        result.append(msg)
        if i in orphan_map:
            for tc in orphan_map[i]:
                # M6: 确保 tool_call_id 非空，某些 LLM API 要求非空值
                tool_call_id = tc.get("id") or f"synth_{uuid.uuid4().hex[:8]}"
                result.append(
                    ToolMessage(
                        content="[工具调用未执行 — 结果不可用]",
                        tool_call_id=tool_call_id,
                        name=tc.get("name", "unknown"),
                    )
                )
    return result
