"""捕获工具异常并将其转换为 ToolMessage。

C3 修复：原 wrap_tool_call 方法从未被 BaseLoop._invoke_agent 调用，
改为 handle_invoke_error 方法，由 _invoke_agent 在智能体调用异常时调用。
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from uniagent.middleware.base import Middleware
from uniagent.state.thread_state import ThreadState

logger = logging.getLogger(__name__)


class ToolErrorHandlingMiddleware(Middleware):
    """捕获智能体调用异常，防止未处理的异常导致循环崩溃。

    失败时返回包含错误信息的状态，让 LLM 在下一轮决定如何继续。
    """

    name = "tool_error_handling"

    # 致命异常类型不应被此中间件捕获，应向上传播以触发降级等机制
    _FATAL_EXCEPTIONS = (
        TimeoutError, KeyboardInterrupt, SystemExit, MemoryError,
        ConnectionError, OSError,
    )

    async def handle_invoke_error(
        self, state: ThreadState, exc: Exception
    ) -> dict[str, Any]:
        """当 _invoke_agent 中 ainvoke 抛出异常时被调用。

        仅处理非致命的工具级异常，将其转换为 AIMessage 错误消息。
        致命异常（超时、连接错误等）会重新抛出以触发上层降级逻辑。
        """
        # 致命异常不处理，让上层（如 MainAgent._degrade）捕获
        if isinstance(exc, self._FATAL_EXCEPTIONS):
            raise exc

        tb = traceback.format_exception_only(type(exc), exc)
        error_msg = "".join(tb).strip()
        logger.error(
            "智能体调用出错 (%s)：%s",
            type(exc).__name__,
            exc,
        )
        messages = list(state.get("messages", []))
        messages.append(
            AIMessage(content=f"[错误] 工具执行失败：{error_msg}")
        )
        result = dict(state)
        result["messages"] = messages
        return result
