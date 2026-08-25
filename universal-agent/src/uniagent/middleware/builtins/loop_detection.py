"""检测重复的工具调用模式并中断无限循环。

在两个层级上运行：
- Agent 节点层：注入警告 HumanMessage（软控制，依赖 LLM 响应）
- 循环层：通过 LoopHook 发出 LoopSignal.BREAK（硬控制，无条件终止）
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from uniagent.middleware.base import Middleware
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.signals import HookResponse, LoopSignal
from uniagent.state.thread_state import ThreadState

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW = 6
_DEFAULT_HARD_LIMIT = 3


class LoopDetectionMiddleware(Middleware):
    """检测重复的 tool_call 模式并注入警告或强制停止。

    Agent 节点层：注入 HumanMessage 警告（软控制，依赖 LLM 响应）。
    循环层：达到 hard_limit 后发出 LoopSignal.BREAK（硬控制，无条件终止）。
    """

    name = "loop_detection"

    def __init__(
        self,
        *,
        window_size: int = _DEFAULT_WINDOW,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
    ) -> None:
        self._window_size = window_size
        self._hard_limit = hard_limit
        self._repeat_count: int = 0
        self._last_sig: str = ""
        # C5: 标志位替代在 before_agent 中重置计数器，确保 on_iteration_end 可检测到
        self._hard_stop_triggered: bool = False

    async def before_agent(self, state: ThreadState) -> ThreadState | None:
        """检查最新 AIMessage 的 tool_calls 是否与上一次相同。

        - 重复次数 >= 2：记录 INFO 警告日志
        - 重复次数 >= hard_limit：注入 HumanMessage 警告并设置硬停止标志
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            self._repeat_count = 0
            return None

        sig = _signature(last.tool_calls)

        if sig == self._last_sig:
            self._repeat_count += 1
        else:
            self._repeat_count = 1
            self._last_sig = sig

        if self._repeat_count >= self._hard_limit:
            logger.warning(
                "循环强制停止：工具模式 %r 已重复 %d 次。",
                sig,
                self._repeat_count,
            )
            # C5: 设置标志位而非直接重置，让 loop_hooks 的 on_iteration_end 能检测到
            self._hard_stop_triggered = True
            warning = HumanMessage(
                content=(
                    "[系统] 您似乎陷入了重复调用相同工具的循环中。"
                    "请停止并尝试不同的方法，或说明您正在尝试完成的任务。"
                )
            )
            return {"messages": messages + [warning]}  # type: ignore[return-value]

        if self._repeat_count >= 2:
            logger.info("循环警告：工具模式 %r 已重复 %d 次。", sig, self._repeat_count)

        return None

    def loop_hooks(self) -> list[Any]:
        """返回循环层硬停止钩子。

        钩子在 on_iteration_end 检测 _hard_stop_triggered 标志，
        若已设置则发出 LoopSignal.BREAK 强制终止 GoalLoop/TurnLoop。
        """
        mw = self

        class _LoopDetectionHook(LoopHook):
            name = "loop_detection_hard"

            async def on_iteration_end(
                self_hook,
                iteration: int,
                state: dict[str, Any],
                agent_output: dict[str, Any] | None,
            ) -> HookResponse:
                # C5: 使用标志位检查，检测后立即重置防止重复触发
                if mw._hard_stop_triggered:
                    mw._hard_stop_triggered = False
                    mw._repeat_count = 0
                    return HookResponse(
                        signal=LoopSignal.BREAK,
                        message=(
                            f"检测到循环：相同工具模式已重复 "
                            f"{mw._hard_limit} 次，强制停止。"
                        ),
                    )
                return HookResponse()

        return [_LoopDetectionHook()]


def _signature(tool_calls: list[dict[str, Any]]) -> str:
    """从 tool_calls 列表生成可哈希的特征签名。

    H6: 使用 json.dumps 序列化所有参数类型，避免复杂参数被忽略导致签名退化。
    """
    parts = []
    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("args", {})
        # H6: json.dumps 确保所有类型参数都参与哈希，而非仅基本类型
        arg_str = json.dumps(args, sort_keys=True, default=str)
        parts.append(f"{name}({arg_str})")
    return "|".join(sorted(parts))
