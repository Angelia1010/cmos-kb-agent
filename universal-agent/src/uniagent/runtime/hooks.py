"""循环级钩子系统 — 每次迭代的生命周期事件。"""

from __future__ import annotations

import logging
from typing import Any

from uniagent.runtime.protocols import LoopHookBase
from uniagent.runtime.signals import HookResponse, LoopSignal

logger = logging.getLogger(__name__)


class LoopHook(LoopHookBase):
    """循环级生命周期钩子的基类。

    H5: 继承 LoopHookBase 以统一跨层类型契约。
    与中间件（在智能体节点级别运行）不同，循环钩子运行于
    GoalLoop / TurnLoop 的 **迭代级别**，可发出硬性控制信号
    （BREAK / RETRY / ROLLBACK）。
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__

    async def on_iteration_start(
        self,
        iteration: int,
        state: dict[str, Any],
    ) -> HookResponse:
        """在每次迭代开始前调用，可阻止本次迭代执行。"""
        return HookResponse()

    async def on_iteration_end(
        self,
        iteration: int,
        state: dict[str, Any],
        agent_output: dict[str, Any] | None,
    ) -> HookResponse:
        """在每次迭代结束后调用，可触发 BREAK/RETRY/ROLLBACK 信号。"""
        return HookResponse()

    async def on_goal_achieved(
        self,
        state: dict[str, Any],
        evidence: str,
    ) -> None:
        """在验证器确认目标达成时调用。"""

    async def on_budget_exhausted(
        self,
        state: dict[str, Any],
        reason: str,
    ) -> None:
        """在任意预算限制触发时调用。"""

    async def on_error(
        self,
        state: dict[str, Any],
        error: Exception,
    ) -> HookResponse:
        """在迭代过程中发生未处理异常时调用。"""
        return HookResponse(signal=LoopSignal.BREAK, message=str(error))


class ProgressLogHook(LoopHook):
    """内置钩子，用于记录迭代进度日志。"""

    name = "progress_log"

    async def on_iteration_start(
        self,
        iteration: int,
        state: dict[str, Any],
    ) -> HookResponse:
        logger.info("[循环] 第 %d 次迭代开始", iteration + 1)
        return HookResponse()

    async def on_iteration_end(
        self,
        iteration: int,
        state: dict[str, Any],
        agent_output: dict[str, Any] | None,
    ) -> HookResponse:
        token_usage = state.get("token_usage", {})
        tokens = token_usage.get("total_tokens", "?")
        logger.info(
            "[循环] 第 %d 次迭代完成（累计 token 数：%s）",
            iteration + 1,
            tokens,
        )
        return HookResponse()

    async def on_goal_achieved(self, state: dict[str, Any], evidence: str) -> None:
        logger.info("[循环] 目标已达成。依据：%s", evidence[:200])

    async def on_budget_exhausted(self, state: dict[str, Any], reason: str) -> None:
        logger.warning("[循环] 预算已耗尽：%s", reason)


class TokenBudgetHook(LoopHook):
    """将智能体输出中的 token 使用量同步回 Budget 的钩子。"""

    name = "token_budget_sync"

    def __init__(self, budget: Any) -> None:
        self._budget = budget

    async def on_iteration_end(
        self,
        iteration: int,
        state: dict[str, Any],
        agent_output: dict[str, Any] | None,
    ) -> HookResponse:
        token_usage = state.get("token_usage", {})
        total = token_usage.get("total_tokens", 0)
        if total > 0:
            # 将最新累计值同步至预算（绝对值，非增量）
            self._budget.tokens_used = total
        return HookResponse()
