"""循环引擎 — 目标驱动与基于轮次的迭代控制器。

这是核心运行时抽象，封装 LangGraph 智能体并通过验证、
预算强制执行和基于钩子的生命周期管理来驱动迭代。

两种循环类型：

- ``TurnLoop``：简单的 N 次迭代循环，支持可选验证。
- ``GoalLoop``：目标驱动循环，包含强制验证和停止条件 —
  这是自主智能体工作流的主要抽象。
"""

from __future__ import annotations

import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from uniagent.runtime.budget import Budget
from uniagent.runtime.hooks import LoopHook, ProgressLogHook, TokenBudgetHook
from uniagent.runtime.signals import HookResponse, LoopResult, LoopSignal
from uniagent.verification.verifier import Verifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 基础循环
# ---------------------------------------------------------------------------

class BaseLoop(ABC):
    """所有循环类型的抽象基类。"""

    def __init__(
        self,
        agent: Any,
        *,
        hooks: Sequence[LoopHook] | None = None,
        budget: Budget | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._agent = agent
        self._budget = budget or Budget()
        self._hooks = list(hooks) if hooks else [ProgressLogHook()]
        self._config = config or {}

        # 自动添加 token 预算同步钩子
        if not any(isinstance(h, TokenBudgetHook) for h in self._hooks):
            self._hooks.append(TokenBudgetHook(self._budget))

    @abstractmethod
    async def run(
        self,
        input_messages: list[dict[str, Any]] | None = None,
        *,
        thread_id: str = "default",
        initial_state: dict[str, Any] | None = None,
    ) -> LoopResult:
        """执行循环直至完成。"""

    async def _invoke_agent(
        self,
        state: dict[str, Any],
        *,
        thread_id: str,
    ) -> dict[str, Any]:
        """执行内部智能体一轮推理。

        (KB 适配) 中间件链在此显式执行:before_agent 正序、after_agent 逆序。
        原实现仅把链挂在 agent 对象上从未调用(依赖已不存在的 langgraph 内部接口)。

        修复 H1: before_agent 前浅拷贝 state，全部成功后才应用。
        修复 C3: 支持中间件的 handle_invoke_error 方法处理智能体调用异常。

        追踪集成：若 ContextVar 中存在 AgentTrace，则在每个中间件的
        before_agent / after_agent 前后记录 MiddlewareEvent（名称、动作、耗时）。
        追踪失败不影响主流程（异常静默忽略）。
        """
        chain = getattr(self._agent, "_uniagent_middleware", None) or []

        # ── 读取当前追踪上下文（无则静默跳过）──
        try:
            from uniagent.logging.trace import MiddlewareEvent, get_current_trace
            _trace = get_current_trace()
            _cur_iter = _trace.current_iteration() if _trace else None
        except Exception:
            _trace = None
            _cur_iter = None

        # H1: 浅拷贝 state，确保中间件链部分失败时不污染原 state
        state_snapshot = {**state}
        try:
            for mw in chain:
                _t0 = time.monotonic()
                patch = await mw.before_agent(state)
                _dur = (time.monotonic() - _t0) * 1000
                # 写入 MiddlewareEvent（追踪集成）
                if _cur_iter is not None:
                    try:
                        _cur_iter.middleware_events.append(MiddlewareEvent(
                            phase="before",
                            middleware=mw.name or type(mw).__name__,
                            action="patched" if patch else "noop",
                            patch_keys=list(patch.keys()) if patch else [],
                            duration_ms=_dur,
                        ))
                    except Exception:
                        pass
                if patch:
                    state.update(patch)
        except Exception:
            state.clear()
            state.update(state_snapshot)
            raise

        # 从中间件链收集 callbacks（如 LLMLoggingMiddleware 注入的 VerboseCallback）
        extra_callbacks: list = []
        for mw in chain:
            extra_cfg = mw.get_invoke_config()
            extra_callbacks.extend(extra_cfg.get("callbacks", []))

        invoke_config: dict = {"configurable": {"thread_id": thread_id}}
        if extra_callbacks:
            invoke_config["callbacks"] = extra_callbacks

        # C3: 智能体调用异常时，让具备 handle_invoke_error 的中间件处理
        try:
            result = await self._agent.ainvoke(state, config=invoke_config)
        except Exception as exc:
            for mw in chain:
                handler = getattr(mw, "handle_invoke_error", None)
                if handler is not None:
                    result = await handler(state, exc)
                    break
            else:
                raise

        for mw in reversed(chain):
            _t0 = time.monotonic()
            patch = await mw.after_agent(result)
            _dur = (time.monotonic() - _t0) * 1000
            # 写入 MiddlewareEvent（追踪集成）
            if _cur_iter is not None:
                try:
                    _cur_iter.middleware_events.append(MiddlewareEvent(
                        phase="after",
                        middleware=mw.name or type(mw).__name__,
                        action="patched" if patch else "noop",
                        patch_keys=list(patch.keys()) if patch else [],
                        duration_ms=_dur,
                    ))
                except Exception:
                    pass
            if patch:
                result.update(patch)
        return result

    async def _run_hooks(
        self,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> HookResponse:
        """执行指定生命周期方法的所有钩子。

        返回第一个非 CONTINUE 的响应；若所有钩子均通过则返回 CONTINUE。
        H2: 应用钩子返回的 state_patch。
        """
        # H2: 累积所有钩子的 state_patch
        accumulated_patch: dict[str, Any] = {}
        for hook in self._hooks:
            handler = getattr(hook, method, None)
            if handler is None:
                continue
            resp = await handler(*args, **kwargs)
            if isinstance(resp, HookResponse):
                # H2: 应用 state_patch（如果存在）
                if resp.state_patch:
                    accumulated_patch.update(resp.state_patch)
                if resp.signal != LoopSignal.CONTINUE:
                    logger.info(
                        "钩子 %s.%s 返回 %s：%s",
                        hook.name, method, resp.signal.name, resp.message,
                    )
                    # 将累积的 patch 合并到返回的响应中
                    if accumulated_patch:
                        merged_patch = dict(accumulated_patch)
                        if resp.state_patch:
                            merged_patch.update(resp.state_patch)
                        resp = HookResponse(
                            signal=resp.signal,
                            message=resp.message,
                            state_patch=merged_patch,
                        )
                    return resp
        if accumulated_patch:
            return HookResponse(state_patch=accumulated_patch)
        return HookResponse()

    async def _notify_hooks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """向所有钩子发送即发即忘通知（不期待返回信号）。

        H3: 单个钩子异常不中断其他钩子的执行。
        """
        for hook in self._hooks:
            handler = getattr(hook, method, None)
            if handler:
                try:
                    await handler(*args, **kwargs)
                except Exception as exc:
                    logger.warning(
                        "通知钩子 %s.%s 时出错：%s",
                        getattr(hook, "name", type(hook).__name__),
                        method,
                        exc,
                    )


# ---------------------------------------------------------------------------
# TurnLoop — 基于迭代次数的简单循环
# ---------------------------------------------------------------------------

class TurnLoop(BaseLoop):
    """最多运行 N 次迭代的简单循环。

    无强制验证 — 仅重复运行智能体，直到迭代预算耗尽或
    某个钩子发出 BREAK 信号。
    适用于对话型智能体和简单任务流水线。
    """

    async def run(
        self,
        input_messages: list[dict[str, Any]] | None = None,
        *,
        thread_id: str = "default",
        initial_state: dict[str, Any] | None = None,
    ) -> LoopResult:
        state = initial_state or {}
        if input_messages:
            state["messages"] = input_messages

        for i in range(self._budget.config.max_iterations or 100):
            # 预算检查
            signal, reason = self._budget.check()
            if signal == LoopSignal.BREAK:
                await self._notify_hooks("on_budget_exhausted", state, reason)
                return LoopResult(
                    success=False, iterations=i, reason=reason, final_state=state
                )

            # 迭代前钩子
            resp = await self._run_hooks("on_iteration_start", i, state)
            # H2: 应用钩子的 state_patch
            if resp.state_patch:
                state.update(resp.state_patch)
            if resp.signal == LoopSignal.BREAK:
                return LoopResult(
                    success=False, iterations=i, reason=resp.message, final_state=state
                )

            # 运行智能体
            # C1: 初始化 result 防止异常路径 UnboundLocalError
            result = None
            try:
                result = await self._invoke_agent(state, thread_id=thread_id)
                state.update(result)
            except Exception as exc:
                resp = await self._run_hooks("on_error", state, exc)
                if resp.signal == LoopSignal.BREAK:
                    return LoopResult(
                        success=False, iterations=i + 1,
                        reason=f"错误：{exc}", final_state=state,
                    )
                # on_error 返回 CONTINUE 时，跳过本轮后续步骤
                self._budget.record_iteration()
                continue

            self._budget.record_iteration()

            # 迭代后钩子
            resp = await self._run_hooks("on_iteration_end", i, state, result)
            if resp.signal == LoopSignal.BREAK:
                return LoopResult(
                    success=True, iterations=i + 1,
                    reason=resp.message, final_state=state,
                )

        return LoopResult(
            success=False,
            iterations=self._budget.iterations_used,
            reason="已达最大迭代次数",
            final_state=state,
        )


# ---------------------------------------------------------------------------
# GoalLoop — 验证驱动的自主循环
# ---------------------------------------------------------------------------

class GoalLoop(BaseLoop):
    """包含强制验证和停止条件的目标驱动循环。

    自主智能体工作流的主要抽象。每次迭代执行步骤：

    1. 检查预算限制（硬性停止）。
    2. 触发 ``on_iteration_start`` 钩子。
    3. 若需要，将目标注入为系统/用户消息。
    4. 调用内部智能体。
    5. 触发 ``on_iteration_end`` 钩子。
    6. 运行 ``Verifier`` 检查目标是否达成。
    7. 若验证通过 → 触发 ``on_goal_achieved`` → 返回成功。
    8. 若未通过 → 根据预算决定继续或放弃。

    支持棘轮模式：每次迭代后保存检查点，确保进度不丢失。
    """

    def __init__(
        self,
        agent: Any,
        *,
        goal: str,
        verifier: Verifier,
        hooks: Sequence[LoopHook] | None = None,
        budget: Budget | None = None,
        config: dict[str, Any] | None = None,
        verify_every: int = 1,
        inject_goal: bool = True,
    ) -> None:
        super().__init__(agent, hooks=hooks, budget=budget, config=config)
        self._goal = goal
        self._verifier = verifier
        self._verify_every = max(1, verify_every)
        self._inject_goal = inject_goal

    async def run(
        self,
        input_messages: list[dict[str, Any]] | None = None,
        *,
        thread_id: str = "default",
        initial_state: dict[str, Any] | None = None,
    ) -> LoopResult:
        state = initial_state or {}
        if input_messages:
            state["messages"] = input_messages

        # 在第一次迭代时将目标注入为系统消息
        if self._inject_goal:
            goal_msg = SystemMessage(
                content=(
                    f"[目标] 您的任务目标：\n{self._goal}\n\n"
                    "请逐步朝该目标推进。"
                    "每一步结束后，请评估当前进度。"
                )
            )
            msgs = state.get("messages", [])
            state["messages"] = [goal_msg] + msgs

        last_checkpoint_state: dict[str, Any] | None = None

        for i in range(self._budget.config.max_iterations or 100):
            # ── 1. 预算检查 ──
            signal, reason = self._budget.check()
            if signal == LoopSignal.BREAK:
                await self._notify_hooks("on_budget_exhausted", state, reason)
                return LoopResult(
                    success=False, iterations=i, reason=reason, final_state=state
                )

            # ── 2. 迭代前钩子 ──
            resp = await self._run_hooks("on_iteration_start", i, state)
            # H2: 应用钩子的 state_patch
            if resp.state_patch:
                state.update(resp.state_patch)
            if resp.signal == LoopSignal.BREAK:
                return LoopResult(
                    success=False, iterations=i, reason=resp.message, final_state=state
                )
            if resp.signal == LoopSignal.ROLLBACK and last_checkpoint_state:
                logger.info("正在回退至第 %d 次迭代的检查点", i)
                state = {**last_checkpoint_state}
                # C2: ROLLBACK 也需记录迭代，防止预算计数器与实际不同步
                self._budget.record_iteration()
                continue

            # ── 3. 调用智能体 ──
            result: dict[str, Any] | None = None
            try:
                result = await self._invoke_agent(state, thread_id=thread_id)
                state.update(result)
            except Exception as exc:
                logger.error("第 %d 次迭代智能体出错：%s", i, exc)
                resp = await self._run_hooks("on_error", state, exc)
                if resp.signal == LoopSignal.BREAK:
                    return LoopResult(
                        success=False, iterations=i + 1,
                        reason=f"错误：{exc}", final_state=state,
                    )
                if resp.signal == LoopSignal.RETRY:
                    # C2: RETRY 也需记录迭代
                    self._budget.record_iteration()
                    continue

            self._budget.record_iteration()

            # ── 4. 保存检查点（棘轮模式）──
            # M2: 深拷贝 messages 列表，防止后续修改污染检查点
            last_checkpoint_state = copy.deepcopy(state)

            # ── 5. 迭代后钩子 ──
            resp = await self._run_hooks("on_iteration_end", i, state, result)
            if resp.signal == LoopSignal.BREAK:
                return LoopResult(
                    success=False, iterations=i + 1,
                    reason=resp.message, final_state=state,
                )
            if resp.signal == LoopSignal.RETRY:
                # C2: on_iteration_end RETRY 无需再次 record（已在上方记录）
                continue

            # ── 6. 验证 ──
            if (i + 1) % self._verify_every == 0:
                try:
                    vr = await self._verifier.verify(self._goal, state)
                except Exception as exc:
                    logger.warning("验证器出错：%s", exc)
                    continue

                # 将验证结果写入当前迭代的追踪记录（若追踪上下文存在）
                try:
                    from uniagent.logging.trace import get_current_trace
                    _tr = get_current_trace()
                    if _tr and _tr.iterations:
                        _tr.iterations[-1].verification = {
                            "passed": vr.passed,
                            "evidence": vr.evidence or "",
                        }
                except Exception:
                    pass

                if vr.passed:
                    await self._notify_hooks(
                        "on_goal_achieved", state, vr.evidence
                    )
                    return LoopResult(
                        success=True,
                        iterations=i + 1,
                        reason="目标验证通过",
                        final_state=state,
                        evidence=vr.evidence,
                    )
                else:
                    # 注入验证反馈，以便智能体进行调整
                    # H4: 清理旧的验证反馈，防止 messages 无限膨胀
                    _FEEDBACK_PREFIX = "[验证失败]"
                    feedback = HumanMessage(
                        content=(
                            f"{_FEEDBACK_PREFIX} 目标尚未达成。\n"
                            f"依据：{vr.evidence}\n"
                            f"请继续朝目标推进：{self._goal}"
                        )
                    )
                    msgs = state.get("messages", [])
                    # 移除之前的验证反馈消息，只保留最新一条
                    cleaned = [
                        m for m in msgs
                        if not (
                            isinstance(m, HumanMessage)
                            and isinstance(getattr(m, "content", ""), str)
                            and getattr(m, "content", "").startswith(_FEEDBACK_PREFIX)
                        )
                    ]
                    state["messages"] = cleaned + [feedback]

        return LoopResult(
            success=False,
            iterations=self._budget.iterations_used,
            reason="已达最大迭代次数，目标未完成",
            final_state=state,
        )
