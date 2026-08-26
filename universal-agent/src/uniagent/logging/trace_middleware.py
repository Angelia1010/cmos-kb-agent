"""TraceMiddleware — 将 LLM 调用与工具调用写入 AgentTrace。

两种机制并行工作：

1. **_TraceCallback** (BaseCallbackHandler)
   → 捕获 on_chat_model_start / on_llm_end / on_tool_start / on_tool_end
   → 通过 ``get_invoke_config()`` 自动注入到每次 ``agent.ainvoke()``
   → ContextVar 保证同一 async task 内正确读取到当前 AgentTrace

2. **_TraceLoopHook** (LoopHook)
   → 在 on_iteration_start 创建 IterationRecord（begin_iteration）
   → 在 on_iteration_end 完成 IterationRecord（end_iteration）
   → on_goal_achieved 写入验证通过结果
   → 通过 ``loop_hooks()`` 自动注册到循环引擎

per-middleware 耗时（MiddlewareEvent）由 ``loop.py`` 的 ``_invoke_agent``
在每个 before/after 调用前后直接写入 AgentTrace，无需本中间件参与。
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from uniagent.logging.trace import (
    LLMCallRecord,
    ToolCallRecord,
    get_current_trace,
)
from uniagent.middleware.base import Middleware
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.signals import HookResponse


def _fmt_content(content: Any) -> str:
    """统一处理 str / list（thinking 模型返回 block 列表）两种格式。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        t = block.get("type", "")
        if t == "thinking":
            parts.append(f"[thinking] {block.get('thinking', '')[:80]}…")
        elif t == "text":
            parts.append(block.get("text", ""))
        elif t == "tool_use":
            parts.append(f"[tool_use] {block.get('name')}")
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _trunc(s: str, limit: int) -> str:
    """截断字符串并追加省略号标记。"""
    return s if len(s) <= limit else s[:limit] + "…(截断)"


# ── LangChain 回调 ───────────────────────────────────────────────────────────

class _TraceCallback(BaseCallbackHandler):
    """将 LLM 推理 / 工具调用事件写入当前 IterationRecord。

    通过 ContextVar 读取 AgentTrace.current_iteration()，无需参数穿透。
    BaseCallbackHandler 的方法是同步的，但在 asyncio task 中调用时
    ContextVar 仍然有效（ContextVar 在 task 层面存储，而非协程帧层面）。
    """

    _PROMPT_MSG_LIMIT = 1000   # 每条 prompt 消息内容截断长度
    _RESPONSE_LIMIT = 2000     # response content 截断长度
    _IO_LIMIT = 500            # tool 输入/输出截断长度

    def __init__(self) -> None:
        super().__init__()
        self._pending_llm: LLMCallRecord | None = None
        # tool_call_id → ToolCallRecord（支持 ReAct 同轮多工具并发）
        self._pending_tools: dict[str, ToolCallRecord] = {}

    def _cur_iter(self):
        trace = get_current_trace()
        return trace.current_iteration() if trace else None

    # ── LLM 推理前 ──────────────────────────────────────────────────────────

    def on_chat_model_start(
        self, serialized: dict, messages: list, **kwargs: Any
    ) -> None:
        cur = self._cur_iter()
        if cur is None:
            return

        rec = LLMCallRecord(call_index=len(cur.llm_calls) + 1)

        # 解析第一批次的消息列表（非批量模式时只有一个批次）
        batch = messages[0] if messages else []
        for msg in batch:
            role = type(msg).__name__
            content = _fmt_content(getattr(msg, "content", ""))
            tc = getattr(msg, "tool_calls", None)
            entry: dict = {
                "role": role,
                "content": _trunc(content, self._PROMPT_MSG_LIMIT),
            }
            if tc:
                entry["tool_calls"] = [t["name"] for t in tc]
            rec.prompt_messages.append(entry)

        self._pending_llm = rec
        cur.llm_calls.append(rec)  # 先占位，on_llm_end 补全 response 和 tokens

    # ── LLM 推理后 ──────────────────────────────────────────────────────────

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        rec = self._pending_llm
        self._pending_llm = None
        if rec is None:
            return

        rec.finish()

        # 提取 response content + tool_calls
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None) or gen
                content = _fmt_content(getattr(msg, "content", ""))
                rec.response_content = _trunc(content, self._RESPONSE_LIMIT)
                tool_calls: list = getattr(msg, "tool_calls", []) or []
                rec.response_tool_calls = [
                    {"name": tc["name"], "args": tc.get("args", {})}
                    for tc in tool_calls
                ]

        # 提取 token 用量（兼容 OpenAI / Anthropic / Azure 多种格式）
        tu: dict = {}

        # 格式1：OpenAI 风格，llm_output.token_usage
        llm_out = getattr(response, "llm_output", {}) or {}
        tu = llm_out.get("token_usage", {}) or {}

        # 格式2：Anthropic 风格，generation_info.usage
        if not tu:
            for gen_list in response.generations:
                for gen in gen_list:
                    gi = getattr(gen, "generation_info", {}) or {}
                    tu = gi.get("usage", {}) or gi.get("usage_metadata", {}) or {}
                    if tu:
                        break
                if tu:
                    break

        rec.input_tokens = tu.get("input_tokens", 0) or tu.get("prompt_tokens", 0)
        rec.output_tokens = tu.get("output_tokens", 0) or tu.get("completion_tokens", 0)
        rec.total_tokens = (
            tu.get("total_tokens", 0) or (rec.input_tokens + rec.output_tokens)
        )

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        rec = self._pending_llm
        self._pending_llm = None
        if rec:
            rec.finish()  # 补全时间戳

    # ── 工具调用 ────────────────────────────────────────────────────────────

    def on_tool_start(
        self, serialized: dict, input_str: str, **kwargs: Any
    ) -> None:
        cur = self._cur_iter()
        if cur is None:
            return

        rec = ToolCallRecord(tool_name=serialized.get("name", "?"))
        rec.input = _trunc(str(input_str), self._IO_LIMIT)

        # tool_call_id 在 kwargs 中（ReAct 框架传入）
        tc_id = str(kwargs.get("tool_call_id", f"_idx_{len(cur.tool_calls)}"))
        self._pending_tools[tc_id] = rec
        cur.tool_calls.append(rec)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        tc_id = str(kwargs.get("tool_call_id", ""))
        rec = self._pending_tools.pop(tc_id, None)
        if rec:
            rec.output = _trunc(str(output), self._IO_LIMIT)
            rec.finish()

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        tc_id = str(kwargs.get("tool_call_id", ""))
        rec = self._pending_tools.pop(tc_id, None)
        if rec:
            rec.error = _trunc(str(error), 300)
            rec.finish()


# ── LoopHook ─────────────────────────────────────────────────────────────────

class _TraceLoopHook(LoopHook):
    """管理 IterationRecord 的生命周期，并写入验证结果。

    on_iteration_start → trace.begin_iteration()（初始化 IterationRecord）
    on_iteration_end   → trace.end_iteration()（完成时间戳 + 消息增量计算）
    on_goal_achieved   → 写入验证通过结果（GoalLoop 专用）
    on_budget_exhausted → 写入错误原因到 trace.error
    """

    name = "trace_loop_hook"

    async def on_iteration_start(
        self, iteration: int, state: dict[str, Any]
    ) -> HookResponse:
        trace = get_current_trace()
        if trace:
            trace.begin_iteration(
                iteration,
                msg_count=len(state.get("messages", [])),
            )
        return HookResponse()

    async def on_iteration_end(
        self,
        iteration: int,
        state: dict[str, Any],
        agent_output: dict[str, Any] | None,
    ) -> HookResponse:
        trace = get_current_trace()
        if trace:
            trace.end_iteration(msg_count=len(state.get("messages", [])))
        return HookResponse()

    async def on_goal_achieved(
        self, state: dict[str, Any], evidence: str
    ) -> None:
        trace = get_current_trace()
        if trace and trace.iterations:
            # on_goal_achieved 在 on_iteration_end 之后调用，迭代已完成
            trace.iterations[-1].verification = {
                "passed": True,
                "evidence": evidence,
            }

    async def on_budget_exhausted(
        self, state: dict[str, Any], reason: str
    ) -> None:
        trace = get_current_trace()
        if trace and not trace.error:
            trace.error = f"预算耗尽: {reason}"

    async def on_error(
        self, state: dict[str, Any], error: Exception
    ) -> HookResponse:
        trace = get_current_trace()
        if trace:
            if not trace.error:
                trace.error = str(error)
            # 若有未完成的迭代，强制结束它
            if trace._current_iter:
                trace.end_iteration(msg_count=len(state.get("messages", [])))
        return HookResponse()


# ── 公开中间件 ───────────────────────────────────────────────────────────────

class TraceMiddleware(Middleware):
    """Agent 执行全链路结构化追踪中间件。

    加入中间件链后，配合 ``agent_trace()`` 上下文管理器（或 ``run_traced()``），
    将从 Agent 创建到执行结束的所有信息写入同一个 ``AgentTrace`` JSON 对象：

    +-------------------+------------------------------------------+
    | 数据层            | 写入内容                                  |
    +===================+==========================================+
    | LLM 调用级        | 每次推理的完整 prompt / response /       |
    |                   | tool_calls / token 用量 / 耗时           |
    +-------------------+------------------------------------------+
    | 工具调用级        | 工具名 / 输入 / 输出 / 耗时 / 错误       |
    +-------------------+------------------------------------------+
    | 中间件级          | 每个中间件 before/after 的 action + 耗时  |
    |                   | （由 loop.py _invoke_agent 直接写入）     |
    +-------------------+------------------------------------------+
    | 迭代级            | 每轮开始/结束时间 / 新增消息数 / 验证结果 |
    +-------------------+------------------------------------------+
    | 顶层              | success / iterations_used / reason /     |
    |                   | token_usage 汇总                         |
    +-------------------+------------------------------------------+

    用法::

        from uniagent.logging import TraceMiddleware, run_traced

        loop = create_agent(
            model=model,
            tools=[...],
            middleware=[TraceMiddleware(), ...其他中间件...],
            goal="...",
            verifier=...,
            budget=Budget(...),
        )

        # 一行完成追踪
        result, trace = await run_traced(loop, messages, name="my_agent")
        logger.info("trace=%s", trace.to_json())
        save_to_monitor(trace.to_dict())  # 推送到监控系统

    注意：``TraceMiddleware`` 本身不需要放在特定位置，
    per-middleware 耗时由 loop.py 统一计时，不受自身位置影响。
    """

    name = "trace_middleware"

    def __init__(self) -> None:
        super().__init__()
        self._callback = _TraceCallback()
        self._hook = _TraceLoopHook()

    def get_invoke_config(self) -> dict[str, Any]:
        """注入 _TraceCallback，自动捕获 LLM 调用和工具调用。

        在 TurnLoop / GoalLoop 模式下由 ``_invoke_agent`` 自动调用，
        无需用户手动配置。
        """
        return {"callbacks": [self._callback]}

    def loop_hooks(self) -> list[Any]:
        """返回 _TraceLoopHook，管理 IterationRecord 的生命周期。

        被 ``factory.py`` 收集并注册到循环引擎的钩子列表中。
        """
        return [self._hook]
