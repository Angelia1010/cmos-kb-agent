"""Agent 执行全链路结构化追踪 — AgentTrace + ContextVar。

设计原则：
  - AgentTrace 是一次 loop.run() 的完整 JSON 日志容器
  - ContextVar 保证 async 任务内全局透明传递，无需参数穿透
  - TraceMiddleware 负责写入 LLM 调用 + 工具调用事件
  - loop.py 的 _invoke_agent 负责写入每个中间件的耗时事件
  - GoalLoop.run() 在验证后将结果写入当前迭代记录

JSON 层次（to_dict() 输出）::

    {
      "trace_id": "uuid",
      "agent_name": "retrieval_agent",
      "thread_id": "req-001",
      "loop_type": "GoalLoop",
      "created_at": "2026-08-26T10:00:00.000+00:00",
      "finished_at": "...",
      "duration_ms": 3200.0,
      "success": true,
      "iterations_used": 2,
      "reason": "目标验证通过",
      "agent_config": {"middleware": [...], "budget": {...}},
      "iterations": [
        {
          "iteration": 1,
          "duration_ms": 1800.0,
          "new_messages_count": 4,
          "middleware_events": [
            {"phase": "before", "middleware": "SkillMiddleware",
             "action": "patched", "patch_keys": ["messages"], "duration_ms": 3.2}
          ],
          "llm_calls": [
            {
              "call_index": 1,
              "duration_ms": 1200.0,
              "prompt": {
                "message_count": 3,
                "messages": [{"role": "SystemMessage", "content": "..."}, ...]
              },
              "response": {
                "content": "我来查询...",
                "tool_calls": [{"name": "query_understanding", "args": {...}}]
              },
              "tokens": {"input": 800, "output": 50, "total": 850}
            }
          ],
          "tool_calls": [
            {"tool": "query_understanding", "duration_ms": 8.0,
             "input": "...", "output": "...", "error": ""}
          ],
          "verification": {"passed": false, "evidence": "top3得分不足"}
        }
      ],
      "token_usage": {"total_tokens": 1700},
      "error": ""
    }

典型用法::

    from uniagent.logging import TraceMiddleware, agent_trace, run_traced

    # 方式1：run_traced 一行搞定（自动提取 agent_config）
    result, trace = await run_traced(loop, messages, thread_id="t1", name="my_agent")
    print(trace.to_json())

    # 方式2：async with 上下文管理器（可精细控制）
    async with agent_trace("my_agent", loop_type="GoalLoop") as trace:
        result = await loop.run(messages, thread_id="t1")
        trace.finish(result)
    print(trace.to_json())
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# ── ContextVar（全局 async 上下文，任意协程内均可读写）─────────────────────
_CURRENT_TRACE: ContextVar[AgentTrace | None] = ContextVar(
    "uniagent_current_trace", default=None
)


def get_current_trace() -> AgentTrace | None:
    """返回当前 async 任务中的 AgentTrace；未设置时返回 None。"""
    return _CURRENT_TRACE.get()


def _ts() -> str:
    """当前 UTC 时间的 ISO 8601 字符串（毫秒精度）。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ── 数据结构（从小到大，按包含关系定义）────────────────────────────────────


class MiddlewareEvent:
    """单个中间件在 before_agent / after_agent 阶段的执行记录。

    由 ``loop.py`` 的 ``_invoke_agent`` 在每次 before/after 调用前后自动写入，
    用于监控中间件链的执行顺序和耗时。
    """

    __slots__ = ("phase", "middleware", "action", "patch_keys", "duration_ms")

    def __init__(
        self,
        *,
        phase: str,                         # "before" | "after"
        middleware: str,                    # 中间件 name 属性
        action: str = "noop",              # "patched" | "noop" | "error"
        patch_keys: list[str] | None = None,  # 返回 patch 包含的字段名
        duration_ms: float = 0.0,
    ) -> None:
        self.phase = phase
        self.middleware = middleware
        self.action = action
        self.patch_keys: list[str] = patch_keys or []
        self.duration_ms = duration_ms

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "middleware": self.middleware,
            "action": self.action,
            "patch_keys": self.patch_keys,
            "duration_ms": round(self.duration_ms, 1),
        }


class LLMCallRecord:
    """单次 LLM 推理的完整记录（prompt、response、token 用量、耗时）。

    由 ``TraceMiddleware`` 内置的 ``_TraceCallback`` 在 LangChain 回调中自动写入。
    """

    _CONTENT_LIMIT = 1000   # 每条 prompt 消息内容截断长度（字符）
    _RESPONSE_LIMIT = 2000  # response content 截断长度

    def __init__(self, call_index: int) -> None:
        self.call_index = call_index
        self.started_at = _ts()
        self.finished_at = ""
        self.duration_ms = 0.0
        self._t0 = time.monotonic()
        self.prompt_messages: list[dict] = []       # [{role, content[, tool_calls]}]
        self.response_content = ""
        self.response_tool_calls: list[dict] = []  # [{name, args}]
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def finish(self) -> None:
        self.finished_at = _ts()
        self.duration_ms = (time.monotonic() - self._t0) * 1000

    def to_dict(self) -> dict:
        return {
            "call_index": self.call_index,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 1),
            "prompt": {
                "message_count": len(self.prompt_messages),
                "messages": self.prompt_messages,
            },
            "response": {
                "content": self.response_content,
                "tool_calls": self.response_tool_calls,
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
            },
        }


class ToolCallRecord:
    """单次工具调用记录（名称、输入、输出、耗时、错误）。

    由 ``_TraceCallback`` 在 on_tool_start / on_tool_end 回调中自动写入。
    """

    _IO_LIMIT = 500  # 输入/输出内容截断长度

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.started_at = _ts()
        self.finished_at = ""
        self.duration_ms = 0.0
        self._t0 = time.monotonic()
        self.input = ""
        self.output = ""
        self.error = ""

    def finish(self) -> None:
        self.finished_at = _ts()
        self.duration_ms = (time.monotonic() - self._t0) * 1000

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 1),
            "input": self.input,
            "output": self.output,
            "error": self.error,
        }


class IterationRecord:
    """一轮循环迭代的完整记录。

    由 ``_TraceLoopHook`` 在 on_iteration_start / on_iteration_end 中创建和完成。
    """

    def __init__(self, iteration: int) -> None:
        self.iteration = iteration          # 1-based
        self.started_at = _ts()
        self.finished_at = ""
        self.duration_ms = 0.0
        self._t0 = time.monotonic()
        self._msg_baseline = 0             # 迭代开始时消息数，用于计算增量
        self.new_messages_count = 0
        self.middleware_events: list[MiddlewareEvent] = []
        self.llm_calls: list[LLMCallRecord] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.verification: dict | None = None   # {"passed": bool, "evidence": str}

    def finish(self, msg_count: int = 0) -> None:
        self.finished_at = _ts()
        self.duration_ms = (time.monotonic() - self._t0) * 1000
        self.new_messages_count = max(0, msg_count - self._msg_baseline)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 1),
            "new_messages_count": self.new_messages_count,
            "middleware_events": [e.to_dict() for e in self.middleware_events],
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "verification": self.verification,
        }


class AgentTrace:
    """一次 Agent ``loop.run()`` 的全量结构化执行日志。

    由 ``agent_trace()`` 上下文管理器或 ``run_traced()`` 辅助函数创建，
    通过 ContextVar 在整个 async 调用链中全局透明传递。

    各组件写入时机：
    - ``loop.py _invoke_agent``    → middleware_events（每个中间件的 before/after 耗时）
    - ``_TraceCallback``           → llm_calls + tool_calls（LLM 推理 + 工具调用）
    - ``_TraceLoopHook``           → 迭代 IterationRecord 生命周期
    - ``GoalLoop.run()``           → verification（每轮验证结果）
    - ``trace.finish(result)``     → 顶层摘要（success / reason / token_usage）
    """

    def __init__(
        self,
        trace_id: str | None = None,
        agent_name: str = "",
        thread_id: str = "",
        loop_type: str = "",
        agent_config: dict | None = None,
    ) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self.agent_name = agent_name
        self.thread_id = thread_id
        self.loop_type = loop_type
        self.agent_config: dict = agent_config or {}
        self.created_at = _ts()
        self.finished_at = ""
        self.duration_ms = 0.0
        self._t0 = time.monotonic()
        self.success: bool | None = None
        self.iterations_used = 0
        self.reason = ""
        self.evidence = ""
        self.token_usage: dict = {}
        self.error = ""
        self.iterations: list[IterationRecord] = []
        self._current_iter: IterationRecord | None = None

    # ── 迭代生命周期管理 ────────────────────────────────────────────────────

    def begin_iteration(self, n: int, msg_count: int = 0) -> IterationRecord:
        """开始新的迭代（n 为 0-based index，内部转为 1-based 存储）。"""
        rec = IterationRecord(iteration=n + 1)
        rec._msg_baseline = msg_count
        self._current_iter = rec
        self.iterations.append(rec)
        return rec

    def end_iteration(self, msg_count: int = 0) -> None:
        """完成当前迭代，计算 duration_ms 和 new_messages_count。"""
        if self._current_iter:
            self._current_iter.finish(msg_count=msg_count)
            self._current_iter = None

    def current_iteration(self) -> IterationRecord | None:
        """返回当前正在执行的迭代记录（若无则 None）。"""
        return self._current_iter

    # ── 完成追踪 ────────────────────────────────────────────────────────────

    def finish(self, result: Any | None = None) -> None:
        """完成追踪，从 LoopResult 提取顶层摘要信息。

        Parameters
        ----------
        result : LoopResult | None
            ``loop.run()`` 返回的 LoopResult 对象。None 时仅更新时间戳。
        """
        self.finished_at = _ts()
        self.duration_ms = (time.monotonic() - self._t0) * 1000
        if result is not None:
            self.success = getattr(result, "success", None)
            self.iterations_used = getattr(result, "iterations", 0)
            self.reason = getattr(result, "reason", "") or ""
            self.evidence = getattr(result, "evidence", "") or ""
            final_state = getattr(result, "final_state", {}) or {}
            self.token_usage = final_state.get("token_usage", {}) or {}

    # ── 序列化 ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """递归转为 Python dict，可直接 json.dumps()。"""
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "thread_id": self.thread_id,
            "loop_type": self.loop_type,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 1),
            "success": self.success,
            "iterations_used": self.iterations_used,
            "reason": self.reason,
            "evidence": self.evidence,
            "agent_config": self.agent_config,
            "iterations": [r.to_dict() for r in self.iterations],
            "token_usage": self.token_usage,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        """输出格式化的 JSON 字符串。"""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,   # 对未知类型 fallback 到 str()
        )


# ── 上下文管理器 & 辅助函数 ──────────────────────────────────────────────────


@asynccontextmanager
async def agent_trace(
    name: str = "",
    *,
    trace_id: str | None = None,
    thread_id: str = "",
    loop_type: str = "",
    agent_config: dict | None = None,
):
    """异步上下文管理器：创建 AgentTrace 并设置为当前 async 上下文的全局追踪。

    退出时若 trace 尚未 finish（未调用 trace.finish(result)），自动补全时间戳。

    用法::

        async with agent_trace("retrieval_agent", loop_type="GoalLoop") as trace:
            result = await loop.run(messages, thread_id="req-001")
            trace.finish(result)          # 从 LoopResult 填充顶层摘要

        print(trace.to_json())           # 完整 JSON 日志
        save_to_monitor(trace.to_dict()) # 写入监控系统
    """
    trace = AgentTrace(
        trace_id=trace_id,
        agent_name=name,
        thread_id=thread_id,
        loop_type=loop_type,
        agent_config=agent_config,
    )
    token = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    except Exception as exc:
        if not trace.error:
            trace.error = str(exc)
        raise
    finally:
        if not trace.finished_at:
            trace.finish()
        _CURRENT_TRACE.reset(token)


async def run_traced(
    loop: Any,
    input_messages: list | None = None,
    *,
    thread_id: str = "default",
    name: str = "",
    trace_id: str | None = None,
) -> tuple[Any, AgentTrace]:
    """包装 ``loop.run()``，自动收集全链路 JSON 追踪日志。

    自动从 loop 对象提取 agent_name、middleware 列表、budget 配置等，
    写入 AgentTrace.agent_config。

    Parameters
    ----------
    loop : TurnLoop | GoalLoop
        uniagent 循环对象。
    input_messages : list | None
        传递给 ``loop.run()`` 的初始消息列表。
    thread_id : str
        LangGraph thread_id。
    name : str
        Agent 名称（留空则从 loop._agent.name 自动提取）。
    trace_id : str | None
        追踪 ID（留空自动生成 UUID），用于关联外部请求链路（如 X-Request-ID）。

    Returns
    -------
    tuple[LoopResult, AgentTrace]

    用法::

        result, trace = await run_traced(
            loop,
            [HumanMessage(content="查询4G套餐费用")],
            thread_id="req-001",
            name="retrieval_agent",
            trace_id=request_id,   # 与 HTTP 请求头关联
        )
        logger.info("agent_trace=%s", trace.to_json())
    """
    _loop_type = type(loop).__name__
    _agent = getattr(loop, "_agent", None)
    _chain = getattr(_agent, "_uniagent_middleware", []) or []
    _hooks = getattr(loop, "_hooks", []) or []
    _budget = getattr(loop, "_budget", None)
    _cfg_obj = getattr(_budget, "config", None)
    _agent_name = name or getattr(_agent, "name", "") or ""

    _agent_cfg = {
        "agent_name": _agent_name,
        "middleware": [type(m).__name__ for m in _chain],
        "loop_hooks": [getattr(h, "name", type(h).__name__) for h in _hooks],
        "budget": {
            "max_iterations": getattr(_cfg_obj, "max_iterations", None),
            "max_time_seconds": getattr(_cfg_obj, "max_time_seconds", None),
        } if _cfg_obj else {},
    }

    async with agent_trace(
        _agent_name,
        trace_id=trace_id,
        thread_id=thread_id,
        loop_type=_loop_type,
        agent_config=_agent_cfg,
    ) as trace:
        result = await loop.run(input_messages, thread_id=thread_id)
        trace.finish(result)

    return result, trace
