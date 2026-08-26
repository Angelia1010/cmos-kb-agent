"""uniagent 结构化执行追踪子包。

公开 API：

    AgentTrace     — 全量 JSON 日志容器（to_dict() / to_json()）
    TraceMiddleware — 中间件形式的追踪采集器（加入 middleware 列表即可）
    agent_trace    — 异步上下文管理器，设置当前 trace 上下文
    run_traced     — 便捷包装，返回 (LoopResult, AgentTrace)
    get_current_trace — 从当前 async 上下文读取 AgentTrace

典型用法::

    from uniagent.logging import TraceMiddleware, run_traced

    loop = create_agent(
        model=model, tools=[...],
        middleware=[TraceMiddleware(), ...],
        ...
    )
    result, trace = await run_traced(loop, messages, name="my_agent")
    print(trace.to_json())
"""
from uniagent.logging.trace import (
    AgentTrace,
    IterationRecord,
    LLMCallRecord,
    MiddlewareEvent,
    ToolCallRecord,
    agent_trace,
    get_current_trace,
    run_traced,
)
from uniagent.logging.trace_middleware import TraceMiddleware

__all__ = [
    "AgentTrace",
    "IterationRecord",
    "LLMCallRecord",
    "MiddlewareEvent",
    "ToolCallRecord",
    "agent_trace",
    "get_current_trace",
    "run_traced",
    "TraceMiddleware",
]
