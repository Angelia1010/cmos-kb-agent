"""带注解归约器的线程级 Agent 状态。"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.prebuilt.chat_agent_executor import AgentState

from uniagent.state.reducers import (
    dedup_list_merge,
    idempotent_merge,
    last_wins,
)


class ThreadState(AgentState):
    """携带制品与循环元数据的扩展 Agent 状态(KB 适配版:移除沙箱)。

    每个字段使用 ``Annotated[..., reducer]``，以便 LangGraph 知道如何
    在并行分支或 ``Command(update=...)`` 中合并局部更新。
    """

    artifacts: Annotated[dict[str, Any], idempotent_merge] = {}  # noqa: RUF012
    """执行过程中产生的任意键值制品。"""

    promoted_tools: Annotated[list[str], dedup_list_merge] = []  # noqa: RUF012
    """由延迟工具搜索机制提升的工具名称。"""

    todos: Annotated[list[dict[str, Any]], dedup_list_merge] = []  # noqa: RUF012
    """执行过程中跟踪的任务/待办事项。"""

    summary: Annotated[str, last_wins] = ""
    """由摘要中间件生成的对话摘要。"""

    token_usage: Annotated[dict[str, int], idempotent_merge] = {}  # noqa: RUF012
    """累计 Token 使用量计数器。"""

    current_task: Annotated[str | None, last_wins] = None
    """当前激活的功能/任务 ID（WIP=1 约束）。"""

    loop_iteration: Annotated[int, last_wins] = 0
    """当前循环迭代编号（由循环引擎设置）。"""

    verification_result: Annotated[dict[str, Any] | None, last_wins] = None
    """来自验证器的最新验证结果。"""
