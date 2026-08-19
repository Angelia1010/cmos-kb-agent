"""基于 LangGraph AgentMiddleware 构建的中间件基类。"""

from __future__ import annotations

import logging
from typing import Any

from uniagent.state.thread_state import ThreadState

logger = logging.getLogger(__name__)


class Middleware:
    """自包含中间件基类(适配层)。

    原实现继承 ``langgraph.prebuilt.chat_agent_executor.AgentMiddleware``,
    该符号在 langgraph 0.6/1.x 中均不存在,导致框架无法导入。
    适配后中间件由 ``BaseLoop._invoke_agent`` 在智能体调用前后显式执行
    (before_agent 正序 / after_agent 逆序),不再依赖 langgraph 内部接口。
    """

    """所有 uniagent 中间件的基类。

    在 **两个层级** 上运行：

    **Agent 节点层**（继承自 AgentMiddleware）：
    - ``before_agent``    — 在 agent 节点之前运行（正序）
    - ``after_agent``     — 在 agent 节点之后运行（逆序）
    - ``wrap_model_call`` — 包装 LLM 调用（洋葱顺序）
    - ``wrap_tool_call``  — 包装每个工具调用（洋葱顺序）

    **循环层**（可选，新增）：
    - ``loop_hooks()``    — 返回 LoopHook 实例以注册到循环引擎。
      允许中间件同时参与 agent 节点和循环迭代的生命周期。

    默认实现为透传（恒等）。
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__

    async def before_agent(self, state: ThreadState) -> ThreadState | None:
        """在 agent 节点之前调用。返回修改后的状态或 None。"""
        return None

    async def after_agent(self, state: ThreadState) -> ThreadState | None:
        """在 agent 节点之后调用。返回修改后的状态或 None。"""
        return None

    def loop_hooks(self) -> list[Any]:
        """返回要注册到循环引擎的 LoopHook 实例。

        覆盖此方法可使中间件参与循环层的生命周期事件
        （on_iteration_start/end、on_goal_achieved 等），
        同时保留对 agent 节点事件的响应。

        默认返回空列表（不参与循环层）。
        """
        return []
