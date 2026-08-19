"""Agent 构建的声明式运行时特性标志。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uniagent.middleware.base import Middleware


@dataclass
class AgentFeatures:
    """``create_agent`` 的声明式特性开关。

    两个层级的特性：

    **Agent 节点层级**（中间件）：
    - ``True``  → 使用默认内置中间件
    - ``False`` → 禁用
    - ``Middleware`` 实例 → 使用该特定实例

    **循环层级**（新增）：
    - ``goal_loop``       → 启用 GoalLoop 包装
    - ``verification``    → GoalLoop 的验证策略

    示例::

        features = AgentFeatures(
            loop_detection=LoopDetectionMiddleware(hard_limit=5),
            skill=True,
            goal_loop=True,
        )
    """

    # ── Agent 节点层级（中间件）──
    dangling_tool_call: bool | Middleware = True
    tool_error_handling: bool | Middleware = True
    loop_detection: bool | Middleware = True
    token_usage: bool | Middleware = True
    skill: bool | Middleware = False
    """启用技能自动匹配中间件。"""

    # ── 循环层级 ──
    goal_loop: bool = False
    """启用 GoalLoop 包装（需要 goal 和 verifier）。"""

    verification: str = "none"
    """验证策略：'llm'、'composite' 或 'none'（KB 场景通常以代码直接传入 Verifier）。"""

    def resolve_middleware(self) -> list[Middleware]:
        """将 Agent 节点层级特性标志解析为有序中间件列表。"""
        from uniagent.middleware.builtins import (
            DanglingToolCallMiddleware,
            LoopDetectionMiddleware,
            SkillMiddleware,
            TokenUsageMiddleware,
            ToolErrorHandlingMiddleware,
        )

        _defaults: list[tuple[str, type[Middleware]]] = [
            ("skill", SkillMiddleware),
            ("dangling_tool_call", DanglingToolCallMiddleware),
            ("tool_error_handling", ToolErrorHandlingMiddleware),
            ("loop_detection", LoopDetectionMiddleware),
            ("token_usage", TokenUsageMiddleware),
        ]

        result: list[Middleware] = []
        for attr, default_cls in _defaults:
            value = getattr(self, attr)
            if value is False:
                continue
            if value is True:
                result.append(default_cls())
            elif isinstance(value, Middleware):
                result.append(value)
        return result

    def default_loop_hooks(self) -> list[Any]:
        """将循环层级特性标志解析为钩子实例。"""
        from uniagent.runtime.hooks import ProgressLogHook

        hooks: list = [ProgressLogHook()]
        # WIP 约束钩子在 factory 设置 external_state 时添加
        return hooks
