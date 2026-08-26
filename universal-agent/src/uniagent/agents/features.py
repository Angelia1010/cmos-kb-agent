"""Agent 构建的声明式运行时特性标志。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uniagent.middleware.base import Middleware
from uniagent.runtime.hooks import ProgressLogHook


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

    # ── Agent 节点层级（中间件开关）──
    dangling_tool_call: bool | Middleware = True
    tool_error_handling: bool | Middleware = True
    loop_detection: bool | Middleware = True
    token_usage: bool | Middleware = True
    skill: bool | Middleware = False
    """启用技能自动匹配中间件（SkillMiddleware）。"""
    logging: bool | Middleware = False
    """启用 LLM 调用日志中间件（LLMLoggingMiddleware）。

    True  → 使用默认实例（verbose=False，log_level=DEBUG）。
    False → 禁用（默认）。
    Middleware 实例 → 使用自定义实例，如 LLMLoggingMiddleware(verbose=True)。
    """

    # ── 循环层级 ──
    goal_loop: bool = False
    """启用 GoalLoop 包装（需配合 goal 和 verifier 使用）。"""

    verification: str = "none"
    """验证策略：'llm'、'composite' 或 'none'（KB 场景通常以代码直接传入 Verifier）。"""

    def resolve_middleware(self) -> list[Middleware]:
        """将 Agent 节点层级特性标志解析为有序中间件列表。

        NOTE: middleware.builtins 的导入保持延迟（不在文件头），原因：
        features.py ← factory.py ← config_factory.py ← skill_middleware.py
                                                              ↑
        skill_middleware.py 在文件头导入 config_factory，
        而 config_factory 导入 factory，factory 导入 features。
        若在 features.py 文件头再导入 middleware.builtins（含 skill_middleware），
        则形成循环依赖。此处延迟导入是整条链的唯一断点，不可移动。
        """
        # 延迟导入：打破循环依赖的唯一断点（见上方注释）
        from uniagent.middleware.builtins import (
            DanglingToolCallMiddleware,
            LLMLoggingMiddleware,
            LoopDetectionMiddleware,
            SkillMiddleware,
            TokenUsageMiddleware,
            ToolErrorHandlingMiddleware,
        )

        # 按照中间件洋葱模型的执行顺序排列
        # LLMLoggingMiddleware 置于链首，在其他中间件修改 state 之前捕获原始快照
        _defaults: list[tuple[str, type[Middleware]]] = [
            ("logging",             LLMLoggingMiddleware),
            ("skill",               SkillMiddleware),
            ("dangling_tool_call",  DanglingToolCallMiddleware),
            ("tool_error_handling", ToolErrorHandlingMiddleware),
            ("loop_detection",      LoopDetectionMiddleware),
            ("token_usage",         TokenUsageMiddleware),
        ]

        result: list[Middleware] = []
        for attr, default_cls in _defaults:
            value = getattr(self, attr)
            if value is False:
                continue          # 禁用
            if value is True:
                result.append(default_cls())       # 使用默认实例
            elif isinstance(value, Middleware):
                result.append(value)               # 使用自定义实例
        return result

    def default_loop_hooks(self) -> list[Any]:
        """将循环层级特性标志解析为钩子实例。"""
        hooks: list = [ProgressLogHook()]
        # WIP 约束钩子在 factory 设置 external_state 时添加
        return hooks
