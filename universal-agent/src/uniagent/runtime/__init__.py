"""运行时支持 — 循环引擎、钩子、预算、上下文、检查点。"""

from uniagent.runtime.context import CurrentUser, get_current_user, set_current_user
from uniagent.runtime.checkpointer import create_checkpointer
from uniagent.runtime.signals import LoopSignal, LoopResult, HookResponse
from uniagent.runtime.budget import Budget, BudgetConfig
from uniagent.runtime.hooks import LoopHook, ProgressLogHook, TokenBudgetHook
from uniagent.runtime.loop import GoalLoop, TurnLoop, BaseLoop

__all__ = [
    "CurrentUser",
    "get_current_user",
    "set_current_user",
    "create_checkpointer",
    "LoopSignal",
    "LoopResult",
    "HookResponse",
    "Budget",
    "BudgetConfig",
    "LoopHook",
    "ProgressLogHook",
    "TokenBudgetHook",
    "GoalLoop",
    "TurnLoop",
    "BaseLoop",
]
