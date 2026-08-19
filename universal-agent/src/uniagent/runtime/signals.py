"""循环控制信号，由钩子和预算检查发出。"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any


class LoopSignal(Enum):
    """循环钩子返回的控制信号，用于引导迭代流程。"""

    CONTINUE = auto()
    """继续执行下一步 / 下一次迭代。"""

    BREAK = auto()
    """立即终止循环（目标达成或强制停止）。"""

    RETRY = auto()
    """重新执行当前迭代（例如在瞬时错误后）。"""

    ROLLBACK = auto()
    """回退至上一个检查点并从该点重试。"""


@dataclass(frozen=True)
class LoopResult:
    """已完成的循环执行结果。"""

    success: bool
    """是否达成目标。"""

    iterations: int
    """已执行的迭代次数。"""

    reason: str = ""
    """人类可读的终止原因说明。"""

    final_state: dict[str, Any] = field(default_factory=dict)
    """终止时的线程状态快照。"""

    evidence: str = ""
    """success=True 时的验证依据（例如测试输出）。"""


@dataclass(frozen=True)
class HookResponse:
    """循环钩子调用的响应结果。"""

    signal: LoopSignal = LoopSignal.CONTINUE
    """控制信号。"""

    message: str = ""
    """注入对话的可选消息。"""

    state_patch: dict[str, Any] | None = None
    """可选的部分状态更新。"""
