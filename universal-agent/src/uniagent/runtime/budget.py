"""Token / 时间 / 迭代次数预算管理，支持硬性限制强制执行。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from uniagent.runtime.signals import LoopSignal

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """预算限制配置 — 0 表示不限制。"""

    max_iterations: int = 25
    max_tokens: int = 0
    max_time_seconds: float = 0


@dataclass
class Budget:
    """可变预算追踪器，支持硬性限制强制执行。

    在每次迭代开始时调用 ``check()``。若任意限制超出，
    将返回 ``LoopSignal.BREAK`` 并附带原因字符串。
    """

    config: BudgetConfig = field(default_factory=BudgetConfig)

    # ── 可变计数器 ──
    iterations_used: int = 0
    tokens_used: int = 0
    _start_time: float = field(default_factory=time.monotonic)
    # M1: 线程安全锁，消除 check() 与 record_*() 之间的 TOCTOU 窗口
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_iteration(self) -> None:
        """递增迭代计数器。"""
        with self._lock:
            self.iterations_used += 1

    def record_tokens(self, count: int) -> None:
        """将 *count* 个 token 累加到总计数中。"""
        with self._lock:
            self.tokens_used += count

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def check(self) -> tuple[LoopSignal, str]:
        """检查所有预算限制。若未超限则返回 (CONTINUE, "")。

        M1: 在锁内读取计数器，避免 TOCTOU 竞态。
        """
        with self._lock:
            iterations = self.iterations_used
            tokens = self.tokens_used
        # 迭代次数限制
        if self.config.max_iterations > 0 and iterations >= self.config.max_iterations:
            reason = (
                f"迭代预算已耗尽：{iterations}"
                f"/{self.config.max_iterations}"
            )
            logger.warning(reason)
            return LoopSignal.BREAK, reason

        # Token 数量限制
        if self.config.max_tokens > 0 and tokens >= self.config.max_tokens:
            reason = (
                f"Token 预算已耗尽：{tokens}"
                f"/{self.config.max_tokens}"
            )
            logger.warning(reason)
            return LoopSignal.BREAK, reason

        # 时间限制
        if self.config.max_time_seconds > 0:
            elapsed = self.elapsed_seconds()
            if elapsed >= self.config.max_time_seconds:
                reason = (
                    f"时间预算已耗尽：{elapsed:.1f}s"
                    f"/{self.config.max_time_seconds}s"
                )
                logger.warning(reason)
                return LoopSignal.BREAK, reason

        return LoopSignal.CONTINUE, ""

    def summary(self) -> dict[str, object]:
        """返回人类可读的预算使用摘要。"""
        return {
            "iterations": f"{self.iterations_used}/{self.config.max_iterations or '∞'}",
            "tokens": f"{self.tokens_used}/{self.config.max_tokens or '∞'}",
            "time": f"{self.elapsed_seconds():.1f}s/{self.config.max_time_seconds or '∞'}s",
        }
