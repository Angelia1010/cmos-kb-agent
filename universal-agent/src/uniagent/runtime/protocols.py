"""跨层类型契约的轻量级协议定义。

本模块仅包含零实现依赖的 Protocol 定义。
中间件及其他底层模块可从此处导入，
而无需引入完整的 runtime/hooks 机制。

H5: LoopSignalBase 和 LoopHookBase 与 signals.py/hooks.py 中的对应类
统一为继承关系，避免独立定义导致类型不兼容。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# H5: 直接复用 signals.py 中的 LoopSignal，而非维护独立副本
from uniagent.runtime.signals import LoopSignal as LoopSignalBase  # noqa: F401


class LoopHookBase(ABC):
    """跨层契约所需的最小循环钩子接口。

    底层模块（中间件）可依赖此协议，而无需导入
    完整的 ``runtime.hooks`` 模块。``runtime/hooks.py`` 中的
    实际 ``LoopHook`` 类继承此基类。
    """

    name: str = ""

    @abstractmethod
    async def on_iteration_start(
        self, iteration: int, state: dict[str, Any]
    ) -> Any: ...

    @abstractmethod
    async def on_iteration_end(
        self, iteration: int, state: dict[str, Any], agent_output: dict[str, Any] | None
    ) -> Any: ...
