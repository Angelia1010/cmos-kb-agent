"""LangGraph 持久化检查点工厂。"""

from __future__ import annotations

import logging
from typing import Any

from uniagent.imports.resolvers import resolve_class

# langgraph.checkpoint.memory 是核心依赖，始终可用
from langgraph.checkpoint.memory import MemorySaver

# langgraph.checkpoint.sqlite 是可选依赖，可能未安装
try:
    from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver
    _SQLITE_AVAILABLE = True
except ImportError:
    _SqliteSaver = None  # type: ignore[assignment,misc]
    _SQLITE_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_checkpointer(backend: str = "memory", **kwargs: Any) -> Any:
    """根据后端名称创建 LangGraph 检查点保存器。

    Parameters
    ----------
    backend:
        可选值为 ``"memory"``、``"sqlite"`` 或点分隔的导入路径。
    **kwargs:
        传递给检查点保存器构造函数的参数。

    Returns
    -------
    BaseCheckpointSaver
        与 LangGraph 兼容的检查点保存器实例。
    """
    # ── 内存模式（默认）──
    if backend == "memory":
        return MemorySaver(**kwargs)

    # ── SQLite 持久化模式 ──
    if backend == "sqlite":
        if _SQLITE_AVAILABLE:
            return _SqliteSaver(**kwargs)
        logger.warning(
            "langgraph-checkpoint-sqlite 未安装，回退至内存模式。"
        )
        return MemorySaver()

    # ── 自定义后端（通过反射加载）──
    cls = resolve_class(backend)
    return cls(**kwargs)
