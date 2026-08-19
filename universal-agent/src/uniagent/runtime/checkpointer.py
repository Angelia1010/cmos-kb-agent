"""LangGraph 持久化检查点工厂。"""

from __future__ import annotations

import logging
from typing import Any

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
    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver(**kwargs)

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            return SqliteSaver(**kwargs)
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite 未安装，回退至内存模式。"
            )
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    # 通过反射加载自定义后端
    from uniagent.imports.resolvers import resolve_class
    cls = resolve_class(backend)
    return cls(**kwargs)
