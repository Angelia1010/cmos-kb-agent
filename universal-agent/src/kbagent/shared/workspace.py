# -*- coding: utf-8 -*-
"""单次请求的共享工作区(ContextVar)。

uniagent 的工具是 LangChain @tool:向 LLM 只返回精简观察值字符串,
真实的 Chunk 对象/DSL/检索参数放在按请求隔离的工作区里,
供验证器(SufficiencyVerifier)与下游阶段读取。
复用了 uniagent runtime/context.py 的 ContextVar 模式。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from .config import Config, DEFAULT_CONFIG
from .search import ESClient
from .tracing import Tracer


@dataclass
class RunWorkspace:
    query: str = ""
    cfg: Config = field(default_factory=lambda: DEFAULT_CONFIG)
    es: Optional[ESClient] = None
    tracer: Tracer = field(default_factory=Tracer)
    stage: str = ""
    data: Dict[str, Any] = field(default_factory=dict)   # chunks / keywords / last_dsl ...


_workspace: ContextVar[Optional[RunWorkspace]] = ContextVar("kb_workspace", default=None)


def set_workspace(ws: RunWorkspace) -> None:
    _workspace.set(ws)


@contextmanager
def workspace_scope(ws: RunWorkspace) -> Iterator[RunWorkspace]:
    """在当前执行上下文绑定 Workspace，并在退出时恢复原值。"""
    token = _workspace.set(ws)
    try:
        yield ws
    finally:
        _workspace.reset(token)


def get_workspace() -> RunWorkspace:
    ws = _workspace.get()
    if ws is None:
        raise RuntimeError("RunWorkspace 未初始化:请先在 pipeline 中调用 set_workspace()")
    return ws
