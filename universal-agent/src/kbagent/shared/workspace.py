# -*- coding: utf-8 -*-
"""单次请求的共享工作区(ContextVar)。

工具只向 LLM 返回精简观察值字符串,真实的 Chunk 对象/DSL/检索参数放在
按请求隔离的工作区里,供验证器与下游阶段读取。
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
    data: Dict[str, Any] = field(default_factory=dict)


_workspace: ContextVar[Optional[RunWorkspace]] = ContextVar("kb_workspace", default=None)


def set_workspace(ws: RunWorkspace) -> None:
    _workspace.set(ws)


def get_workspace() -> RunWorkspace:
    ws = _workspace.get()
    if ws is None:
        raise RuntimeError("RunWorkspace 未初始化:请先调用 set_workspace()")
    return ws
