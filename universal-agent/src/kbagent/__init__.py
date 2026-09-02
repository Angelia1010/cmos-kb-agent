# -*- coding: utf-8 -*-
"""kbagent — 编排好的主智能体 + 三个自主规划的子智能体(基于裁剪版 uniagent)。"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .scripted_model import ScriptedChatModel
from .shared.config import Config, DEFAULT_CONFIG
from .shared.search import ESClient, MockESClient

if TYPE_CHECKING:
    from .answer.agent import AnswerSubAgent
    from .main_agent import MainAgent
    from .retrieval.agent import RetrievalSubAgent


_LAZY_EXPORTS = {
    "MainAgent": (".main_agent", "MainAgent"),
    "RetrievalSubAgent": (".retrieval.agent", "RetrievalSubAgent"),
    "AnswerSubAgent": (".answer.agent", "AnswerSubAgent"),
}


def __getattr__(name: str) -> Any:
    """按需加载 Agent，避免导入独立子模块时初始化完整主链。"""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "Config", "DEFAULT_CONFIG",
    "MainAgent",
    "ScriptedChatModel",
    "ESClient", "MockESClient",
    "RetrievalSubAgent", "AnswerSubAgent",
]
