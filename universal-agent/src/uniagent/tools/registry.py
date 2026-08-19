"""工具注册表 —— 基于反射的加载与去重。"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from uniagent.config.app_config import AppConfig
from uniagent.imports.resolvers import resolve_variable, ResolveError

logger = logging.getLogger(__name__)


def get_available_tools(
    config: AppConfig,
    *,
    extra_tools: Sequence[Any] | None = None,
    mcp_tools: Sequence[Any] | None = None,
    excluded_names: set[str] | None = None,
) -> list[Any]:
    """从配置、额外工具及 MCP 加载并去重工具列表。

    顺序：配置工具 → extra_tools → mcp_tools。
    按 ``tool.name`` 去重（先出现者优先）。

    Parameters
    ----------
    config:
        包含 ``tools`` 列表的应用配置。
    extra_tools:
        通过代码直接传入的工具（如内置工具）。
    mcp_tools:
        从 MCP 服务器发现的工具。
    excluded_names:
        需要完全跳过的工具名称集合。
    """
    seen: set[str] = set()
    result: list[Any] = []
    excluded = excluded_names or set()

    def _add(tool: Any) -> None:
        name = _tool_name(tool)
        if name in seen or name in excluded:
            return
        seen.add(name)
        result.append(tool)

    # 1. 配置定义的工具（通过反射加载）
    for tc in config.tools:
        if not tc.enabled:
            continue
        try:
            tool = resolve_variable(tc.use)
        except ResolveError as exc:
            logger.warning("跳过工具 %r：%s", tc.name, exc)
            continue
        # 如果是类，则尝试实例化
        if isinstance(tool, type):
            try:
                tool = tool(**tc.kwargs)
            except Exception as exc:
                logger.warning("实例化工具 %r 失败：%s", tc.name, exc)
                continue
        _add(tool)

    # 2. 额外工具（通过代码传入）
    for tool in extra_tools or []:
        _add(tool)

    # 3. MCP 工具
    for tool in mcp_tools or []:
        _add(tool)

    return result


def _tool_name(tool: Any) -> str:
    """从工具对象中提取名称（支持 BaseTool、函数或含 .name 属性的对象）。"""
    if isinstance(tool, BaseTool):
        return tool.name
    if hasattr(tool, "name"):
        return str(tool.name)
    if callable(tool) and hasattr(tool, "__name__"):
        return tool.__name__
    return str(id(tool))
