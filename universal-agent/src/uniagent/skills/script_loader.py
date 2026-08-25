"""技能脚本动态加载器 — 扫描 scripts/ 目录，提取 @tool 标记的函数。"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from uniagent.skills.manifest import SkillManifest

logger = logging.getLogger(__name__)


def load_skill_scripts(
    manifest: SkillManifest, skill_dir: Path
) -> list[Any]:
    """扫描技能 scripts/ 目录，加载所有 @tool 标记的函数。

    Parameters
    ----------
    manifest:
        技能清单，其中 ``scripts`` 字段列出了要加载的脚本文件名。
    skill_dir:
        技能的根目录。

    Returns
    -------
    list[BaseTool]
        从脚本中提取的 LangChain 工具列表。
    """
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return []

    if not manifest.scripts:
        return []

    tools: list[Any] = []

    for script_file in manifest.scripts:
        path = scripts_dir / script_file
        if not path.is_file() or path.suffix != ".py":
            logger.warning(
                "技能 %r 的脚本 %r 不存在或非 .py 文件，跳过。",
                manifest.name,
                script_file,
            )
            continue

        try:
            loaded = _load_tools_from_script(path, manifest.skill_id)
            tools.extend(loaded)
            logger.info(
                "从技能 %r 的脚本 %r 中加载了 %d 个工具。",
                manifest.name,
                script_file,
                len(loaded),
            )
        except Exception as exc:
            logger.error(
                "加载技能 %r 的脚本 %r 失败：%s",
                manifest.name,
                script_file,
                exc,
            )

    return tools


def _load_tools_from_script(
    script_path: Path, skill_id: str
) -> list[Any]:
    """从单个 Python 脚本中提取 BaseTool 实例。

    使用 importlib 动态加载模块，然后扫描模块属性，
    找出所有 BaseTool 的实例（即用 @tool 装饰的函数）。
    模块名加入 skill_id 前缀，避免不同技能的同名脚本冲突。
    """
    # ── 构造唯一模块名，避免与其他技能脚本冲突 ──
    module_name = f"_skill_script_{skill_id}_{script_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        logger.warning("无法为 %s 创建模块 spec。", script_path)
        return []

    module = importlib.util.module_from_spec(spec)
    # 临时加入 sys.modules，支持模块内部的相对导入
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # 加载失败时清理 sys.modules，防止留下损坏的模块
        sys.modules.pop(module_name, None)
        raise

    # ── 扫描模块属性，提取所有 BaseTool 实例 ──
    tools: list[Any] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name, None)
        if isinstance(attr, BaseTool):
            tools.append(attr)

    return tools
