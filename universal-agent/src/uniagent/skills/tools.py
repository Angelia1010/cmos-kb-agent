"""技能系统的 LangChain 工具 — 供 LLM 在推理过程中主动调用。"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

# NOTE: config_factory 依赖 factory → features，但不依赖 skills.tools，
#       因此此处可安全地在文件头导入（无循环依赖）。
from uniagent.agents.config_factory import get_skill_registry
from uniagent.skills.loader import SkillLoader

logger = logging.getLogger(__name__)


@tool
def load_skill_reference(skill_name: str, filename: str) -> str:
    """按需加载技能的参考文档。当你需要更详细的业务规范时调用此工具。

    Args:
        skill_name: 技能名称（如 taocan-skill）
        filename: 参考文档文件名（如 guide.md）
    """
    # ── 1. 获取全局技能注册表 ──
    registry = get_skill_registry()
    if registry is None:
        return "错误：技能注册表未初始化。"

    # ── 2. 按名称/ID 查找技能 ──
    match = registry.match_by_name(skill_name)
    if match is None:
        available = [s["name"] for s in registry.list_skills()]
        return f"错误：未找到技能 '{skill_name}'。可用技能：{available}"

    # ── 3. 按需加载指定参考文档 ──
    loader = SkillLoader()
    # 先加载技能内容（初始化 eager_references），再按需加载目标文件
    content = loader.load(match.manifest, match.skill_dir)
    text = loader.load_reference(content, match.skill_dir, filename)

    if text is None:
        # 列出该技能下所有可用的参考文档供 LLM 参考
        available_refs = loader.list_available_references(
            match.manifest, match.skill_dir
        )
        ref_names = [r["filename"] for r in available_refs]
        return (
            f"错误：技能 '{skill_name}' 中未找到参考文档 '{filename}'。"
            f"可用参考文档：{ref_names}"
        )

    logger.info(
        "按需加载了技能 %r 的参考文档 %r（%d 字符）。",
        skill_name,
        filename,
        len(text),
    )
    return text
