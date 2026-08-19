"""技能 → 系统提示注入流水线。"""

from __future__ import annotations

import logging

from uniagent.skills.loader import SkillContent

logger = logging.getLogger(__name__)

# 系统提示中技能内容的段落标记
_SKILL_SECTION_START = "\n\n<!-- SKILL: {skill_id} -->\n"
_SKILL_SECTION_END = "\n<!-- /SKILL: {skill_id} -->\n"


class SkillInjector:
    """将已激活的技能内容注入到 Agent 的系统提示中。

    注入格式::

        <原始系统提示>

        <!-- SKILL: my-skill -->
        ## Skill: My Skill
        <SKILL.md 内容>

        ### References
        <即时参考内容>
        <!-- /SKILL: my-skill -->

    支持同时激活多个技能。
    """

    def __init__(self, *, max_active_skills: int = 3) -> None:
        self._max_active = max_active_skills
        self._active: list[SkillContent] = []

    @property
    def active_skills(self) -> list[SkillContent]:
        return list(self._active)

    def activate(self, content: SkillContent) -> None:
        """将技能添加到激活集合中。

        如果技能已处于激活状态，则不重复添加。
        如果达到最大激活数量，则驱逐最旧的技能。
        """
        sid = content.manifest.skill_id
        # 去重
        if any(c.manifest.skill_id == sid for c in self._active):
            logger.debug("技能 %r 已处于激活状态，跳过。", sid)
            return

        if len(self._active) >= self._max_active:
            evicted = self._active.pop(0)
            logger.info(
                "驱逐技能 %r 以为 %r 腾出空间。",
                evicted.manifest.skill_id,
                sid,
            )

        self._active.append(content)
        logger.info("技能 %r 已激活。", sid)

    def deactivate(self, skill_id: str) -> bool:
        """从激活集合中移除技能。"""
        before = len(self._active)
        self._active = [
            c for c in self._active if c.manifest.skill_id != skill_id
        ]
        removed = len(self._active) < before
        if removed:
            logger.info("技能 %r 已停用。", skill_id)
        return removed

    def clear(self) -> None:
        """移除所有激活的技能。"""
        self._active.clear()

    def inject(self, base_prompt: str) -> str:
        """构建包含所有激活技能段落的最终系统提示。

        参数
        ----------
        base_prompt:
            技能注入前的原始系统提示。

        返回
        -------
        str
            追加了技能内容的系统提示。
        """
        if not self._active:
            return base_prompt

        parts = [base_prompt]

        for content in self._active:
            section = self._format_skill_section(content)
            parts.append(section)

        return "".join(parts)

    def get_promoted_tools(self) -> list[str]:
        """收集激活技能希望提升的工具名称。"""
        tools: list[str] = []
        seen: set[str] = set()
        for content in self._active:
            for tool_name in content.manifest.promoted_tools:
                if tool_name not in seen:
                    tools.append(tool_name)
                    seen.add(tool_name)
        return tools

    def _format_skill_section(self, content: SkillContent) -> str:
        """将单个技能的内容格式化为系统提示段落。"""
        sid = content.manifest.skill_id
        parts: list[str] = []

        parts.append(_SKILL_SECTION_START.format(skill_id=sid))
        parts.append(f"## Skill: {content.manifest.name}\n")

        if content.manifest.description:
            parts.append(f"{content.manifest.description}\n\n")

        # 主要指令（SKILL.md）
        if content.instruction:
            parts.append(content.instruction)
            if not content.instruction.endswith("\n"):
                parts.append("\n")

        # 即时参考
        all_refs = content.all_references
        if all_refs:
            parts.append("\n### References\n")
            for filename, text in all_refs.items():
                parts.append(f"\n#### {filename}\n")
                parts.append(text)
                if not text.endswith("\n"):
                    parts.append("\n")

        # 模板列表（仅显示名称，不含完整内容）
        if content.manifest.templates:
            parts.append("\n### Available Templates\n")
            for tmpl in content.manifest.templates:
                parts.append(f"- `{tmpl}`\n")

        # 脚本列表
        if content.manifest.scripts:
            parts.append("\n### Available Scripts\n")
            for script in content.manifest.scripts:
                parts.append(f"- `{script}`\n")

        parts.append(_SKILL_SECTION_END.format(skill_id=sid))
        return "".join(parts)
