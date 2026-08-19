"""技能内容的渐进式披露加载器。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from uniagent.skills.manifest import SkillManifest

logger = logging.getLogger(__name__)


@dataclass
class SkillContent:
    """单个技能实例的已加载内容。

    渐进式披露阶段：
    1. ``instruction`` — 始终从 SKILL.md（入口）加载。
    2. ``eager_references`` — 标记为 ``when="always"`` 的参考，在激活时加载。
    3. ``loaded_references`` — 通过 ``load_reference()`` 按需加载的参考。
    4. ``templates`` — 模板文件内容，按需加载。
    """

    manifest: SkillManifest
    instruction: str = ""
    eager_references: dict[str, str] = field(default_factory=dict)
    loaded_references: dict[str, str] = field(default_factory=dict)
    _templates_cache: dict[str, str] = field(default_factory=dict)

    @property
    def all_references(self) -> dict[str, str]:
        """合并即时加载与按需加载的参考。"""
        merged = dict(self.eager_references)
        merged.update(self.loaded_references)
        return merged


class SkillLoader:
    """从文件系统加载技能内容，支持渐进式披露。

    用法::

        loader = SkillLoader()
        content = loader.load(manifest, skill_dir)
        # content.instruction → SKILL.md 全文
        # content.eager_references → 即时加载的参考

        # 稍后，按需加载：
        loader.load_reference(content, skill_dir, "guide.md")
        # content.loaded_references["guide.md"] → 文件文本
    """

    def load(self, manifest: SkillManifest, skill_dir: Path) -> SkillContent:
        """加载技能的核心指令和即时参考。

        参数
        ----------
        manifest:
            此技能的已解析清单。
        skill_dir:
            技能的根目录（包含 SKILL.md、references/ 等）。

        返回
        -------
        SkillContent
            已填充 ``instruction`` 和 ``eager_references``。
        """
        content = SkillContent(manifest=manifest)

        # 1. 加载 SKILL.md
        skill_md = skill_dir / "SKILL.md"
        if skill_md.is_file():
            content.instruction = skill_md.read_text(encoding="utf-8")
        else:
            logger.warning(
                "技能 %r 在 %s 处没有 SKILL.md", manifest.name, skill_dir
            )

        # 2. 加载即时参考（when="always"）
        refs_dir = skill_dir / "references"
        for ref in manifest.references:
            if ref.when == "always":
                text = self._read_reference(refs_dir, ref.filename)
                if text is not None:
                    content.eager_references[ref.filename] = text

        return content

    def load_reference(
        self, content: SkillContent, skill_dir: Path, filename: str
    ) -> str | None:
        """按需加载单个参考文件。

        结果会缓存到 ``content.loaded_references`` 中。
        """
        # 已加载？
        if filename in content.loaded_references:
            return content.loaded_references[filename]
        if filename in content.eager_references:
            return content.eager_references[filename]

        refs_dir = skill_dir / "references"
        text = self._read_reference(refs_dir, filename)
        if text is not None:
            content.loaded_references[filename] = text
        return text

    def load_template(
        self, content: SkillContent, skill_dir: Path, template_name: str
    ) -> str | None:
        """按需加载模板文件。"""
        if template_name in content._templates_cache:
            return content._templates_cache[template_name]

        tmpl_path = skill_dir / "templates" / template_name
        if not tmpl_path.is_file():
            logger.warning(
                "技能 %r 的模板 %r 未找到",
                content.manifest.name,
                template_name,
            )
            return None

        text = tmpl_path.read_text(encoding="utf-8")
        content._templates_cache[template_name] = text
        return text

    def list_available_references(
        self, manifest: SkillManifest, skill_dir: Path
    ) -> list[dict[str, str]]:
        """列出所有可用参考及其描述。"""
        result: list[dict[str, str]] = []
        refs_dir = skill_dir / "references"
        for ref in manifest.references:
            exists = (refs_dir / ref.filename).is_file()
            result.append({
                "filename": ref.filename,
                "description": ref.description,
                "when": ref.when,
                "exists": str(exists),
            })
        return result

    @staticmethod
    def _read_reference(refs_dir: Path, filename: str) -> str | None:
        """安全读取参考文件。

        H7: 校验解析后的路径不超出 refs_dir 边界，防止路径穿越攻击。
        """
        ref_path = (refs_dir / filename).resolve()
        refs_dir_resolved = refs_dir.resolve()
        # H7: 路径穿越检查
        try:
            ref_path.relative_to(refs_dir_resolved)
        except ValueError:
            logger.error(
                "路径穿越检测：%r 超出参考目录 %s 的边界，已拒绝。",
                filename,
                refs_dir_resolved,
            )
            return None
        if not ref_path.is_file():
            logger.warning("参考文件 %s 未找到。", ref_path)
            return None
        try:
            return ref_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("读取参考文件 %s 失败：%s", ref_path, exc)
            return None
