"""技能清单 — 从 ``metadata.json`` 解析的数据模型。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriggerRule:
    """单条技能激活触发条件。

    支持的类型：
    - ``keyword``：精确或子串匹配用户消息。
    - ``prefix``：消息以特定前缀开头（例如 ``/skill-name``）。
    - ``regex``：正则表达式匹配。
    - ``intent``：语义意图标签（需要外部分类器）。
    """

    type: str  # "keyword" | "prefix" | "regex" | "intent"
    value: str
    case_sensitive: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriggerRule:
        return cls(
            type=data.get("type", "keyword"),
            value=data["value"],
            case_sensitive=data.get("case_sensitive", False),
        )


@dataclass(frozen=True)
class ReferenceEntry:
    """可用于渐进式披露的参考文档。"""

    filename: str
    description: str = ""
    when: str = "on_demand"  # "always" | "on_demand" | "never"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceEntry:
        return cls(
            filename=data["filename"],
            description=data.get("description", ""),
            when=data.get("when", "on_demand"),
        )


@dataclass(frozen=True)
class SkillManifest:
    """从技能 ``metadata.json`` 解析出的不可变清单。

    预期目录结构::

        skills/
        └── my-skill/
            ├── metadata.json      ← 解析为此数据类
            ├── SKILL.md           ← 主要指令入口
            ├── references/        ← 渐进式披露文档
            │   ├── guide.md
            │   └── examples.md
            ├── templates/         ← 输出模板
            │   └── report.md
            └── scripts/           ← 可执行脚本
                └── validate.py
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    triggers: tuple[TriggerRule, ...] = field(default_factory=tuple)
    references: tuple[ReferenceEntry, ...] = field(default_factory=tuple)
    templates: tuple[str, ...] = field(default_factory=tuple)
    scripts: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    # 当此技能激活时应提升的工具列表
    promoted_tools: tuple[str, ...] = field(default_factory=tuple)
    # 来自 metadata.json 的额外元数据（透传）
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def skill_id(self) -> str:
        """从名称派生的稳定标识符。"""
        return self.name.lower().replace(" ", "-").replace("_", "-")

    @classmethod
    def from_json(cls, path: Path) -> SkillManifest:
        """将 ``metadata.json`` 文件解析为 ``SkillManifest``。"""
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillManifest:
        known_keys = {
            "name", "description", "version", "triggers", "references",
            "templates", "scripts", "tags", "promoted_tools",
        }
        triggers = tuple(
            TriggerRule.from_dict(t) for t in data.get("triggers", [])
        )
        references = tuple(
            ReferenceEntry.from_dict(r) for r in data.get("references", [])
        )
        extra = {k: v for k, v in data.items() if k not in known_keys}

        # M11: 明确的错误信息，而非默认 KeyError
        if "name" not in data:
            raise ValueError(
                f"技能清单缺少必需的 'name' 字段。已有键：{list(data.keys())}"
            )
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            triggers=triggers,
            references=references,
            templates=tuple(data.get("templates", [])),
            scripts=tuple(data.get("scripts", [])),
            tags=tuple(data.get("tags", [])),
            promoted_tools=tuple(data.get("promoted_tools", [])),
            extra=extra,
        )
