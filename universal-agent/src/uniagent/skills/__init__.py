"""技能子系统 — 基于触发器匹配、渐进式披露的技能注入。"""

from uniagent.skills.manifest import SkillManifest, TriggerRule, ReferenceEntry
from uniagent.skills.loader import SkillLoader, SkillContent
from uniagent.skills.registry import SkillRegistry, MatchResult
from uniagent.skills.injector import SkillInjector

__all__ = [
    "SkillManifest",
    "TriggerRule",
    "ReferenceEntry",
    "SkillLoader",
    "SkillContent",
    "SkillRegistry",
    "MatchResult",
    "SkillInjector",
]
