"""技能注册表 — 目录扫描、触发器索引与匹配。"""

from __future__ import annotations

import logging
import re
import signal as _signal
from dataclasses import dataclass
from pathlib import Path

from uniagent.skills.loader import SkillContent, SkillLoader
from uniagent.skills.manifest import SkillManifest, TriggerRule

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """触发器匹配尝试的结果。"""

    manifest: SkillManifest
    skill_dir: Path
    score: float  # 0.0–1.0，值越高表示匹配越好
    matched_trigger: TriggerRule | None = None


class SkillRegistry:
    """扫描技能目录、构建触发器索引并匹配用户输入。

    用法::

        registry = SkillRegistry()
        registry.scan("./skills")
        matches = registry.match("帮我创建一个harness")
        if matches:
            content = registry.activate(matches[0])
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillManifest] = {}  # skill_id → 清单
        self._skill_dirs: dict[str, Path] = {}  # skill_id → 目录
        self._loader = SkillLoader()
        # 已编译正则缓存
        self._regex_cache: dict[str, re.Pattern[str]] = {}

    @property
    def skills(self) -> dict[str, SkillManifest]:
        """已注册技能的只读视图。"""
        return dict(self._skills)

    def scan(self, *directories: str | Path) -> int:
        """扫描目录以查找技能包。

        每个包含 ``metadata.json`` 的子目录均视为一个技能。
        返回发现的技能数量。
        """
        count = 0
        for d in directories:
            dir_path = Path(d)
            if not dir_path.is_dir():
                logger.warning("技能目录 %s 不存在，跳过。", d)
                continue
            for child in sorted(dir_path.iterdir()):
                if not child.is_dir():
                    continue
                meta_file = child / "metadata.json"
                if not meta_file.is_file():
                    continue
                try:
                    manifest = SkillManifest.from_json(meta_file)
                    self._register(manifest, child)
                    count += 1
                except Exception as exc:
                    logger.error(
                        "从 %s 加载技能失败：%s", child, exc
                    )
        logger.info("从 %d 个目录中扫描到 %d 个技能。", len(directories), count)
        return count

    def register(self, manifest: SkillManifest, skill_dir: Path) -> None:
        """手动注册技能（适用于测试）。"""
        self._register(manifest, skill_dir)

    def _register(self, manifest: SkillManifest, skill_dir: Path) -> None:
        sid = manifest.skill_id
        if sid in self._skills:
            logger.warning(
                "技能 %r 已注册，将被 %s 覆盖。",
                sid,
                skill_dir,
            )
        self._skills[sid] = manifest
        self._skill_dirs[sid] = skill_dir
        # 预编译正则触发器
        for trigger in manifest.triggers:
            if trigger.type == "regex":
                key = f"{sid}:{trigger.value}"
                flags = 0 if trigger.case_sensitive else re.IGNORECASE
                try:
                    self._regex_cache[key] = re.compile(trigger.value, flags)
                except re.error as exc:
                    logger.error(
                        "技能 %r 的正则触发器 %r 无效：%s",
                        sid,
                        trigger.value,
                        exc,
                    )

    def unregister(self, skill_id: str) -> bool:
        """从注册表中移除技能。"""
        if skill_id not in self._skills:
            return False
        del self._skills[skill_id]
        del self._skill_dirs[skill_id]
        # 清理正则缓存
        prefix = f"{skill_id}:"
        to_delete = [k for k in self._regex_cache if k.startswith(prefix)]
        for k in to_delete:
            del self._regex_cache[k]
        return True

    def match(
        self, user_input: str, *, max_results: int = 3
    ) -> list[MatchResult]:
        """将用户输入与所有已注册技能的触发器进行匹配。

        返回按得分降序排列的匹配列表。
        """
        results: list[MatchResult] = []

        for sid, manifest in self._skills.items():
            best_score = 0.0
            best_trigger: TriggerRule | None = None

            for trigger in manifest.triggers:
                score = self._evaluate_trigger(trigger, user_input, sid)
                if score > best_score:
                    best_score = score
                    best_trigger = trigger

            if best_score > 0:
                results.append(
                    MatchResult(
                        manifest=manifest,
                        skill_dir=self._skill_dirs[sid],
                        score=best_score,
                        matched_trigger=best_trigger,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def match_by_name(self, name: str) -> MatchResult | None:
        """通过技能名称或 skill_id 直接查找。"""
        # 先尝试 skill_id
        name_normalized = name.lower().replace(" ", "-").replace("_", "-")
        if name_normalized in self._skills:
            m = self._skills[name_normalized]
            return MatchResult(
                manifest=m,
                skill_dir=self._skill_dirs[name_normalized],
                score=1.0,
            )
        # 尝试原始名称匹配
        for sid, manifest in self._skills.items():
            if manifest.name.lower() == name.lower():
                return MatchResult(
                    manifest=manifest,
                    skill_dir=self._skill_dirs[sid],
                    score=1.0,
                )
        return None

    def activate(self, match: MatchResult) -> SkillContent:
        """为已激活的匹配加载技能内容。

        加载 SKILL.md 和即时参考。
        """
        return self._loader.load(match.manifest, match.skill_dir)

    def get_skill_dir(self, skill_id: str) -> Path | None:
        """获取已注册技能的目录。"""
        return self._skill_dirs.get(skill_id)

    def _evaluate_trigger(
        self, trigger: TriggerRule, user_input: str, skill_id: str
    ) -> float:
        """对单个触发器与用户输入进行评估，返回 0.0–1.0 的得分。"""
        if trigger.type == "prefix":
            prefix = trigger.value
            if not trigger.case_sensitive:
                if user_input.lower().startswith(prefix.lower()):
                    return 1.0
            else:
                if user_input.startswith(prefix):
                    return 1.0
            return 0.0

        if trigger.type == "keyword":
            keyword = trigger.value
            text = user_input if trigger.case_sensitive else user_input.lower()
            kw = keyword if trigger.case_sensitive else keyword.lower()
            if kw == text:
                return 1.0  # 精确匹配
            if kw in text:
                # 部分匹配 — 按覆盖率打分
                return min(0.9, len(kw) / max(len(text), 1) + 0.3)
            return 0.0

        if trigger.type == "regex":
            cache_key = f"{skill_id}:{trigger.value}"
            pattern = self._regex_cache.get(cache_key)
            if pattern is None:
                return 0.0
            # H8: 限制正则匹配时间，防止 ReDoS
            try:
                m = _safe_regex_search(pattern, user_input, timeout=1.0)
            except _RegexTimeoutError:
                logger.warning(
                    "技能 %r 的正则触发器 %r 匹配超时（可能的 ReDoS），跳过。",
                    skill_id,
                    trigger.value,
                )
                return 0.0
            if m:
                coverage = len(m.group()) / max(len(user_input), 1)
                return min(0.95, coverage + 0.4)
            return 0.0

        if trigger.type == "intent":
            # 意图匹配需要外部分类器 — 此处未实现。
            # 除非注册了意图分类器钩子，否则返回 0.0。
            logger.debug(
                "意图触发器 %r 需要外部分类器（未实现）。",
                trigger.value,
            )
            return 0.0

        logger.warning("未知触发器类型：%r", trigger.type)
        return 0.0

    def list_skills(self) -> list[dict[str, str]]:
        """返回所有已注册技能的摘要信息。"""
        return [
            {
                "skill_id": sid,
                "name": m.name,
                "description": m.description,
                "triggers": str(len(m.triggers)),
                "tags": ", ".join(m.tags),
            }
            for sid, m in self._skills.items()
        ]


# ---------------------------------------------------------------------------
# H8: 正则匹配超时保护
# ---------------------------------------------------------------------------

class _RegexTimeoutError(Exception):
    """正则表达式匹配超时。"""


def _safe_regex_search(
    pattern: re.Pattern[str], text: str, *, timeout: float = 1.0
) -> re.Match[str] | None:
    """带超时保护的正则搜索。

    在 Unix 上使用 SIGALRM 实现硬超时；在 Windows 或不支持 SIGALRM 的
    平台上回退为普通搜索（不限时）并限制输入长度作为缓解措施。
    """
    import sys

    # Windows 或其他不支持 SIGALRM 的平台：限制输入长度作为缓解
    if sys.platform == "win32" or not hasattr(_signal, "SIGALRM"):
        # 截断过长输入以降低 ReDoS 风险
        max_len = 10_000
        return pattern.search(text[:max_len])

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise _RegexTimeoutError(f"正则匹配超时 ({timeout}s)")

    old_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
    _signal.setitimer(_signal.ITIMER_REAL, timeout)
    try:
        return pattern.search(text)
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, old_handler)
