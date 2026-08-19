"""skills 模块测试 —— 清单、加载器、注入器、注册表。"""

import json
import tempfile
import unittest
from pathlib import Path

from uniagent.skills.manifest import SkillManifest, TriggerRule, ReferenceEntry
from uniagent.skills.loader import SkillLoader, SkillContent
from uniagent.skills.injector import SkillInjector
from uniagent.skills.registry import SkillRegistry, MatchResult


def _create_skill_dir(
    tmpdir: str,
    name: str = "test-skill",
    triggers: list = None,
    skill_md: str = "# Test Skill\nDo something.",
) -> tuple[Path, SkillManifest]:
    """在临时目录下创建一个技能包。"""
    skill_dir = Path(tmpdir) / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "name": name,
        "description": f"A test skill: {name}",
        "triggers": triggers or [{"type": "keyword", "value": name}],
        "tags": ["test"],
    }
    (skill_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    # 创建 references 目录
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "guide.md").write_text("# Guide\nSome reference.", encoding="utf-8")

    manifest = SkillManifest.from_json(skill_dir / "metadata.json")
    return skill_dir, manifest


# ── Manifest 测试 ──


class TestTriggerRule(unittest.TestCase):
    def test_from_dict(self):
        tr = TriggerRule.from_dict({"type": "keyword", "value": "hello"})
        self.assertEqual(tr.type, "keyword")
        self.assertEqual(tr.value, "hello")
        self.assertFalse(tr.case_sensitive)

    def test_defaults(self):
        tr = TriggerRule.from_dict({"value": "x"})
        self.assertEqual(tr.type, "keyword")


class TestSkillManifest(unittest.TestCase):
    def test_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "my-skill")
            self.assertEqual(manifest.name, "my-skill")
            self.assertEqual(manifest.skill_id, "my-skill")
            self.assertEqual(len(manifest.triggers), 1)

    def test_skill_id_normalization(self):
        m = SkillManifest(name="My Cool Skill")
        self.assertEqual(m.skill_id, "my-cool-skill")

    def test_extra_fields_preserved(self):
        data = {"name": "x", "custom_field": "hello"}
        m = SkillManifest.from_dict(data)
        self.assertEqual(m.extra.get("custom_field"), "hello")


# ── Loader 测试 ──


class TestSkillLoader(unittest.TestCase):
    def test_load_skill_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "loader-test")
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)
            self.assertIn("Test Skill", content.instruction)

    def test_load_reference_on_demand(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "ref-test")
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)
            text = loader.load_reference(content, skill_dir, "guide.md")
            self.assertIsNotNone(text)
            self.assertIn("Guide", text)

    def test_load_missing_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "missing-ref")
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)
            text = loader.load_reference(content, skill_dir, "nonexistent.md")
            self.assertIsNone(text)

    def test_no_skill_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "no-md", skill_md="")
            (skill_dir / "SKILL.md").unlink()
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)
            self.assertEqual(content.instruction, "")


# ── Injector 测试 ──


class TestSkillInjector(unittest.TestCase):
    def test_no_active_returns_base(self):
        inj = SkillInjector()
        result = inj.inject("base prompt")
        self.assertEqual(result, "base prompt")

    def test_activate_and_inject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "inject-test")
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)

            inj = SkillInjector()
            inj.activate(content)
            result = inj.inject("base prompt")
            self.assertIn("inject-test", result)
            self.assertIn("Test Skill", result)

    def test_dedup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "dup-test")
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)

            inj = SkillInjector()
            inj.activate(content)
            inj.activate(content)  # 重复激活
            self.assertEqual(len(inj.active_skills), 1)

    def test_max_active_eviction(self):
        inj = SkillInjector(max_active_skills=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                skill_dir, manifest = _create_skill_dir(tmpdir, f"skill-{i}")
                loader = SkillLoader()
                content = loader.load(manifest, skill_dir)
                inj.activate(content)
            self.assertEqual(len(inj.active_skills), 2)

    def test_deactivate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "deact-test")
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)

            inj = SkillInjector()
            inj.activate(content)
            self.assertTrue(inj.deactivate("deact-test"))
            self.assertEqual(len(inj.active_skills), 0)

    def test_clear(self):
        inj = SkillInjector()
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir, manifest = _create_skill_dir(tmpdir, "clear-test")
            loader = SkillLoader()
            content = loader.load(manifest, skill_dir)
            inj.activate(content)
            inj.clear()
            self.assertEqual(len(inj.active_skills), 0)


# ── Registry 测试 ──


class TestSkillRegistry(unittest.TestCase):
    def test_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "skill-a")
            _create_skill_dir(tmpdir, "skill-b")
            reg = SkillRegistry()
            count = reg.scan(tmpdir)
            self.assertEqual(count, 2)
            self.assertEqual(len(reg.skills), 2)

    def test_match_keyword(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "taocan", triggers=[
                {"type": "keyword", "value": "套餐"}
            ])
            reg = SkillRegistry()
            reg.scan(tmpdir)
            matches = reg.match("我想查看套餐详情")
            self.assertGreater(len(matches), 0)
            self.assertEqual(matches[0].manifest.name, "taocan")

    def test_match_no_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "kuandai", triggers=[
                {"type": "keyword", "value": "宽带"}
            ])
            reg = SkillRegistry()
            reg.scan(tmpdir)
            matches = reg.match("完全无关的内容")
            self.assertEqual(len(matches), 0)

    def test_match_by_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "my-skill")
            reg = SkillRegistry()
            reg.scan(tmpdir)
            result = reg.match_by_name("my-skill")
            self.assertIsNotNone(result)
            self.assertEqual(result.score, 1.0)

    def test_match_by_name_not_found(self):
        reg = SkillRegistry()
        result = reg.match_by_name("nonexistent")
        self.assertIsNone(result)

    def test_unregister(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "removable")
            reg = SkillRegistry()
            reg.scan(tmpdir)
            self.assertTrue(reg.unregister("removable"))
            self.assertEqual(len(reg.skills), 0)
            self.assertFalse(reg.unregister("removable"))  # 已移除

    def test_activate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "act-skill")
            reg = SkillRegistry()
            reg.scan(tmpdir)
            matches = reg.match("act-skill")
            self.assertGreater(len(matches), 0)
            content = reg.activate(matches[0])
            self.assertIn("Test Skill", content.instruction)

    def test_regex_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "regex-skill", triggers=[
                {"type": "regex", "value": r"\d+元套餐"}
            ])
            reg = SkillRegistry()
            reg.scan(tmpdir)
            matches = reg.match("请查询99元套餐")
            self.assertGreater(len(matches), 0)

    def test_prefix_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "prefix-skill", triggers=[
                {"type": "prefix", "value": "/help"}
            ])
            reg = SkillRegistry()
            reg.scan(tmpdir)
            matches = reg.match("/help me")
            self.assertGreater(len(matches), 0)

    def test_list_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_skill_dir(tmpdir, "list-test")
            reg = SkillRegistry()
            reg.scan(tmpdir)
            skills = reg.list_skills()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0]["skill_id"], "list-test")


if __name__ == "__main__":
    unittest.main()
