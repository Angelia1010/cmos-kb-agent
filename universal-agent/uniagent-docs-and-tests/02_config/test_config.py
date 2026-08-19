"""config 模块测试 —— YAML配置加载、环境变量展开、热重载边界、ContextVar栈。"""

import os
import json
import tempfile
import unittest
from pathlib import Path

from uniagent.config.app_config import (
    AppConfig,
    _substitute_env,
    _walk_substitute,
    get_app_config,
    push_current_app_config,
    pop_current_app_config,
)
from uniagent.config.sub_configs import (
    ModelConfig,
    ToolConfig,
    LoopConfig,
    SkillConfig,
)
from uniagent.config.reload_boundary import (
    check_reload_safety,
    register_startup_only,
    get_startup_only_fields,
)


class TestEnvSubstitution(unittest.TestCase):
    """环境变量占位符展开。"""

    def test_substitute_with_value(self):
        os.environ["_TEST_UNIAGENT_VAR"] = "hello"
        result = _substitute_env("prefix-${_TEST_UNIAGENT_VAR}-suffix")
        self.assertEqual(result, "prefix-hello-suffix")
        del os.environ["_TEST_UNIAGENT_VAR"]

    def test_substitute_with_default(self):
        result = _substitute_env("${_NONEXIST_VAR_XYZ:fallback}")
        self.assertEqual(result, "fallback")

    def test_substitute_missing_no_default(self):
        """无环境变量且无默认值时保留原样"""
        result = _substitute_env("${_NONEXIST_VAR_ABC}")
        self.assertEqual(result, "${_NONEXIST_VAR_ABC}")

    def test_walk_substitute_recursive(self):
        os.environ["_TEST_WALK"] = "val"
        data = {"a": "${_TEST_WALK}", "b": ["${_TEST_WALK}"], "c": 42}
        result = _walk_substitute(data)
        self.assertEqual(result, {"a": "val", "b": ["val"], "c": 42})
        del os.environ["_TEST_WALK"]


class TestAppConfigDefaults(unittest.TestCase):
    """AppConfig 默认值。"""

    def test_defaults(self):
        cfg = AppConfig()
        self.assertEqual(cfg.log_level, "info")
        self.assertEqual(cfg.models, [])
        self.assertEqual(cfg.tools, [])
        self.assertEqual(cfg.loop.max_iterations, 25)
        self.assertFalse(cfg.skills.enabled)

    def test_from_yaml(self):
        """从临时 YAML 文件加载配置。"""
        yaml_content = """\
log_level: debug
loop:
  max_iterations: 5
  max_time_seconds: 10.0
skills:
  enabled: true
  directories:
    - my_skills
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            cfg = AppConfig.from_yaml(f.name)

        self.assertEqual(cfg.log_level, "debug")
        self.assertEqual(cfg.loop.max_iterations, 5)
        self.assertEqual(cfg.loop.max_time_seconds, 10.0)
        self.assertTrue(cfg.skills.enabled)
        self.assertIn("my_skills", cfg.skills.directories)
        os.unlink(f.name)


class TestSubConfigs(unittest.TestCase):
    """子配置模型。"""

    def test_model_config(self):
        mc = ModelConfig(use="langchain_openai:ChatOpenAI", model="gpt-4o")
        self.assertEqual(mc.temperature, 0.0)
        self.assertEqual(mc.name, "default")

    def test_tool_config(self):
        tc = ToolConfig(name="search", use="mytools:search_tool")
        self.assertTrue(tc.enabled)
        self.assertEqual(tc.kwargs, {})

    def test_loop_config_validation(self):
        lc = LoopConfig(max_iterations=1, max_tokens=0)
        self.assertEqual(lc.max_iterations, 1)

    def test_skill_config(self):
        sc = SkillConfig(enabled=True, max_active=5)
        self.assertTrue(sc.auto_match)
        self.assertEqual(sc.max_active, 5)


class TestReloadBoundary(unittest.TestCase):
    """热重载边界安全检查。"""

    def test_no_violations(self):
        old = {"sandbox": "a", "loop": "b"}
        new = {"sandbox": "a", "loop": "c"}
        violations = check_reload_safety(old, new)
        self.assertEqual(violations, [])

    def test_detects_startup_field_change(self):
        old = {"sandbox": "a"}
        new = {"sandbox": "b"}
        violations = check_reload_safety(old, new)
        self.assertIn("sandbox", violations)


class TestContextVarStack(unittest.TestCase):
    """ContextVar 配置栈隔离。"""

    def test_push_pop(self):
        cfg1 = AppConfig(log_level="debug")
        push_current_app_config(cfg1)
        # 栈顶应是 cfg1
        got = get_app_config()
        self.assertEqual(got.log_level, "debug")
        popped = pop_current_app_config()
        self.assertIs(popped, cfg1)

    def test_pop_empty_raises(self):
        with self.assertRaises(RuntimeError):
            pop_current_app_config()


if __name__ == "__main__":
    unittest.main()
