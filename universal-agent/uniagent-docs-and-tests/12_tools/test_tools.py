"""tools 模块测试 —— 工具注册表。"""

import unittest
from unittest.mock import MagicMock

from uniagent.config.app_config import AppConfig
from uniagent.config.sub_configs import ToolConfig
from uniagent.tools.registry import get_available_tools, _tool_name


class FakeTool:
    """模拟工具对象。"""
    def __init__(self, name="fake"):
        self.name = name


class TestToolName(unittest.TestCase):
    def test_from_name_attr(self):
        tool = FakeTool("search")
        self.assertEqual(_tool_name(tool), "search")

    def test_from_callable(self):
        def my_func(): pass
        self.assertEqual(_tool_name(my_func), "my_func")

    def test_fallback_to_id(self):
        obj = object()
        name = _tool_name(obj)
        self.assertTrue(name)  # 非空


class TestGetAvailableTools(unittest.TestCase):
    def test_empty_config(self):
        cfg = AppConfig()
        tools = get_available_tools(cfg)
        self.assertEqual(tools, [])

    def test_extra_tools(self):
        cfg = AppConfig()
        extra = [FakeTool("tool_a"), FakeTool("tool_b")]
        tools = get_available_tools(cfg, extra_tools=extra)
        self.assertEqual(len(tools), 2)

    def test_dedup_by_name(self):
        cfg = AppConfig()
        extra = [FakeTool("dup"), FakeTool("dup")]
        tools = get_available_tools(cfg, extra_tools=extra)
        self.assertEqual(len(tools), 1)

    def test_excluded_names(self):
        cfg = AppConfig()
        extra = [FakeTool("keep"), FakeTool("exclude")]
        tools = get_available_tools(cfg, extra_tools=extra, excluded_names={"exclude"})
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "keep")

    def test_mcp_tools(self):
        cfg = AppConfig()
        mcp = [FakeTool("mcp_tool")]
        tools = get_available_tools(cfg, mcp_tools=mcp)
        self.assertEqual(len(tools), 1)

    def test_priority_extra_over_mcp(self):
        """同名工具 extra_tools 优先于 mcp_tools。"""
        cfg = AppConfig()
        extra = [FakeTool("same")]
        mcp = [FakeTool("same")]
        tools = get_available_tools(cfg, extra_tools=extra, mcp_tools=mcp)
        self.assertEqual(len(tools), 1)
        self.assertIs(tools[0], extra[0])

    def test_config_tool_bad_path(self):
        """配置中无效的工具路径应被跳过，不崩溃。"""
        cfg = AppConfig(tools=[
            ToolConfig(name="bad", use="nonexistent.module:BadTool"),
        ])
        tools = get_available_tools(cfg)
        self.assertEqual(len(tools), 0)  # 加载失败但不崩溃

    def test_config_tool_disabled(self):
        cfg = AppConfig(tools=[
            ToolConfig(name="disabled", use="json:dumps", enabled=False),
        ])
        tools = get_available_tools(cfg)
        self.assertEqual(len(tools), 0)


if __name__ == "__main__":
    unittest.main()
