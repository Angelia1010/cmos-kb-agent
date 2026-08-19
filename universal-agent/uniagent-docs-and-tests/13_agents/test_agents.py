"""agents 模块测试 —— AgentFeatures + create_agent 工厂。"""

import unittest
from unittest.mock import MagicMock, patch

from uniagent.agents.features import AgentFeatures
from uniagent.middleware.base import Middleware
from uniagent.middleware.builtins import (
    DanglingToolCallMiddleware,
    ToolErrorHandlingMiddleware,
    LoopDetectionMiddleware,
    TokenUsageMiddleware,
    SkillMiddleware,
)


# ── AgentFeatures 测试 ──


class TestAgentFeatures(unittest.TestCase):
    def test_defaults(self):
        f = AgentFeatures()
        self.assertTrue(f.dangling_tool_call)
        self.assertTrue(f.tool_error_handling)
        self.assertTrue(f.loop_detection)
        self.assertTrue(f.token_usage)
        self.assertFalse(f.skill)
        self.assertFalse(f.goal_loop)

    def test_resolve_all_defaults(self):
        """默认特性应解析出 4 个中间件（skill 默认关）。"""
        f = AgentFeatures()
        chain = f.resolve_middleware()
        self.assertEqual(len(chain), 4)
        types = {type(m) for m in chain}
        self.assertIn(DanglingToolCallMiddleware, types)
        self.assertIn(ToolErrorHandlingMiddleware, types)
        self.assertIn(LoopDetectionMiddleware, types)
        self.assertIn(TokenUsageMiddleware, types)

    def test_disable_feature(self):
        f = AgentFeatures(loop_detection=False)
        chain = f.resolve_middleware()
        types = {type(m) for m in chain}
        self.assertNotIn(LoopDetectionMiddleware, types)

    def test_custom_middleware_instance(self):
        custom = LoopDetectionMiddleware(hard_limit=10)
        f = AgentFeatures(loop_detection=custom)
        chain = f.resolve_middleware()
        loop_mw = [m for m in chain if isinstance(m, LoopDetectionMiddleware)]
        self.assertEqual(len(loop_mw), 1)
        self.assertIs(loop_mw[0], custom)

    def test_enable_skill(self):
        f = AgentFeatures(skill=True)
        chain = f.resolve_middleware()
        types = {type(m) for m in chain}
        self.assertIn(SkillMiddleware, types)

    def test_resolve_order(self):
        """解析顺序应为: skill → dangling → error → loop → token。"""
        f = AgentFeatures(skill=True)
        chain = f.resolve_middleware()
        names = [type(m).__name__ for m in chain]
        # skill 应在最前
        self.assertEqual(names[0], "SkillMiddleware")
        # token_usage 应在最后
        self.assertEqual(names[-1], "TokenUsageMiddleware")

    def test_default_loop_hooks(self):
        f = AgentFeatures()
        hooks = f.default_loop_hooks()
        self.assertGreater(len(hooks), 0)


# ── create_agent 集成测试 ──


class TestCreateAgent(unittest.TestCase):
    """create_agent 工厂的基本行为（需要 langgraph 可用）。"""

    def test_bare_agent(self):
        """不设 goal → 返回 CompiledGraph。"""
        from langchain_core.language_models import BaseChatModel
        from uniagent.agents.factory import create_agent

        model = MagicMock(spec=BaseChatModel)
        agent = create_agent(model, tools=[], name="test_bare")
        # 裸 Agent 应有 invoke 方法
        self.assertTrue(hasattr(agent, "invoke") or hasattr(agent, "ainvoke"))

    def test_with_goal_returns_goal_loop(self):
        """设 goal + verifier → 返回 GoalLoop。"""
        from langchain_core.language_models import BaseChatModel
        from uniagent.agents.factory import create_agent
        from uniagent.runtime.loop import GoalLoop
        from uniagent.verification.builtins.always_pass import AlwaysPassVerifier

        model = MagicMock(spec=BaseChatModel)
        result = create_agent(
            model, tools=[],
            goal="test goal",
            verifier=AlwaysPassVerifier(),
        )
        self.assertIsInstance(result, GoalLoop)

    def test_middleware_and_features_exclusive(self):
        """同时指定 middleware 和 features 应报错。"""
        from langchain_core.language_models import BaseChatModel
        from uniagent.agents.factory import create_agent

        model = MagicMock(spec=BaseChatModel)
        with self.assertRaises(ValueError):
            create_agent(
                model, tools=[],
                middleware=[],
                features=AgentFeatures(),
            )

    def test_goal_without_verifier_uses_always_pass(self):
        """设 goal 但不设 verifier → 使用 AlwaysPassVerifier（附带警告）。"""
        from langchain_core.language_models import BaseChatModel
        from uniagent.agents.factory import create_agent
        from uniagent.runtime.loop import GoalLoop

        model = MagicMock(spec=BaseChatModel)
        result = create_agent(model, tools=[], goal="no verifier test")
        self.assertIsInstance(result, GoalLoop)


if __name__ == "__main__":
    unittest.main()
