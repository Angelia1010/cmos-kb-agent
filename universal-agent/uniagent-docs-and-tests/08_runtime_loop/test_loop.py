"""runtime/loop 测试 —— TurnLoop / GoalLoop。

使用 Mock Agent 模拟 LangGraph 智能体行为，不依赖真实 LLM。
"""

import asyncio
import unittest
from typing import Any

from uniagent.runtime.loop import TurnLoop, GoalLoop
from uniagent.runtime.budget import Budget, BudgetConfig
from uniagent.runtime.hooks import LoopHook
from uniagent.runtime.signals import HookResponse, LoopSignal
from uniagent.verification.verifier import VerificationResult


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class MockAgent:
    """模拟 LangGraph agent，支持自定义 ainvoke 返回值。"""

    def __init__(self, responses=None):
        self._responses = responses or [{}]
        self._call_count = 0
        self._uniagent_middleware = []

    async def ainvoke(self, state, config=None):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        result = dict(state)
        result.update(self._responses[idx])
        return result


class MockVerifier:
    """模拟验证器：前 N 次 fail，之后 pass。"""

    def __init__(self, pass_after=0):
        self._pass_after = pass_after
        self._call_count = 0

    async def verify(self, goal, state):
        self._call_count += 1
        if self._call_count > self._pass_after:
            return VerificationResult(passed=True, evidence="verified", layer="mock")
        return VerificationResult(passed=False, evidence="not yet", layer="mock")


# ── TurnLoop 测试 ──


class TestTurnLoop(unittest.TestCase):
    def test_runs_max_iterations(self):
        agent = MockAgent()
        budget = Budget(config=BudgetConfig(max_iterations=3))
        loop = TurnLoop(agent=agent, budget=budget)
        result = _run(loop.run())
        self.assertEqual(result.iterations, 3)
        self.assertFalse(result.success)

    def test_hook_can_break_early(self):
        class BreakHook(LoopHook):
            async def on_iteration_end(self, iteration, state, output):
                if iteration >= 1:
                    return HookResponse(signal=LoopSignal.BREAK, message="enough")
                return HookResponse()

        agent = MockAgent()
        budget = Budget(config=BudgetConfig(max_iterations=10))
        loop = TurnLoop(agent=agent, hooks=[BreakHook()], budget=budget)
        result = _run(loop.run())
        self.assertTrue(result.success)  # BREAK from on_iteration_end → success
        self.assertEqual(result.iterations, 2)

    def test_with_initial_state(self):
        agent = MockAgent()
        budget = Budget(config=BudgetConfig(max_iterations=1))
        loop = TurnLoop(agent=agent, budget=budget)
        result = _run(loop.run(initial_state={"custom_key": "value"}))
        self.assertIn("custom_key", result.final_state)

    def test_budget_time_breaks(self):
        import time
        agent = MockAgent()
        budget = Budget(config=BudgetConfig(max_iterations=100, max_time_seconds=0.001))
        time.sleep(0.01)  # 确保超时
        loop = TurnLoop(agent=agent, budget=budget)
        result = _run(loop.run())
        self.assertFalse(result.success)
        self.assertIn("时间", result.reason)


# ── GoalLoop 测试 ──


class TestGoalLoop(unittest.TestCase):
    def test_passes_on_first_verify(self):
        agent = MockAgent()
        verifier = MockVerifier(pass_after=0)
        budget = Budget(config=BudgetConfig(max_iterations=5))
        loop = GoalLoop(agent=agent, goal="test goal", verifier=verifier, budget=budget)
        result = _run(loop.run())
        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 1)
        self.assertIn("验证通过", result.reason)

    def test_verify_after_retry(self):
        """验证失败后注入反馈，第二轮通过。"""
        agent = MockAgent()
        verifier = MockVerifier(pass_after=1)  # 第1次fail，第2次pass
        budget = Budget(config=BudgetConfig(max_iterations=5))
        loop = GoalLoop(agent=agent, goal="test goal", verifier=verifier, budget=budget)
        result = _run(loop.run())
        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 2)

    def test_budget_exhausted(self):
        agent = MockAgent()
        verifier = MockVerifier(pass_after=999)  # 永不通过
        budget = Budget(config=BudgetConfig(max_iterations=3))
        loop = GoalLoop(agent=agent, goal="impossible", verifier=verifier, budget=budget)
        result = _run(loop.run())
        self.assertFalse(result.success)
        self.assertEqual(result.iterations, 3)

    def test_goal_injection(self):
        """目标应作为 SystemMessage 注入到 messages 中。"""
        agent = MockAgent()
        verifier = MockVerifier(pass_after=0)
        budget = Budget(config=BudgetConfig(max_iterations=1))
        loop = GoalLoop(agent=agent, goal="find the answer", verifier=verifier, budget=budget)
        result = _run(loop.run())
        # 检查 final_state 的 messages 中包含目标
        msgs = result.final_state.get("messages", [])
        goal_found = any("find the answer" in str(getattr(m, "content", "")) for m in msgs)
        self.assertTrue(goal_found)

    def test_verify_every(self):
        """verify_every=2 时，仅偶数轮次验证。"""
        agent = MockAgent()
        verifier = MockVerifier(pass_after=0)  # 一旦验证就通过
        budget = Budget(config=BudgetConfig(max_iterations=5))
        loop = GoalLoop(
            agent=agent, goal="test", verifier=verifier,
            budget=budget, verify_every=2,
        )
        result = _run(loop.run())
        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 2)  # 第2次迭代才验证

    def test_feedback_injected_on_failure(self):
        """验证失败时应注入反馈 HumanMessage。"""
        call_states = []

        class TrackingAgent(MockAgent):
            async def ainvoke(self, state, config=None):
                call_states.append(dict(state))
                return await super().ainvoke(state, config)

        agent = TrackingAgent()
        verifier = MockVerifier(pass_after=1)
        budget = Budget(config=BudgetConfig(max_iterations=5))
        loop = GoalLoop(agent=agent, goal="do it", verifier=verifier, budget=budget)
        _run(loop.run())
        # 第二次调用的 state 中应包含验证失败反馈
        if len(call_states) >= 2:
            msgs = call_states[1].get("messages", [])
            feedback_found = any("验证失败" in str(getattr(m, "content", "")) for m in msgs)
            self.assertTrue(feedback_found)


if __name__ == "__main__":
    unittest.main()
