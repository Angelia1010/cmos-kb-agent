"""runtime/hooks 测试 —— LoopHook 基类与内置钩子。"""

import asyncio
import unittest

from uniagent.runtime.hooks import LoopHook, ProgressLogHook, TokenBudgetHook
from uniagent.runtime.signals import HookResponse, LoopSignal
from uniagent.runtime.budget import Budget, BudgetConfig


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestLoopHookBase(unittest.TestCase):
    """LoopHook 基类默认行为。"""

    def test_default_iteration_start_continues(self):
        class MyHook(LoopHook):
            pass
        hook = MyHook()
        resp = _run(hook.on_iteration_start(0, {}))
        self.assertEqual(resp.signal, LoopSignal.CONTINUE)

    def test_default_iteration_end_continues(self):
        class MyHook(LoopHook):
            pass
        hook = MyHook()
        resp = _run(hook.on_iteration_end(0, {}, None))
        self.assertEqual(resp.signal, LoopSignal.CONTINUE)

    def test_default_on_error_breaks(self):
        class MyHook(LoopHook):
            pass
        hook = MyHook()
        resp = _run(hook.on_error({}, RuntimeError("boom")))
        self.assertEqual(resp.signal, LoopSignal.BREAK)

    def test_auto_name(self):
        class CustomHook(LoopHook):
            pass
        self.assertEqual(CustomHook.name, "CustomHook")


class TestProgressLogHook(unittest.TestCase):
    """ProgressLogHook: 仅验证不崩溃，日志输出。"""

    def test_iteration_start(self):
        hook = ProgressLogHook()
        resp = _run(hook.on_iteration_start(0, {}))
        self.assertEqual(resp.signal, LoopSignal.CONTINUE)

    def test_iteration_end(self):
        hook = ProgressLogHook()
        resp = _run(hook.on_iteration_end(0, {"token_usage": {"total_tokens": 42}}, None))
        self.assertEqual(resp.signal, LoopSignal.CONTINUE)

    def test_goal_achieved(self):
        hook = ProgressLogHook()
        _run(hook.on_goal_achieved({}, "all tests passed"))  # 不应抛异常

    def test_budget_exhausted(self):
        hook = ProgressLogHook()
        _run(hook.on_budget_exhausted({}, "too many iterations"))


class TestTokenBudgetHook(unittest.TestCase):
    """TokenBudgetHook: 将 token 用量同步到 Budget。"""

    def test_syncs_tokens(self):
        budget = Budget(config=BudgetConfig(max_tokens=1000))
        hook = TokenBudgetHook(budget)
        state = {"token_usage": {"total_tokens": 500}}
        _run(hook.on_iteration_end(0, state, None))
        self.assertEqual(budget.tokens_used, 500)

    def test_no_token_usage_no_sync(self):
        budget = Budget()
        hook = TokenBudgetHook(budget)
        _run(hook.on_iteration_end(0, {}, None))
        self.assertEqual(budget.tokens_used, 0)

    def test_absolute_sync(self):
        """同步是绝对值，不是增量。"""
        budget = Budget(config=BudgetConfig(max_tokens=1000))
        hook = TokenBudgetHook(budget)
        _run(hook.on_iteration_end(0, {"token_usage": {"total_tokens": 100}}, None))
        self.assertEqual(budget.tokens_used, 100)
        _run(hook.on_iteration_end(1, {"token_usage": {"total_tokens": 200}}, None))
        self.assertEqual(budget.tokens_used, 200)  # 绝对值，非 300


class TestCustomHook(unittest.TestCase):
    """自定义钩子示例：在第 N 次迭代时 BREAK。"""

    def test_break_at_iteration_3(self):
        class BreakAt3(LoopHook):
            async def on_iteration_start(self, iteration, state):
                if iteration >= 3:
                    return HookResponse(signal=LoopSignal.BREAK, message="stop at 3")
                return HookResponse()

        hook = BreakAt3()
        resp = _run(hook.on_iteration_start(2, {}))
        self.assertEqual(resp.signal, LoopSignal.CONTINUE)
        resp = _run(hook.on_iteration_start(3, {}))
        self.assertEqual(resp.signal, LoopSignal.BREAK)


if __name__ == "__main__":
    unittest.main()
