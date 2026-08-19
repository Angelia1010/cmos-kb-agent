"""runtime/signals + budget 测试。"""

import time
import unittest

from uniagent.runtime.signals import LoopSignal, LoopResult, HookResponse
from uniagent.runtime.budget import Budget, BudgetConfig


class TestLoopSignal(unittest.TestCase):
    def test_signal_values(self):
        self.assertIsNotNone(LoopSignal.CONTINUE)
        self.assertIsNotNone(LoopSignal.BREAK)
        self.assertIsNotNone(LoopSignal.RETRY)
        self.assertIsNotNone(LoopSignal.ROLLBACK)

    def test_all_different(self):
        signals = [LoopSignal.CONTINUE, LoopSignal.BREAK, LoopSignal.RETRY, LoopSignal.ROLLBACK]
        self.assertEqual(len(set(signals)), 4)


class TestLoopResult(unittest.TestCase):
    def test_success_result(self):
        r = LoopResult(success=True, iterations=3, reason="done", evidence="all tests pass")
        self.assertTrue(r.success)
        self.assertEqual(r.iterations, 3)

    def test_failure_result(self):
        r = LoopResult(success=False, iterations=5, reason="budget exhausted")
        self.assertFalse(r.success)

    def test_frozen(self):
        r = LoopResult(success=True, iterations=1)
        with self.assertRaises(AttributeError):
            r.success = False  # type: ignore


class TestHookResponse(unittest.TestCase):
    def test_default_continue(self):
        hr = HookResponse()
        self.assertEqual(hr.signal, LoopSignal.CONTINUE)
        self.assertEqual(hr.message, "")

    def test_break_with_message(self):
        hr = HookResponse(signal=LoopSignal.BREAK, message="stopped")
        self.assertEqual(hr.signal, LoopSignal.BREAK)


class TestBudgetConfig(unittest.TestCase):
    def test_defaults(self):
        bc = BudgetConfig()
        self.assertEqual(bc.max_iterations, 25)
        self.assertEqual(bc.max_tokens, 0)
        self.assertEqual(bc.max_time_seconds, 0)


class TestBudget(unittest.TestCase):
    def test_fresh_budget_continues(self):
        b = Budget()
        signal, reason = b.check()
        self.assertEqual(signal, LoopSignal.CONTINUE)
        self.assertEqual(reason, "")

    def test_iteration_limit(self):
        b = Budget(config=BudgetConfig(max_iterations=2))
        b.record_iteration()
        b.record_iteration()
        signal, reason = b.check()
        self.assertEqual(signal, LoopSignal.BREAK)
        self.assertIn("迭代", reason)

    def test_token_limit(self):
        b = Budget(config=BudgetConfig(max_tokens=100))
        b.record_tokens(150)
        signal, reason = b.check()
        self.assertEqual(signal, LoopSignal.BREAK)
        self.assertIn("Token", reason)

    def test_time_limit(self):
        b = Budget(config=BudgetConfig(max_time_seconds=0.01))
        time.sleep(0.02)
        signal, reason = b.check()
        self.assertEqual(signal, LoopSignal.BREAK)
        self.assertIn("时间", reason)

    def test_summary(self):
        b = Budget(config=BudgetConfig(max_iterations=10, max_tokens=500))
        b.record_iteration()
        b.record_tokens(100)
        s = b.summary()
        self.assertIn("iterations", s)
        self.assertIn("tokens", s)
        self.assertIn("time", s)

    def test_elapsed_seconds(self):
        b = Budget()
        time.sleep(0.01)
        self.assertGreater(b.elapsed_seconds(), 0)

    def test_record_iteration_increments(self):
        b = Budget()
        self.assertEqual(b.iterations_used, 0)
        b.record_iteration()
        self.assertEqual(b.iterations_used, 1)
        b.record_iteration()
        self.assertEqual(b.iterations_used, 2)


if __name__ == "__main__":
    unittest.main()
