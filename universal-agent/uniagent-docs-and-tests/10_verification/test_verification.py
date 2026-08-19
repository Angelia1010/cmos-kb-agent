"""verification 模块测试 —— 验证器协议与内置验证器。"""

import asyncio
import unittest
from typing import Any

from uniagent.verification.verifier import Verifier, VerificationResult
from uniagent.verification.builtins.always_pass import AlwaysPassVerifier
from uniagent.verification.builtins.composite_verifier import CompositeVerifier


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestVerificationResult(unittest.TestCase):
    def test_passed(self):
        r = VerificationResult(passed=True, evidence="ok", confidence=0.9, layer="test")
        self.assertTrue(r.passed)
        self.assertEqual(r.confidence, 0.9)

    def test_failed(self):
        r = VerificationResult(passed=False, evidence="missing", layer="lint")
        self.assertFalse(r.passed)

    def test_frozen(self):
        r = VerificationResult(passed=True)
        with self.assertRaises(AttributeError):
            r.passed = False  # type: ignore

    def test_defaults(self):
        r = VerificationResult(passed=True)
        self.assertEqual(r.evidence, "")
        self.assertEqual(r.confidence, 1.0)
        self.assertEqual(r.layer, "")
        self.assertEqual(r.details, {})


class TestAlwaysPassVerifier(unittest.TestCase):
    def test_always_passes(self):
        v = AlwaysPassVerifier()
        result = _run(v.verify("any goal", {}))
        self.assertTrue(result.passed)
        self.assertEqual(result.layer, "noop")

    def test_protocol_check(self):
        v = AlwaysPassVerifier()
        self.assertIsInstance(v, Verifier)


class TestCompositeVerifier(unittest.TestCase):
    def test_all_pass(self):
        v = CompositeVerifier([AlwaysPassVerifier(), AlwaysPassVerifier()])
        result = _run(v.verify("goal", {}))
        self.assertTrue(result.passed)
        self.assertIn("all_passed", result.layer)

    def test_first_fail_short_circuits(self):
        class FailVerifier:
            async def verify(self, goal: str, state: dict) -> VerificationResult:
                return VerificationResult(passed=False, evidence="fail", layer="lint")

        class NeverCalled:
            called = False
            async def verify(self, goal: str, state: dict) -> VerificationResult:
                NeverCalled.called = True
                return VerificationResult(passed=True, layer="test")

        v = CompositeVerifier([FailVerifier(), NeverCalled()])
        result = _run(v.verify("goal", {}))
        self.assertFalse(result.passed)
        self.assertIn("lint", result.layer)
        self.assertFalse(NeverCalled.called)

    def test_empty_verifiers_pass(self):
        v = CompositeVerifier([])
        result = _run(v.verify("goal", {}))
        self.assertTrue(result.passed)

    def test_evidence_aggregated(self):
        v = CompositeVerifier([AlwaysPassVerifier(), AlwaysPassVerifier()])
        result = _run(v.verify("goal", {}))
        self.assertIn("通过", result.evidence)


class TestCustomVerifier(unittest.TestCase):
    """自定义验证器满足 Protocol。"""

    def test_custom_implements_protocol(self):
        class MyVerifier:
            async def verify(self, goal: str, state: dict[str, Any]) -> VerificationResult:
                return VerificationResult(passed=True, evidence="custom", layer="custom")

        v = MyVerifier()
        self.assertIsInstance(v, Verifier)
        result = _run(v.verify("x", {}))
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
