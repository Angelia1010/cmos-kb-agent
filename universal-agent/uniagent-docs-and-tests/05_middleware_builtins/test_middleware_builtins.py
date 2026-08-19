"""内置中间件测试。"""

import asyncio
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from uniagent.middleware.builtins.dangling_tool_call import (
    DanglingToolCallMiddleware,
    _patch_dangling,
)
from uniagent.middleware.builtins.loop_detection import (
    LoopDetectionMiddleware,
    _signature,
)
from uniagent.middleware.builtins.token_usage import TokenUsageMiddleware, _extract_usage


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── DanglingToolCallMiddleware ──


class TestPatchDangling(unittest.TestCase):
    """悬空工具调用修补。"""

    def test_no_orphans_returns_same(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="ok")]
        result = _patch_dangling(msgs)
        self.assertIs(result, msgs)

    def test_patches_orphan(self):
        ai = AIMessage(content="", tool_calls=[{"id": "tc1", "name": "foo", "args": {}}])
        msgs = [HumanMessage(content="hi"), ai]
        result = _patch_dangling(msgs)
        self.assertIsNot(result, msgs)
        # 应在 ai 后插入一个合成 ToolMessage
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0].tool_call_id, "tc1")

    def test_existing_tool_msg_not_patched(self):
        ai = AIMessage(content="", tool_calls=[{"id": "tc1", "name": "foo", "args": {}}])
        tm = ToolMessage(content="result", tool_call_id="tc1", name="foo")
        msgs = [ai, tm]
        result = _patch_dangling(msgs)
        self.assertIs(result, msgs)  # 无孤立调用，返回原列表


class TestDanglingMiddleware(unittest.TestCase):
    def test_before_agent_patches(self):
        mw = DanglingToolCallMiddleware()
        ai = AIMessage(content="", tool_calls=[{"id": "tc2", "name": "bar", "args": {}}])
        state = {"messages": [ai]}
        result = _run(mw.before_agent(state))
        self.assertIsNotNone(result)
        self.assertGreater(len(result["messages"]), 1)

    def test_before_agent_no_patch_needed(self):
        mw = DanglingToolCallMiddleware()
        state = {"messages": [HumanMessage(content="hi")]}
        result = _run(mw.before_agent(state))
        self.assertIsNone(result)


# ── LoopDetectionMiddleware ──


class TestSignature(unittest.TestCase):
    def test_signature_deterministic(self):
        tc = [{"name": "search", "args": {"q": "hello"}}]
        s1 = _signature(tc)
        s2 = _signature(tc)
        self.assertEqual(s1, s2)

    def test_signature_different_args(self):
        tc1 = [{"name": "search", "args": {"q": "a"}}]
        tc2 = [{"name": "search", "args": {"q": "b"}}]
        self.assertNotEqual(_signature(tc1), _signature(tc2))


class TestLoopDetection(unittest.TestCase):
    def test_no_repeat_no_warning(self):
        mw = LoopDetectionMiddleware(hard_limit=3)
        ai = AIMessage(content="", tool_calls=[{"id": "c1", "name": "a", "args": {}}])
        state = {"messages": [ai]}
        result = _run(mw.before_agent(state))
        self.assertIsNone(result)  # 首次，无需干预

    def test_hard_limit_triggers_warning(self):
        mw = LoopDetectionMiddleware(hard_limit=2)
        tc = [{"id": "c1", "name": "search", "args": {"q": "x"}}]
        # 模拟连续 2 次相同调用
        for _ in range(2):
            ai = AIMessage(content="", tool_calls=tc)
            state = {"messages": [ai]}
            result = _run(mw.before_agent(state))
        # 第 2 次应触发警告注入
        self.assertIsNotNone(result)

    def test_different_calls_reset_count(self):
        mw = LoopDetectionMiddleware(hard_limit=3)
        ai1 = AIMessage(content="", tool_calls=[{"id": "c1", "name": "a", "args": {}}])
        ai2 = AIMessage(content="", tool_calls=[{"id": "c2", "name": "b", "args": {}}])
        _run(mw.before_agent({"messages": [ai1]}))
        result = _run(mw.before_agent({"messages": [ai2]}))
        self.assertIsNone(result)

    def test_exposes_loop_hooks(self):
        mw = LoopDetectionMiddleware()
        hooks = mw.loop_hooks()
        self.assertEqual(len(hooks), 1)


# ── TokenUsageMiddleware ──


class TestExtractUsage(unittest.TestCase):
    def test_from_usage_metadata(self):
        ai = AIMessage(content="hi")
        ai.usage_metadata = MagicMock(input_tokens=10, output_tokens=20)
        usage = _extract_usage(ai)
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 20)

    def test_from_response_metadata(self):
        ai = AIMessage(content="hi")
        ai.response_metadata = {
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 15}
        }
        usage = _extract_usage(ai)
        self.assertEqual(usage["prompt_tokens"], 5)

    def test_no_metadata_returns_none(self):
        ai = AIMessage(content="hi")
        self.assertIsNone(_extract_usage(ai))


class TestTokenUsageMiddleware(unittest.TestCase):
    def test_accumulates(self):
        mw = TokenUsageMiddleware()
        ai = AIMessage(content="hi")
        ai.usage_metadata = MagicMock(input_tokens=100, output_tokens=50)
        state = {"messages": [ai]}
        result = _run(mw.after_agent(state))
        self.assertIsNotNone(result)
        self.assertEqual(result["token_usage"]["total_tokens"], 150)

    def test_no_ai_messages(self):
        mw = TokenUsageMiddleware()
        state = {"messages": [HumanMessage(content="hi")]}
        result = _run(mw.after_agent(state))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
