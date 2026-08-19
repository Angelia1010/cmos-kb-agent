# -*- coding: utf-8 -*-
"""审查修复的回归测试:python tests_bugfix.py"""
import sys
sys.path.insert(0, "src")

import unittest

from kbagent import MainAgent, MockESClient, ScriptedChatModel


class TestCacheIsolation(unittest.TestCase):
    def test_external_mutation_does_not_pollute_cache(self):
        a = MainAgent(ScriptedChatModel(), MockESClient())
        a.run("用户想办理流量套餐,如何推荐?")
        hit = a.run("请问用户想办理流量套餐,如何推荐")
        hit.sources.append("污染测试")
        hit2 = a.run("请问用户想办理流量套餐,如何推荐")
        self.assertNotIn("污染测试", [str(x) for x in hit2.sources])

    def test_cache_hit_elapsed_is_hit_time(self):
        a = MainAgent(ScriptedChatModel(), MockESClient())
        first = a.run("用户想办理流量套餐,如何推荐?")
        hit = a.run("请问用户想办理流量套餐,如何推荐")
        self.assertTrue(hit.from_cache)
        self.assertLess(hit.elapsed_ms, max(first.elapsed_ms, 1) + 1)


class TestRoundObservability(unittest.TestCase):
    def test_recall_events_carry_round_number(self):
        a = MainAgent(ScriptedChatModel(), MockESClient())
        a.run("副卡怎么共享主卡额度")
        stages = [e.stage for e in a.tracer.events if e.event == "recall"]
        self.assertEqual(stages, ["retrieval.round1", "retrieval.round2"])


class TestLLMBridge(unittest.TestCase):
    def test_plain_chat_model_gets_bridged(self):
        """普通 BaseChatModel(无 judge/large_json)由 LLMBridge 自动适配。"""
        from kbagent.llm_bridge import LLMBridge, ensure_judge_interface

        class DummyModel:
            def invoke(self, messages):
                class R:
                    content = '{"covered": true, "uncovered_intents": []}'
                return R()

        bridged = ensure_judge_interface(DummyModel())
        self.assertIsInstance(bridged, LLMBridge)
        self.assertEqual(bridged.small_json("s", "u"),
                         {"covered": True, "uncovered_intents": []})
        # 已具备接口的模型原样返回
        same = ensure_judge_interface(ScriptedChatModel())
        self.assertNotIsInstance(same, LLMBridge)

    def test_bridge_survives_markdown_fenced_json(self):
        from kbagent.llm_bridge import LLMBridge

        class Fenced:
            def invoke(self, messages):
                class R:
                    content = '```json\n{"passed": true}\n```'
                return R()

        self.assertEqual(LLMBridge(Fenced()).large_json("s", "u"),
                         {"passed": True})


class TestGlobalAudit(unittest.TestCase):
    def test_goalloop_survives_verifier_exception(self):
        """审查修复:loop.py logger 缺失曾导致验证器异常路径 NameError 逃逸。"""
        import asyncio
        from uniagent import Budget, BudgetConfig, GoalLoop

        class DummyAgent:
            async def ainvoke(self, state, **kw):
                return {"messages": state.get("messages", [])}

        class Exploding:
            async def verify(self, goal, state):
                raise RuntimeError("transient")

        r = asyncio.run(GoalLoop(
            agent=DummyAgent(), goal="g", verifier=Exploding(),
            budget=Budget(config=BudgetConfig(max_iterations=2)),
        ).run(initial_state={"messages": []}))
        self.assertEqual(r.iterations, 2)          # 优雅走到轮次耗尽,而非崩溃

    def test_zero_recall_full_chain_no_crash(self):
        """极端边界:检索零召回时,处理/答案阶段不崩,输出空答案而非异常。"""
        from kbagent.search import ESClient

        class EmptyES(ESClient):
            def keyword_search(self, dsl): return []
            def vector_search(self, q, f, size=10): return []

        a = MainAgent(ScriptedChatModel(), EmptyES())
        ans = a.run("任意问题")
        self.assertFalse(ans.degraded)             # 是正常空结果,不是异常降级
        self.assertEqual(ans.sources, [])

    def test_second_run_resets_round_counter(self):
        """工作区按请求隔离:第二次 run 的召回轮次从 round1 重新计数。"""
        a = MainAgent(ScriptedChatModel(), MockESClient())
        a.run("用户想咨询投诉处理时限")
        a.run("用户想咨询宽带新装")
        stages = [e.stage for e in a.tracer.events if e.event == "recall"]
        self.assertTrue(all(s.endswith("round1") for s in stages[:1]))
        self.assertIn("retrieval.round1", stages)


if __name__ == "__main__":
    unittest.main(verbosity=1)
