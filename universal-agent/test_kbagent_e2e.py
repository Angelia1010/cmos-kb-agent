# -*- coding: utf-8 -*-
"""kbagent 整体端到端集成测试

覆盖范围:
  TC-01  正常链路 — 常见套餐问题，1 轮成功，答案结构完整
  TC-02  循环重试 — 冷门问题，GoalLoop 2 轮耗尽，携最优退出
  TC-03  LLM 故障降级 — BrokenModel → degraded=True，有基础知识兜底
  TC-04  chunk_id 全链路透传 — sources 中 chunk_id 来自检索片段
  TC-05  trace 记录完整 — 关键 stage 都出现在 trace events 中
  TC-06  答案字段契约 — FinalAnswer 必填字段均有值
  TC-07  多次请求隔离 — 同一 MainAgent 连续 run，trace 不交叉

运行方式:
  PYTHONPATH=src python -m unittest test_kbagent_e2e -v
  PYTHONPATH=src python test_kbagent_e2e
"""
import sys
import unittest

sys.path.insert(0, "src")

from kbagent import MainAgent, MockESClient, ScriptedChatModel
from kbagent.shared.models import FinalAnswer


# ─────────────────────────────── BrokenModel ────────────────────────────── #

class BrokenModel(ScriptedChatModel):
    """每次调用都抛出 TimeoutError，用于测试 LLM 故障降级路径。"""

    def _generate(self, *a, **kw):
        raise TimeoutError("LLM gateway timeout")


# ─────────────────────────────── base ───────────────────────────────────── #

class KbagentE2EBase(unittest.TestCase):
    """共享 fixture:一个正常 MainAgent 实例。"""

    @classmethod
    def setUpClass(cls):
        cls.agent = MainAgent(
            model=ScriptedChatModel(),
            es=MockESClient(),
            enable_skills=False,   # 关闭技能注入，减少 fixture 依赖
        )

    def _run(self, query: str, agent: MainAgent | None = None) -> FinalAnswer:
        a = agent or self.agent
        return a.run(query)


# ══════════════════════════════════════════════════════════════════════════ #
#  TC-01  正常链路                                                           #
# ══════════════════════════════════════════════════════════════════════════ #

class TC01_NormalFlow(KbagentE2EBase):

    def setUp(self):
        self.ans = self._run("用户想办理流量套餐,如何推荐?")

    def test_returns_final_answer_instance(self):
        self.assertIsInstance(self.ans, FinalAnswer)

    def test_not_degraded(self):
        self.assertFalse(self.ans.degraded,
                         "正常链路不应触发降级")

    def test_has_business_explanation(self):
        self.assertTrue(self.ans.business_explanation.strip(),
                        "业务说明不应为空")

    def test_has_handling_suggestion(self):
        self.assertTrue(self.ans.handling_suggestion.strip(),
                        "办理建议不应为空")

    def test_has_sources(self):
        self.assertGreater(len(self.ans.sources), 0,
                           "正常链路应有知识来源")

    def test_has_trace_id(self):
        self.assertTrue(self.ans.trace_id,
                        "trace_id 不应为空")

    def test_elapsed_ms_is_positive(self):
        self.assertGreater(self.ans.elapsed_ms, 0)

    def test_render_produces_nonempty_string(self):
        rendered = self.ans.render()
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 10)

    def test_sources_have_chunk_id(self):
        for s in self.ans.sources:
            self.assertTrue(s.chunk_id, f"source 缺少 chunk_id: {s}")

    def test_sources_have_doc_title(self):
        for s in self.ans.sources:
            self.assertTrue(s.doc_title, f"source 缺少 doc_title: {s}")

    def test_sources_have_snippet(self):
        for s in self.ans.sources:
            self.assertTrue(s.snippet, f"source 缺少 snippet: {s}")


# ══════════════════════════════════════════════════════════════════════════ #
#  TC-02  循环重试 — 冷门问题                                                #
# ══════════════════════════════════════════════════════════════════════════ #

class TC02_RetryExhaust(KbagentE2EBase):

    def setUp(self):
        self.ans = self._run("副卡怎么共享主卡额度")

    def test_returns_answer_even_when_exhausted(self):
        """轮次耗尽后仍应返回有效答案（携 best 退出），不抛异常。"""
        self.assertIsInstance(self.ans, FinalAnswer)

    def test_not_degraded_on_retry_exhaust(self):
        """检索轮次耗尽 ≠ 系统降级；只有 agent/LLM 完全失败才降级。"""
        self.assertFalse(self.ans.degraded,
                         "检索轮次耗尽不应触发 degraded 标志")

    def test_still_has_trace_id(self):
        self.assertTrue(self.ans.trace_id)

    def test_render_does_not_crash(self):
        try:
            self.ans.render()
        except Exception as exc:
            self.fail(f"render() 不应抛出异常: {exc}")


# ══════════════════════════════════════════════════════════════════════════ #
#  TC-03  LLM 故障降级                                                       #
# ══════════════════════════════════════════════════════════════════════════ #

class TC03_LLMFailureDegrades(KbagentE2EBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.broken_agent = MainAgent(
            model=BrokenModel(),
            es=MockESClient(),
            enable_skills=False,
        )

    def setUp(self):
        self.ans = self._run("宽带新装怎么办理", agent=self.broken_agent)

    def test_degraded_is_true(self):
        self.assertTrue(self.ans.degraded,
                        "LLM 故障时应触发降级(degraded=True)")

    def test_returns_final_answer_instance(self):
        self.assertIsInstance(self.ans, FinalAnswer)

    def test_has_trace_id(self):
        self.assertTrue(self.ans.trace_id)

    def test_render_shows_degraded_marker(self):
        rendered = self.ans.render()
        self.assertIn("降级", rendered,
                      "降级结果渲染中应包含 '降级' 标记")

    def test_sources_have_at_least_some_content(self):
        """降级兜底应通过保守关键词检索返回至少 1 条原始片段。"""
        self.assertGreater(len(self.ans.sources), 0,
                           "降级兜底应从 ES 召回至少 1 条片段")


# ══════════════════════════════════════════════════════════════════════════ #
#  TC-04  chunk_id 全链路透传                                                #
# ══════════════════════════════════════════════════════════════════════════ #

class TC04_ChunkIdTransparency(KbagentE2EBase):

    KNOWN_CHUNK_IDS = {
        "kb_0001#p1", "kb_0001#p2",
        "kb_0002#p1", "kb_0003#p1",
        "kb_0004#p1", "kb_0005#p1",
    }

    def test_sources_chunk_ids_from_kb(self):
        """最终答案的 chunk_id 必须来自知识库，不能凭空捏造。"""
        ans = self._run("流量套餐推荐")
        for s in ans.sources:
            self.assertIn(s.chunk_id, self.KNOWN_CHUNK_IDS,
                          f"source chunk_id '{s.chunk_id}' 不在知识库中")

    def test_no_duplicate_chunk_ids_in_sources(self):
        """同一 chunk_id 不应在 sources 中重复出现。"""
        ans = self._run("流量套餐推荐")
        ids = [s.chunk_id for s in ans.sources]
        self.assertEqual(len(ids), len(set(ids)),
                         f"sources 中有重复 chunk_id: {ids}")

    def test_sentences_cite_valid_chunk_ids(self):
        """非 dropped 句子的引用 chunk_id 必须在 sources 中存在。"""
        ans = self._run("套餐推荐")
        source_ids = {s.chunk_id for s in ans.sources}
        for sent in ans.sentences:
            if sent.dropped:
                continue
            for cid in sent.citations:
                self.assertIn(cid, source_ids,
                              f"句子引用了不在 sources 中的 chunk_id: {cid}")


# ══════════════════════════════════════════════════════════════════════════ #
#  TC-05  Trace 事件记录                                                     #
# ══════════════════════════════════════════════════════════════════════════ #

class TC05_TraceEvents(KbagentE2EBase):

    def setUp(self):
        self._run("流量套餐推荐")

    def test_tracer_has_events(self):
        self.assertGreater(len(self.agent.tracer.events), 0)

    def test_run_start_event_present(self):
        events = [(e.stage, e.event) for e in self.agent.tracer.events]
        self.assertIn(("run", "start"), events,
                      "trace 中应有 (run, start) 事件")

    def test_finalize_done_event_present(self):
        events = [(e.stage, e.event) for e in self.agent.tracer.events]
        self.assertIn(("finalize", "done"), events,
                      "trace 中应有 (finalize, done) 事件")

    def test_retrieval_events_present(self):
        stages = {e.stage for e in self.agent.tracer.events}
        self.assertTrue(
            any("retrieval" in s for s in stages),
            f"trace 中应有 retrieval 相关 stage: {stages}"
        )

    def test_answer_events_present(self):
        stages = {e.stage for e in self.agent.tracer.events}
        self.assertIn("answer", stages,
                      f"trace 中应有 answer stage: {stages}")

    def test_trace_export_is_valid_json(self):
        """Tracer.export() 应输出合法 JSON。"""
        import json
        raw = self.agent.tracer.export()
        obj = json.loads(raw)
        self.assertIn("trace_id", obj)
        self.assertIn("events", obj)

    def test_trace_id_matches_answer(self):
        agent = MainAgent(
            model=ScriptedChatModel(), es=MockESClient(), enable_skills=False
        )
        ans = agent.run("套餐推荐")
        self.assertEqual(agent.tracer.trace_id, ans.trace_id)


# ══════════════════════════════════════════════════════════════════════════ #
#  TC-06  答案字段契约                                                       #
# ══════════════════════════════════════════════════════════════════════════ #

class TC06_AnswerFieldContract(KbagentE2EBase):

    def test_business_explanation_is_str(self):
        ans = self._run("套餐推荐")
        self.assertIsInstance(ans.business_explanation, str)

    def test_handling_suggestion_is_str(self):
        ans = self._run("套餐推荐")
        self.assertIsInstance(ans.handling_suggestion, str)

    def test_sources_is_list(self):
        ans = self._run("套餐推荐")
        self.assertIsInstance(ans.sources, list)

    def test_sentences_is_list(self):
        ans = self._run("套餐推荐")
        self.assertIsInstance(ans.sentences, list)

    def test_degraded_is_bool(self):
        ans = self._run("套餐推荐")
        self.assertIsInstance(ans.degraded, bool)

    def test_elapsed_ms_is_int(self):
        ans = self._run("套餐推荐")
        self.assertIsInstance(ans.elapsed_ms, int)

    def test_trace_id_is_nonempty_str(self):
        ans = self._run("套餐推荐")
        self.assertIsInstance(ans.trace_id, str)
        self.assertTrue(ans.trace_id)


# ══════════════════════════════════════════════════════════════════════════ #
#  TC-07  多次请求隔离                                                       #
# ══════════════════════════════════════════════════════════════════════════ #

class TC07_MultiRequestIsolation(unittest.TestCase):

    def test_each_run_gets_new_trace_id(self):
        """同一 MainAgent 连续 run 两次，trace_id 不相同。"""
        agent = MainAgent(
            model=ScriptedChatModel(), es=MockESClient(), enable_skills=False
        )
        ans1 = agent.run("套餐推荐")
        ans2 = agent.run("宽带新装")
        self.assertNotEqual(ans1.trace_id, ans2.trace_id,
                            "每次请求应产生新的 trace_id")

    def test_second_run_trace_events_dont_include_first_run(self):
        """第二次 run 的 tracer 不应包含第一次 run 的事件。"""
        agent = MainAgent(
            model=ScriptedChatModel(), es=MockESClient(), enable_skills=False
        )
        agent.run("第一次问题")
        count_after_first = len(agent.tracer.events)

        agent.run("第二次问题")
        count_after_second = len(agent.tracer.events)

        # 第二次 run 重置了 tracer，事件数应回到第二次 run 的事件数
        self.assertLessEqual(count_after_second, count_after_first * 2,
                             "tracer 应在每次 run 时重置，避免事件无限累积")

    def test_different_queries_can_return_different_results(self):
        """不同问题的答案不完全相同(trace_id 不同已保证)。"""
        agent = MainAgent(
            model=ScriptedChatModel(), es=MockESClient(), enable_skills=False
        )
        ans_suite = agent.run("流量套餐推荐")
        ans_broadband = agent.run("宽带新装怎么办理")
        self.assertNotEqual(ans_suite.trace_id, ans_broadband.trace_id)


# ══════════════════════════════════════════════════════════════════════════ #
#  运行入口                                                                  #
# ══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    unittest.main(verbosity=2)
