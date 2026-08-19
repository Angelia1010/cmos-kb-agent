# -*- coding: utf-8 -*-
"""关键约束测试:PYTHONPATH=src python tests.py"""
import sys
sys.path.insert(0, "src")

import unittest

from kbagent import MainAgent, MockESClient, ScriptedChatModel
from kbagent.models import RetrievalParams
from kbagent.search import build_dsl


class TestDSLWhitelist(unittest.TestCase):
    def test_filter_whitelist_blocks_injection(self):
        p = RetrievalParams(keywords=["套餐"],
                            filters={"category": "套餐",
                                     "script": "ctx._source.x=1"})
        dsl = build_dsl(p)
        fields = [list(f["term"].keys())[0] for f in dsl["query"]["bool"]["filter"]]
        self.assertEqual(fields, ["category"])


class TestGoalLoopCap(unittest.TestCase):
    def test_budget_caps_retrieval_rounds(self):
        """冷门问题两轮都不充分 → Budget(max_iterations=2) 硬退出并携最优候选。"""
        p = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        ans = p.run("完全不存在的冷门问题xyz")
        recalls = [e for e in p.tracer.events if e.event == "recall"]
        self.assertLessEqual(len(recalls), 2)
        exits = [e for e in p.tracer.events if e.event == "exit_with_best"]
        self.assertEqual(len(exits), 1)
        self.assertFalse(ans.degraded)          # 携最优候选走完全链路,而非降级

    def test_feedback_drives_second_round(self):
        """第2轮必须发生真实的改写+重新召回(验证失败反馈生效)。"""
        p = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        p.run("副卡怎么共享主卡额度")
        recalls = [e for e in p.tracer.events if e.event == "recall"]
        self.assertEqual(len(recalls), 2)
        q1 = recalls[0].payload["dsl"]["query"]["bool"]["must"][0]["multi_match"]["query"]
        q2 = recalls[1].payload["dsl"]["query"]["bool"]["must"][0]["multi_match"]["query"]
        self.assertNotEqual(q1, q2)             # 改写后的检索词不同


class TestBusinessSkill(unittest.TestCase):
    def test_skill_package_triggers_normalization(self):
        """套餐问题触发 taocan-skill → apply_business_skill 完成字段归一。"""
        p = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        ans = p.run("用户想办理流量套餐,如何推荐?")
        self.assertIn("月费(每月)", ans.business_explanation)

    def test_skills_disabled_no_normalization(self):
        p = MainAgent(model=ScriptedChatModel(), es=MockESClient(),
                       enable_skills=False)
        ans = p.run("用户想咨询流量套餐资费")
        # 无技能提示时,scripted 模型不套用业务skill(SKILL: 标记缺失)
        self.assertNotIn("skill_applied",
                         str([s.snippet for s in ans.sources])) 


class TestAnchoringAndDegrade(unittest.TestCase):
    def test_hard_fact_without_valid_citation_dropped(self):
        from kbagent.answer import generate
        from kbagent.config import DEFAULT_CONFIG
        from kbagent.models import Chunk
        from kbagent.tracing import Tracer
        from kbagent.workspace import RunWorkspace, set_workspace

        class FakeModel(ScriptedChatModel):
            def large_json(self, system, user):
                return {"business_explanation": "月费99元。",
                        "handling_suggestion": "",
                        "sentences": [{"text": "月费99元。",
                                       "citations": ["kb_fake#p9"],
                                       "hard_fact": True}]}

        set_workspace(RunWorkspace())
        mat = [Chunk(chunk_id="kb_a#p1", doc_id="kb_a", doc_title="T",
                     content="月费59元。", category="套餐",
                     updated_at="2026-06-01")]
        ans = generate(FakeModel(), "q", mat, DEFAULT_CONFIG, Tracer(), "t")
        self.assertEqual(len(ans.sentences), 0)
        self.assertNotIn("99元", ans.business_explanation)

    def test_model_failure_degrades_with_sources(self):
        class Broken(ScriptedChatModel):
            def _generate(self, *a, **kw):
                raise TimeoutError("gateway timeout")

        p = MainAgent(model=Broken(), es=MockESClient())
        ans = p.run("宽带新装怎么办理")
        self.assertTrue(ans.degraded)
        self.assertGreater(len(ans.sources), 0)


class TestCache(unittest.TestCase):
    def test_cache_hit_and_invalidation(self):
        p = MainAgent(model=ScriptedChatModel(), es=MockESClient())
        p.run("用户想办理流量套餐,如何推荐?")
        hit = p.run("请问用户想办理流量套餐,如何推荐")
        self.assertTrue(hit.from_cache)
        n = p.cache.invalidate_doc("kb_0001")
        self.assertGreater(n, 0)
        miss = p.run("请问用户想办理流量套餐,如何推荐")
        self.assertFalse(miss.from_cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
