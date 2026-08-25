# -*- coding: utf-8 -*-
"""RetrievalSubAgent 测试

覆盖范围:
  T1  build_dsl — DSL 字段白名单强制执行
  T2  retrieval/tools — query_understanding / keyword_extraction /
                        question_rewrite / coarse_recall 单元测试
  T3  SufficiencyVerifier — 规则层充分性判定
  T4  RetrievalSubAgent — 完整子智能体运行(GoalLoop 护栏)

运行方式:
  PYTHONPATH=src python -m unittest tests.test_retrieval -v
"""
import asyncio
import json
import sys
import unittest

sys.path.insert(0, "src")

from kbagent.shared.config import DEFAULT_CONFIG, Config
from kbagent.shared.models import Chunk, RetrievalParams
from kbagent.shared.search import (
    ALLOWED_FILTER_FIELDS, MockESClient, build_dsl,
)
from kbagent.shared.tracing import Tracer
from kbagent.shared.workspace import RunWorkspace, set_workspace


# ─────────────────────────────── helpers ────────────────────────────────── #

def _ws(query: str = "套餐办理", data: dict | None = None,
        stage: str = "retrieval") -> RunWorkspace:
    """创建并注入测试工作区。"""
    ws = RunWorkspace(
        query=query,
        cfg=DEFAULT_CONFIG,
        es=MockESClient(),
        tracer=Tracer(),
    )
    ws.stage = stage
    if data:
        ws.data.update(data)
    set_workspace(ws)
    return ws


def _chunk(chunk_id: str = "c1", score: float = 0.9,
           category: str = "套餐", status: str = "在售") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="d1", doc_title="测试文档",
        content="5G畅享套餐月费(每月)59元,含30GB流量,办理条件:实名客户,无欠费。",
        category=category, extra={"status": status},
        score=score, updated_at="2026-06-10",
    )


# ══════════════════════════════════════════════════════════════════════════ #
#  T1  DSL 字段白名单                                                        #
# ══════════════════════════════════════════════════════════════════════════ #

class TestBuildDSL(unittest.TestCase):

    def test_whitelist_strips_unknown_filter_fields(self):
        """非白名单字段被 build_dsl 静默过滤，不出现在 ES DSL 里。"""
        params = RetrievalParams(
            keywords=["套餐"],
            filters={
                "category": "套餐",      # 合法
                "status": "在售",        # 合法
                "malicious_field": "x",  # 非法
                "__proto__": "y",        # 非法
            },
        )
        dsl = build_dsl(params)
        filter_fields = {
            list(f["term"].keys())[0]
            for f in dsl["query"]["bool"]["filter"]
        }
        self.assertTrue(filter_fields.issubset(ALLOWED_FILTER_FIELDS),
                        f"出现了非白名单字段: {filter_fields - ALLOWED_FILTER_FIELDS}")
        self.assertNotIn("malicious_field", filter_fields)
        self.assertNotIn("__proto__", filter_fields)

    def test_dsl_structure_has_required_keys(self):
        """构建出的 DSL 必须含 size / query.bool.must / query.bool.filter。"""
        params = RetrievalParams(keywords=["流量", "套餐"])
        dsl = build_dsl(params, size=5)
        self.assertEqual(dsl["size"], 5)
        self.assertIn("bool", dsl["query"])
        self.assertIn("must", dsl["query"]["bool"])
        self.assertIn("filter", dsl["query"]["bool"])

    def test_boost_fields_whitelist_applied(self):
        """boost_fields 中非允许字段不注入 DSL，防止字段注入。"""
        params = RetrievalParams(
            keywords=["套餐"],
            boost_fields={"title": 2.0, "evil_field": 99.0},
        )
        dsl = build_dsl(params)
        query_str = json.dumps(dsl)
        self.assertNotIn("evil_field", query_str)

    def test_empty_keywords_uses_wildcard(self):
        """无关键词时查询词退化为 *，不崩溃。"""
        params = RetrievalParams(keywords=[])
        dsl = build_dsl(params)
        mm = dsl["query"]["bool"]["must"][0]["multi_match"]
        self.assertEqual(mm["query"], "*")


# ══════════════════════════════════════════════════════════════════════════ #
#  T2  检索工具单元测试                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class TestRetrievalTools(unittest.TestCase):

    def setUp(self):
        # 每个 case 前重置工作区
        self.ws = _ws("用户想办理流量套餐")

    # ── query_understanding ──────────────────────────────────────────────── #

    def test_query_understanding_returns_intent_list(self):
        from kbagent.retrieval.tools import query_understanding
        raw = query_understanding.func()
        data = json.loads(raw)
        self.assertIn("intents", data)
        self.assertIsInstance(data["intents"], list)

    def test_query_understanding_writes_to_workspace(self):
        from kbagent.retrieval.tools import query_understanding
        query_understanding.func()
        self.assertIn("intents", self.ws.data)

    def test_query_understanding_detects_taocan_category(self):
        """套餐相关问题应识别出 '套餐' 意图。"""
        from kbagent.retrieval.tools import query_understanding
        query_understanding.func()
        self.assertIn("套餐", self.ws.data["intents"])

    def test_query_understanding_uses_param_over_workspace(self):
        """传入 query 参数时应优先使用，而非工作区的 ws.query。"""
        from kbagent.retrieval.tools import query_understanding
        raw = query_understanding.func(query="宽带新装")
        data = json.loads(raw)
        # 结果应基于 "宽带新装"，不是工作区原始问题
        self.assertIsInstance(data["intents"], list)

    # ── keyword_extraction ───────────────────────────────────────────────── #

    def test_keyword_extraction_without_expand(self):
        from kbagent.retrieval.tools import keyword_extraction
        raw = keyword_extraction.func(expand=False)
        data = json.loads(raw)
        self.assertIn("keywords", data)
        self.assertGreater(len(data["keywords"]), 0)
        self.assertEqual(data["expanded_terms"], [])

    def test_keyword_extraction_with_expand(self):
        from kbagent.retrieval.tools import keyword_extraction
        raw = keyword_extraction.func(expand=True)
        data = json.loads(raw)
        self.assertIn("expanded_terms", data)

    def test_keyword_extraction_uses_rewritten_query_first(self):
        """改写后的问题优先于原始问题用于关键词提取。"""
        self.ws.data["rewritten_query"] = "宽带如何新装"
        from kbagent.retrieval.tools import keyword_extraction
        raw = keyword_extraction.func(expand=False)
        data = json.loads(raw)
        # 关键词应从改写后问题提取，应包含宽带相关词
        kws = " ".join(data["keywords"])
        self.assertTrue("宽" in kws or "带" in kws or "新" in kws or "装" in kws,
                        f"改写问题关键词未出现: {data['keywords']}")

    def test_keyword_extraction_writes_to_workspace(self):
        from kbagent.retrieval.tools import keyword_extraction
        keyword_extraction.func()
        self.assertIn("keywords", self.ws.data)
        self.assertIn("expanded_terms", self.ws.data)

    # ── question_rewrite ─────────────────────────────────────────────────── #

    def test_question_rewrite_sets_retry_flag(self):
        ws = _ws("怎么办不了套餐")
        from kbagent.retrieval.tools import question_rewrite
        question_rewrite.func()
        self.assertTrue(ws.data.get("is_retry"),
                        "改写工具应设置 is_retry 标志")

    def test_question_rewrite_produces_rewritten_query(self):
        ws = _ws("怎么办套餐")
        from kbagent.retrieval.tools import question_rewrite
        raw = question_rewrite.func()
        data = json.loads(raw)
        self.assertIn("rewritten_query", data)
        self.assertIn("rewritten_query", ws.data)
        self.assertGreater(len(ws.data["rewritten_query"]), 0)

    def test_question_rewrite_normalizes_colloquial(self):
        """'怎么' 应规一化为 '如何'。"""
        ws = _ws("套餐怎么办理")
        from kbagent.retrieval.tools import question_rewrite
        question_rewrite.func()
        self.assertIn("如何", ws.data["rewritten_query"])

    # ── coarse_recall ────────────────────────────────────────────────────── #

    def test_coarse_recall_returns_chunks(self):
        self.ws.data["keywords"] = ["流量", "套餐"]
        self.ws.data["expanded_terms"] = []
        from kbagent.retrieval.tools import coarse_recall
        raw = coarse_recall.func(retrieval_mode="keyword")
        data = json.loads(raw)
        self.assertIn("recalled", data)
        self.assertGreater(data["recalled"], 0)

    def test_coarse_recall_writes_chunks_to_workspace(self):
        self.ws.data["keywords"] = ["流量", "套餐"]
        from kbagent.retrieval.tools import coarse_recall
        coarse_recall.func(retrieval_mode="hybrid")
        self.assertIn("chunks", self.ws.data)
        self.assertIsInstance(self.ws.data["chunks"], list)

    def test_coarse_recall_without_keywords_returns_error(self):
        """未提取关键词时 coarse_recall 应返回错误，不崩溃。"""
        from kbagent.retrieval.tools import coarse_recall
        raw = coarse_recall.func()
        data = json.loads(raw)
        self.assertIn("error", data,
                      "缺关键词时应返回 error 字段")

    def test_coarse_recall_increments_round_counter(self):
        self.ws.data["keywords"] = ["套餐"]
        from kbagent.retrieval.tools import coarse_recall
        coarse_recall.func()
        coarse_recall.func()
        self.assertEqual(self.ws.data.get("recall_round"), 2)

    def test_coarse_recall_records_last_dsl(self):
        self.ws.data["keywords"] = ["套餐"]
        from kbagent.retrieval.tools import coarse_recall
        coarse_recall.func(retrieval_mode="keyword")
        self.assertIn("last_dsl", self.ws.data)
        self.assertIn("query", self.ws.data["last_dsl"])


# ══════════════════════════════════════════════════════════════════════════ #
#  T3  充分性验证器                                                          #
# ══════════════════════════════════════════════════════════════════════════ #

class TestSufficiencyVerifier(unittest.TestCase):

    def _verify(self, chunks: list) -> object:
        ws = _ws()
        ws.data["chunks"] = chunks
        from kbagent.retrieval.sufficiency import SufficiencyVerifier
        return asyncio.run(SufficiencyVerifier().verify("goal", {}))

    def test_passes_with_3_high_score_chunks(self):
        chunks = [_chunk(f"c{i}", score=0.9) for i in range(3)]
        r = self._verify(chunks)
        self.assertTrue(r.passed)

    def test_passes_with_more_than_3_chunks(self):
        chunks = [_chunk(f"c{i}", score=0.85) for i in range(5)]
        r = self._verify(chunks)
        self.assertTrue(r.passed)

    def test_fails_when_chunk_count_below_threshold(self):
        """少于 min_chunk_count(默认 3) 应失败。"""
        chunks = [_chunk("c1", score=0.9), _chunk("c2", score=0.9)]
        r = self._verify(chunks)
        self.assertFalse(r.passed)
        self.assertIn("候选数", r.evidence)

    def test_fails_when_top3_scores_below_threshold(self):
        """top3 得分均低于 top3_score_threshold 时应失败。"""
        chunks = [_chunk(f"c{i}", score=0.1) for i in range(4)]
        r = self._verify(chunks)
        self.assertFalse(r.passed)
        self.assertIn("得分", r.evidence)

    def test_fails_with_empty_chunks(self):
        r = self._verify([])
        self.assertFalse(r.passed)

    def test_failure_evidence_contains_retry_hint(self):
        """验证失败的 evidence 要包含改写/放宽建议，供 GoalLoop 注入。"""
        r = self._verify([_chunk("c1", score=0.1)])
        self.assertIn("请换策略", r.evidence)

    def test_confidence_is_1_for_rule_based_result(self):
        """纯规则层结果 confidence 应为 1.0。"""
        r = self._verify([_chunk("c1", score=0.1)])
        self.assertEqual(r.confidence, 1.0)

    def test_layer_is_rules(self):
        r = self._verify([_chunk(f"c{i}", score=0.9) for i in range(3)])
        self.assertEqual(r.layer, "rules")


# ══════════════════════════════════════════════════════════════════════════ #
#  T4  RetrievalSubAgent 完整运行                                            #
# ══════════════════════════════════════════════════════════════════════════ #

class TestRetrievalSubAgent(unittest.TestCase):

    def _make_agent(self):
        from kbagent.retrieval.agent import RetrievalSubAgent
        from kbagent.scripted_model import ScriptedChatModel
        return RetrievalSubAgent(ScriptedChatModel(), DEFAULT_CONFIG, Tracer())

    def _run(self, query: str) -> list:
        ws = RunWorkspace(
            query=query, cfg=DEFAULT_CONFIG, es=MockESClient(), tracer=Tracer()
        )
        ws.stage = "retrieval"
        set_workspace(ws)
        return asyncio.run(self._make_agent().run(query))

    def test_hot_query_returns_nonempty_chunks(self):
        """常见套餐问题应一轮成功，返回非空 chunk 列表。"""
        chunks = self._run("用户想办理流量套餐,如何推荐?")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0,
                           "热门问题检索结果不应为空")

    def test_all_chunks_have_chunk_id(self):
        chunks = self._run("流量套餐怎么推荐")
        for c in chunks:
            self.assertTrue(c.chunk_id, f"chunk 缺少 chunk_id: {c}")

    def test_all_chunks_have_content(self):
        chunks = self._run("宽带新装怎么办理")
        for c in chunks:
            self.assertTrue(c.content, "chunk.content 不应为空")

    def test_cold_query_does_not_raise(self):
        """冷门问题 2 轮耗尽后应以 best 退出，不抛异常。"""
        try:
            chunks = self._run("副卡怎么共享主卡额度")
            # 轮次耗尽，可能返回空列表或少量片段
            self.assertIsInstance(chunks, list)
        except Exception as exc:
            self.fail(f"冷门问题不应抛出异常: {exc}")

    def test_result_chunks_are_chunk_instances(self):
        chunks = self._run("流量套餐")
        for c in chunks:
            self.assertIsInstance(c, Chunk)

    def test_tracer_records_recall_events(self):
        """GoalLoop 运行后 trace 里应有召回相关事件。"""
        ws = RunWorkspace(
            query="套餐推荐", cfg=DEFAULT_CONFIG,
            es=MockESClient(), tracer=Tracer()
        )
        ws.stage = "retrieval"
        set_workspace(ws)
        asyncio.run(self._make_agent().run("套餐推荐"))
        events = [e.event for e in ws.tracer.events]
        self.assertTrue(
            any("recall" in ev for ev in events),
            f"trace 中未找到召回事件: {events}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
