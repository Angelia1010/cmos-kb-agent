# -*- coding: utf-8 -*-
"""RetrievalSubAgent 测试(合并 kbagent-retrieval-dev 后的新检索栈)

覆盖范围:
  T1  search 转换层 — kresult_to_chunks / vresult_to_chunks / get_kid_score
                      + MockESClient 新接口契约
  T2  retrieval/tools — intergrate_all / keyword_recall / vector_recall /
                        query_rewrite 单元测试
  T3  SufficiencyVerifier — 固定三轮计数判定
  T4  RetrievalSubAgent — GoalLoop(检索→处理→ProcessingVerifier 验证)完整运行
      + DirectRetrievalSubAgent 零 LLM 直调

运行方式:
  PYTHONPATH=src python -m unittest tests.test_retrieval -v
"""
import asyncio
import json
import sys
import unittest

sys.path.insert(0, "src")

from kbagent.scripted_model import ScriptedChatModel
from kbagent.shared.config import DEFAULT_CONFIG
from kbagent.shared.models import Chunk
from kbagent.shared.search import (
    MockESClient,
    kresult_to_chunks,
    vresult_to_chunks,
    get_kid_score,
)
from kbagent.shared.tracing import Tracer
from kbagent.shared.workspace import RunWorkspace, set_workspace


# ─────────────────────────────── helpers ────────────────────────────────── #

def _ws(query: str = "套餐办理", data: dict | None = None,
        stage: str = "retrieval", model=None) -> RunWorkspace:
    """创建并注入测试工作区。"""
    ws = RunWorkspace(
        query=query,
        cfg=DEFAULT_CONFIG,
        es=MockESClient(),
        tracer=Tracer(),
        model=model,
    )
    ws.stage = stage
    if data:
        ws.data.update(data)
    set_workspace(ws)
    return ws


def _ngkm_entry(kid: str = "K001", title: str = "5G畅享套餐资费说明",
                content: str = "月费59元,含30GB全国流量") -> dict:
    """ngkm 一体化流水线知识条目(info+atoms 结构)。"""
    return {
        "knowledgeId": kid,
        "knowledgeName": title,
        "category": "套餐",
        "status": "在售",
        "updateTime": "2026-06-10",
        "atoms": [
            {"paramName": "资费说明", "content": content},
            {"paramName": "空原子", "content": ""},          # 应被丢弃
            {"error": "原子表查询失败"},                        # 应被丢弃
        ],
    }


def _vector_entry(kid: str = "K001", title: str = "流量加油包",
                  content: str = "10元5GB加油包") -> dict:
    return {"knowledgeId": kid, "knowledgeName": title, "content": content}


# ══════════════════════════════════════════════════════════════════════════ #
#  T1  search 转换层 + MockESClient 契约                                     #
# ══════════════════════════════════════════════════════════════════════════ #

class TestKresultToChunks(unittest.TestCase):

    def test_entry_maps_to_chunk_with_atom_content(self):
        chunks = kresult_to_chunks([_ngkm_entry()])
        self.assertEqual(1, len(chunks))
        c = chunks[0]
        self.assertIsInstance(c, Chunk)
        self.assertEqual("K001", c.chunk_id)
        self.assertEqual("K001", c.doc_id)
        self.assertEqual("5G畅享套餐资费说明", c.doc_title)
        self.assertIn("资费说明:月费59元", c.content)
        self.assertNotIn("空原子", c.content)
        self.assertEqual("套餐", c.category)
        self.assertEqual("2026-06-10", c.updated_at)
        self.assertEqual("ngkm", c.extra["source"])
        self.assertEqual("在售", c.extra["status"])

    def test_score_decays_with_rank_and_floor(self):
        chunks = kresult_to_chunks(
            [_ngkm_entry(f"K{i:03d}") for i in range(20)])
        self.assertEqual(1.0, chunks[0].score)
        self.assertEqual(0.95, chunks[1].score)
        self.assertTrue(all(c.score >= 0.5 for c in chunks))

    def test_empty_and_invalid_entries(self):
        self.assertEqual([], kresult_to_chunks([]))
        self.assertEqual([], kresult_to_chunks(None))
        # 非 dict 条目与无内容条目被丢弃
        self.assertEqual([], kresult_to_chunks(
            ["not-a-dict", {"knowledgeId": "X", "atoms": []}]))

    def test_dict_input_yields_no_chunks(self):
        """整包 dict(非 merged 列表)传入时迭代出字符串 key,静默产出 0 条 —
        intergrate_all 曾因此丢失 keyword 通道,回归防护。"""
        self.assertEqual([], kresult_to_chunks(
            {"merged": [_ngkm_entry()], "keywords": ["套餐"]}))


class TestVresultToChunks(unittest.TestCase):

    def test_both_mode_all_list(self):
        chunks = vresult_to_chunks(
            {"new": None, "old": None,
             "all": [_vector_entry("K1"), _vector_entry("K2", "宽带", "300M")]})
        self.assertEqual(["K1", "K2"], [c.chunk_id for c in chunks])
        self.assertEqual("流量加油包", chunks[0].doc_title)
        self.assertEqual("vector", chunks[0].extra["source"])

    def test_dedupe_by_kid(self):
        chunks = vresult_to_chunks(
            {"all": [_vector_entry("K1"), _vector_entry("K1", "重复", "重复")]})
        self.assertEqual(1, len(chunks))

    def test_walk_nested_object_json_string(self):
        """new/old 单路原始响应:object 为 JSON 字符串包裹也能提取。"""
        raw = {"object": json.dumps(
            {"document": [_vector_entry("K9")]}, ensure_ascii=False)}
        chunks = vresult_to_chunks(raw)
        self.assertEqual(["K9"], [c.chunk_id for c in chunks])

    def test_none_and_empty(self):
        self.assertEqual([], vresult_to_chunks(None))
        self.assertEqual([], vresult_to_chunks({}))

    def test_updated_at_passthrough(self):
        """向量条目携带 updateTime 时应透传到 Chunk.updated_at(缺失则空)。"""
        entry = _vector_entry("K1")
        entry["updateTime"] = "2026-04-01"
        chunks = vresult_to_chunks({"all": [entry, _vector_entry("K2")]})
        self.assertEqual("2026-04-01", chunks[0].updated_at)
        self.assertEqual("", chunks[1].updated_at)


class TestGetKidScore(unittest.TestCase):

    def test_dual_path_scores_higher(self):
        scores = get_kid_score(["A", "B"], ["B", "C"])
        self.assertEqual(2.0, scores["B"])     # 双路命中
        self.assertEqual(1.0, scores["A"])     # 仅 keyword
        self.assertEqual(1.0, scores["C"])     # 仅 vector

    def test_empty_inputs(self):
        self.assertEqual({}, get_kid_score([], []))
        self.assertEqual({"A": 1.0}, get_kid_score(["A"], None))


class TestMockESClientContract(unittest.TestCase):

    def test_keyword_search_returns_pipeline_shape(self):
        result = MockESClient().keyword_search(query="流量套餐怎么推荐")
        self.assertIsInstance(result, dict)
        for key in ("merged", "keywords", "knowledge_ids"):
            self.assertIn(key, result)
        self.assertTrue(result["merged"])
        entry = result["merged"][0]
        self.assertIn("knowledgeId", entry)
        self.assertIn("atoms", entry)
        self.assertEqual([e["knowledgeId"] for e in result["merged"]],
                         result["knowledge_ids"])

    def test_keyword_search_accepts_explicit_keywords(self):
        result = MockESClient().keyword_search(query="随便什么", keywords=["宽带"])
        self.assertTrue(result["merged"])
        self.assertTrue(all("宽带" in json.dumps(e, ensure_ascii=False)
                            for e in result["merged"]))

    def test_keyword_search_zero_hit_message(self):
        result = MockESClient().keyword_search(query="量子隐形传态资费",
                                               keywords=["量子隐形传态"])
        self.assertEqual([], result["merged"])
        self.assertIn("message", result)

    def test_vector_search_returns_both_shape(self):
        result = MockESClient().vector_search("流量不够用怎么办")
        self.assertIsInstance(result, dict)
        self.assertIn("all", result)
        self.assertTrue(result["all"])
        chunks = vresult_to_chunks(result)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))

    def test_vector_search_empty_query(self):
        self.assertIsNone(MockESClient().vector_search("  "))


# ══════════════════════════════════════════════════════════════════════════ #
#  T2  检索工具单元测试                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class TestRetrievalTools(unittest.TestCase):

    def setUp(self):
        self.ws = _ws("用户想办理流量套餐", model=ScriptedChatModel())

    # ── intergrate_all ───────────────────────────────────────────────────── #

    def test_intergrate_all_dual_channel_recall(self):
        from kbagent.retrieval.tools import intergrate_all
        data = json.loads(intergrate_all.func(query="流量套餐推荐"))
        self.assertGreater(data.get("recalled", 0), 0)
        # keyword 通道必须产出 Chunk(回归:kresult_to_chunks 整包 dict bug)
        self.assertTrue(self.ws.data["keyword_chunks"])
        self.assertTrue(self.ws.data["vector_chunks"])
        self.assertTrue(self.ws.data["chunks"])
        self.assertTrue(self.ws.data["ranked_kids"])
        self.assertEqual(1, self.ws.data["recall_round"])

    def test_intergrate_all_kid_ranking_dual_hit_first(self):
        """双路命中的 kid 得分最高,merged_chunks 按其排序靠前。"""
        from kbagent.retrieval.tools import intergrate_all
        intergrate_all.func(query="流量套餐推荐")
        kid_scores = self.ws.data["kid_scores"]
        ranked = self.ws.data["ranked_kids"]
        self.assertTrue(ranked)
        self.assertEqual(max(kid_scores.values()), kid_scores[ranked[0]])

    def test_intergrate_all_explicit_keywords_skip_extraction(self):
        from kbagent.retrieval.tools import intergrate_all
        data = json.loads(intergrate_all.func(query="随便什么", keywords=["宽带"]))
        self.assertGreater(data.get("recalled", 0), 0)
        self.assertEqual(["宽带"], self.ws.data["keywords"])

    def test_intergrate_all_zero_recall_returns_error(self):
        """双路零召回 → error 观测,不抛异常。"""
        class _EmptyES:
            def keyword_search(self, query, region_code="", keywords=None,
                               timeout=30):
                return {"merged": [], "keywords": [], "knowledge_ids": [],
                        "message": "未提取到有效关键词"}

            def vector_search(self, query_text, region_code="",
                              vector_mode="both"):
                return None

        ws = _ws("量子隐形传态")
        ws.es = _EmptyES()
        from kbagent.retrieval.tools import intergrate_all
        data = json.loads(intergrate_all.func(query="量子隐形传态资费"))
        self.assertIn("error", data)
        self.assertEqual([], ws.data["chunks"])

    def test_intergrate_all_round_counter_increments(self):
        """不同参数推进轮次;同参数重复调用命中缓存不推进。"""
        from kbagent.retrieval.tools import intergrate_all
        intergrate_all.func(query="流量套餐")
        intergrate_all.func(query="流量套餐", keywords=["宽带"])
        self.assertEqual(2, self.ws.data["recall_round"])

    def test_intergrate_all_same_params_deduped(self):
        """同参数重复调用:返回相同观测、ES 不再被调、trace 记 recall_cached。"""
        from kbagent.retrieval.tools import intergrate_all
        calls = {"n": 0}
        orig = self.ws.es.keyword_search

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        self.ws.es.keyword_search = counting
        first = intergrate_all.func(query="流量套餐")
        second = intergrate_all.func(query="流量套餐")
        self.assertEqual(first, second)
        self.assertEqual(1, calls["n"], "同参数第二次调用不应再打 ES")
        self.assertEqual(1, self.ws.data["recall_round"], "缓存命中不推进轮次")
        events = [(e.stage, e.event) for e in self.ws.tracer.events]
        self.assertIn(("retrieval.round1", "recall_cached"), events)

    def test_intergrate_all_zero_recall_not_cached(self):
        """零召回不缓存:同参数再次调用仍真实请求 ES(瞬时故障可重试)。"""
        class _EmptyES:
            def __init__(self):
                self.calls = 0

            def keyword_search(self, query, region_code="", keywords=None,
                               timeout=30):
                self.calls += 1
                return {"merged": [], "keywords": [], "knowledge_ids": [],
                        "message": "未提取到有效关键词"}

            def vector_search(self, query_text, region_code="",
                              vector_mode="both"):
                return None

        ws = _ws("量子隐形传态")
        ws.es = _EmptyES()
        from kbagent.retrieval.tools import intergrate_all
        intergrate_all.func(query="量子隐形传态资费")
        intergrate_all.func(query="量子隐形传态资费")
        self.assertEqual(2, ws.es.calls, "零召回结果不应进缓存")

    # ── keyword_recall / vector_recall ──────────────────────────────────── #

    def test_keyword_recall_returns_chunks(self):
        from kbagent.retrieval.tools import keyword_recall
        data = json.loads(keyword_recall.func(query="宽带新装怎么办理"))
        self.assertGreater(data.get("recalled", 0), 0)
        titles = [c.doc_title for c in self.ws.data["chunks"]]
        self.assertIn("家庭宽带新装流程", titles)

    def test_vector_recall_returns_chunks(self):
        from kbagent.retrieval.tools import vector_recall
        data = json.loads(vector_recall.func(query="流量不够用怎么办"))
        self.assertGreater(data.get("recalled", 0), 0)
        self.assertTrue(all(c.extra["source"] == "vector"
                            for c in self.ws.data["chunks"]))

    def test_recall_tools_report_error_on_unsupported_backend(self):
        class _NoVectorES(MockESClient):
            vector_search = None

        ws = _ws("流量套餐")
        ws.es = _NoVectorES()
        from kbagent.retrieval.tools import vector_recall
        # getattr(ws.es, "vector_search") 为 None → error 观测
        data = json.loads(vector_recall.func(query="流量套餐"))
        self.assertIn("error", data)

    # ── query_rewrite ───────────────────────────────────────────────────── #

    def test_query_rewrite_produces_rewritten_keywords(self):
        self.ws.data["original_query"] = "用户想办理流量套餐"
        self.ws.data["keywords"] = ["流量", "套餐"]
        from kbagent.retrieval.tools import query_rewrite
        data = json.loads(query_rewrite.func())
        kws = data.get("rewritten_keywords")
        self.assertIsInstance(kws, list)
        self.assertTrue(kws)
        self.assertEqual(kws, self.ws.data["rewritten_keywords"])
        self.assertLessEqual(len(kws), 3)

    def test_query_rewrite_without_model_returns_error(self):
        ws = _ws("流量套餐")          # model=None
        from kbagent.retrieval.tools import query_rewrite
        data = json.loads(query_rewrite.func())
        self.assertIn("error", data)


# ══════════════════════════════════════════════════════════════════════════ #
#  T3  SufficiencyVerifier — 固定三轮计数                                    #
# ══════════════════════════════════════════════════════════════════════════ #

class TestSufficiencyVerifier(unittest.TestCase):

    def _verify_once(self, verifier) -> object:
        ws = _ws()
        return asyncio.run(verifier.verify("goal", {})), ws

    def test_fixed_three_round_pattern(self):
        from kbagent.retrieval.sufficiency import SufficiencyVerifier
        v = SufficiencyVerifier()
        r1, _ = self._verify_once(v)
        r2, _ = self._verify_once(v)
        r3, _ = self._verify_once(v)
        self.assertFalse(r1.passed)
        self.assertFalse(r2.passed)
        self.assertTrue(r3.passed)

    def test_result_metadata(self):
        from kbagent.retrieval.sufficiency import SufficiencyVerifier
        r, ws = self._verify_once(SufficiencyVerifier())
        self.assertEqual("fixed_rounds", r.layer)
        self.assertEqual(1.0, r.confidence)
        self.assertIn("sufficiency.fixed_round_fail",
                      [e.event for e in ws.tracer.events])

    def test_llm_judge_param_kept_for_compat(self):
        from kbagent.retrieval.sufficiency import SufficiencyVerifier
        v = SufficiencyVerifier(llm_judge=lambda system, user: "{}")
        r, _ = self._verify_once(v)
        self.assertFalse(r.passed)     # 计数模式不受 judge 影响


# ══════════════════════════════════════════════════════════════════════════ #
#  T4  RetrievalSubAgent(GoalLoop + ProcessingVerifier)/ Direct 直调       #
# ══════════════════════════════════════════════════════════════════════════ #

class TestRetrievalSubAgent(unittest.TestCase):

    def _run(self, query: str, agent_cls: str = "RetrievalSubAgent"):
        from kbagent.retrieval import agent as agent_mod
        ws = RunWorkspace(
            query=query, cfg=DEFAULT_CONFIG, es=MockESClient(),
            tracer=Tracer(), model=ScriptedChatModel(),
        )
        ws.stage = "retrieval"
        set_workspace(ws)
        agent = getattr(agent_mod, agent_cls)(
            ScriptedChatModel(), DEFAULT_CONFIG, ws.tracer)
        chunks = asyncio.run(agent.run(query))
        return ws, chunks

    def test_hot_query_returns_processed_chunks(self):
        """GoalLoop 走通:召回 → Processing → Top3 验证,返回非空结果。"""
        ws, chunks = self._run("用户想办理流量套餐,如何推荐?")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
        events = [(e.stage, e.event) for e in ws.tracer.events]
        self.assertIn(("retrieval", "loop_result"), events)

    def test_loop_produces_verification_trace(self):
        """每轮迭代后 ProcessingVerifier 自动运行,trace 留下验证事件。"""
        ws, _ = self._run("流量套餐怎么推荐")
        verify_events = [e.event for e in ws.tracer.events
                         if e.stage == "retrieval.verify"]
        self.assertTrue(verify_events, f"缺少验证事件: {ws.tracer.events}")

    def test_all_chunks_have_identity_and_content(self):
        _, chunks = self._run("宽带新装怎么办理")
        for c in chunks:
            self.assertTrue(c.chunk_id, f"chunk 缺少 chunk_id: {c}")
            self.assertTrue(c.content, "chunk.content 不应为空")

    def test_cold_query_does_not_raise(self):
        """冷门问题轮次耗尽后携最优退出,不抛异常。"""
        try:
            _, chunks = self._run("副卡怎么共享主卡额度")
            self.assertIsInstance(chunks, list)
        except Exception as exc:
            self.fail(f"冷门问题不应抛出异常: {exc}")

    def test_direct_subagent_zero_llm_recall(self):
        """DirectRetrievalSubAgent:直调 intergrate_all,零 LLM。"""
        ws, chunks = self._run("流量套餐推荐", agent_cls="DirectRetrievalSubAgent")
        self.assertTrue(chunks)
        self.assertIn(("retrieval", "done"),
                      [(e.stage, e.event) for e in ws.tracer.events])

    def test_direct_subagent_fallback_and_raise(self):
        """intergrate_all 报错且零召回 → keyword_recall 兜底,仍空则抛错。"""
        from kbagent.retrieval.agent import DirectRetrievalSubAgent

        class _DeadES(MockESClient):
            def keyword_search(self, query, region_code="", keywords=None,
                               timeout=30):
                return {"merged": [], "keywords": [], "knowledge_ids": [],
                        "error": "ngkm 不可用"}

            def vector_search(self, query_text, region_code="",
                              vector_mode="both"):
                return None

        ws = RunWorkspace(query="流量套餐", cfg=DEFAULT_CONFIG, es=_DeadES(),
                          tracer=Tracer())
        ws.stage = "retrieval"
        set_workspace(ws)
        agent = DirectRetrievalSubAgent(None, DEFAULT_CONFIG, ws.tracer)
        with self.assertRaises(RuntimeError):
            asyncio.run(agent.run("流量套餐"))
        self.assertIn("intergrate_all_fallback",
                      [e.event for e in ws.tracer.events])


if __name__ == "__main__":
    unittest.main(verbosity=2)
