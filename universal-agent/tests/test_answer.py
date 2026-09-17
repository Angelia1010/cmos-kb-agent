# -*- coding: utf-8 -*-
"""AnswerSubAgent 测试(精简契约版:话术 + 相关度文档 + 关键片段 + 原文)

覆盖范围:
  T1  select_fragments — 片段精选(top-N, 同文档限额)
  T2  _parse_json — JSON 解析鲁棒性
  T3  generate — 话术组织 + 来源组装(相关度/关键片段/全文)+ 批量一致性校验
  T4  AnswerSubAgent — 完整子智能体运行(locate → 精选 → generate)
  T4b 可用性判定(LLM 自评 + 规则只收紧)
  T5  FinalAnswer.render — 渲染格式
  T6  locate_fragments — 文档内证据片段定位(逐字、可溯源、相关度)

运行方式:
  PYTHONPATH=src python -m unittest tests.test_answer -v
"""
import json
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from kbagent.shared.config import DEFAULT_CONFIG
from kbagent.shared.models import (
    Chunk, DocFragments, FinalAnswer, LocatedFragment, SourceRef,
)
from kbagent.shared.tracing import Tracer


# ─────────────────────────────── helpers ────────────────────────────────── #

def _chunk(chunk_id: str = "c1", doc_id: str = "d1",
           content: str = "5G畅享套餐月费59元。",
           score: float = 0.9, updated_at: str = "2026-06-10") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, doc_title=f"文档_{doc_id}",
        content=content, category="套餐",
        extra={"status": "在售"},
        score=score, updated_at=updated_at,
    )


def _answer(query: str, model: object, chunks: list,
            matched: list = None) -> FinalAnswer:
    from kbagent.answer.generate import generate
    return generate(model, query, chunks, DEFAULT_CONFIG, Tracer(),
                    "trace_test", matched=matched)


def _doc_frags(chunk: Chunk, relevance: int,
               frag_text: str = "") -> DocFragments:
    """为 chunk 构造 DocFragments;frag_text 必须是 content 的逐字子串。"""
    frags = []
    if frag_text and frag_text in chunk.content:
        start = chunk.content.index(frag_text)
        frags.append(LocatedFragment(text=frag_text, start=start,
                                     end=start + len(frag_text), reason="测试"))
    return DocFragments(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                        doc_title=chunk.doc_title,
                        answerable=bool(frags), relevance=relevance,
                        fragments=frags)


# ══════════════════════════════════════════════════════════════════════════ #
#  T1  select_fragments                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class TestSelectFragments(unittest.TestCase):

    def test_returns_top_n_chunks(self):
        from kbagent.answer.generate import select_fragments
        chunks = [_chunk(f"c{i}", doc_id=f"d{i}") for i in range(8)]
        selected = select_fragments("q", chunks, top_n=4)
        self.assertEqual(len(selected), 4)

    def test_same_doc_limit_2(self):
        """同一 doc_id 最多取 2 个片段。"""
        from kbagent.answer.generate import select_fragments
        chunks = [_chunk(f"c{i}", doc_id="d1") for i in range(5)]  # all same doc
        selected = select_fragments("q", chunks, top_n=4)
        same_doc = [c for c in selected if c.doc_id == "d1"]
        self.assertLessEqual(len(same_doc), 2)

    def test_respects_order(self):
        """选出的片段应按输入顺序(已排序)排列。"""
        from kbagent.answer.generate import select_fragments
        chunks = [
            _chunk("c1", doc_id="d1", score=0.9),
            _chunk("c2", doc_id="d2", score=0.8),
        ]
        selected = select_fragments("q", chunks)
        self.assertEqual(selected[0].chunk_id, "c1")

    def test_empty_input_returns_empty(self):
        from kbagent.answer.generate import select_fragments
        self.assertEqual(select_fragments("q", []), [])

    def test_fewer_chunks_than_top_n_returns_all(self):
        from kbagent.answer.generate import select_fragments
        chunks = [_chunk("c1", doc_id="d1"), _chunk("c2", doc_id="d2")]
        selected = select_fragments("q", chunks, top_n=10)
        self.assertEqual(len(selected), 2)

    def test_multi_doc_all_included(self):
        """不同 doc_id 的片段可各自最多 2 个。"""
        from kbagent.answer.generate import select_fragments
        chunks = (
            [_chunk(f"d1c{i}", doc_id="d1") for i in range(3)] +
            [_chunk(f"d2c{i}", doc_id="d2") for i in range(3)]
        )
        selected = select_fragments("q", chunks, top_n=6)
        doc_counts = {}
        for c in selected:
            doc_counts[c.doc_id] = doc_counts.get(c.doc_id, 0) + 1
        for doc_id, cnt in doc_counts.items():
            self.assertLessEqual(cnt, 2, f"doc {doc_id} 超出 2 个片段上限")


# ══════════════════════════════════════════════════════════════════════════ #
#  T2  _parse_json                                                           #
# ══════════════════════════════════════════════════════════════════════════ #

class TestParseJson(unittest.TestCase):

    def _parse(self, raw: str) -> dict:
        from kbagent.answer.generate import _parse_json
        return _parse_json(raw)

    def test_valid_json(self):
        d = self._parse('{"a": 1, "b": "x"}')
        self.assertEqual(d["a"], 1)

    def test_markdown_fenced_json(self):
        raw = "```json\n{\"key\": \"val\"}\n```"
        d = self._parse(raw)
        self.assertEqual(d["key"], "val")

    def test_markdown_fence_without_lang(self):
        raw = "```\n{\"k\": 2}\n```"
        d = self._parse(raw)
        self.assertEqual(d["k"], 2)

    def test_invalid_json_returns_empty_dict(self):
        d = self._parse("这不是 JSON")
        self.assertEqual(d, {})

    def test_list_json_returns_empty_dict(self):
        """根节点为列表时应返回空 dict，不崩溃。"""
        d = self._parse("[1, 2, 3]")
        self.assertEqual(d, {})

    def test_empty_string_returns_empty_dict(self):
        d = self._parse("")
        self.assertEqual(d, {})

    def test_nested_json(self):
        raw = '{"usability": {"level": "verify_first", "reasons": ["r1"]}}'
        d = self._parse(raw)
        self.assertEqual(d["usability"]["reasons"][0], "r1")


# ══════════════════════════════════════════════════════════════════════════ #
#  T3  generate — 话术组织 + 来源组装 + 批量一致性校验                          #
# ══════════════════════════════════════════════════════════════════════════ #

def _scripted_model():
    from kbagent.scripted_model import ScriptedChatModel
    return ScriptedChatModel()


class TestGenerate(unittest.TestCase):

    def _chunks(self) -> list:
        return [
            _chunk("kb_0001#p1", doc_id="kb_0001",
                   content="5G畅享套餐月费59元,含30GB流量。"),
            _chunk("kb_0001#p2", doc_id="kb_0001",
                   content="办理条件:实名客户,无欠费。"),
            _chunk("kb_0002#p1", doc_id="kb_0002",
                   content="10元5GB加油包,当月有效,立即生效。"),
        ]

    def test_returns_final_answer(self):
        ans = _answer("套餐推荐", _scripted_model(), self._chunks())
        self.assertIsInstance(ans, FinalAnswer)

    def test_answer_has_trace_id(self):
        ans = _answer("套餐推荐", _scripted_model(), self._chunks())
        self.assertTrue(ans.trace_id)

    def test_answer_has_query(self):
        ans = _answer("套餐推荐", _scripted_model(), self._chunks())
        self.assertEqual(ans.query, "套餐推荐")

    def test_script_populated(self):
        """离线脚本模型应产出非空话术。"""
        ans = _answer("套餐推荐", _scripted_model(), self._chunks())
        self.assertTrue(ans.script)

    def test_sources_correspond_to_chunks(self):
        """sources 的 chunk_id 应是检索片段中存在的。"""
        chunks = self._chunks()
        valid_ids = {c.chunk_id for c in chunks}
        ans = _answer("套餐推荐", _scripted_model(), chunks)
        for s in ans.sources:
            self.assertIn(s.chunk_id, valid_ids)

    def test_sources_include_all_materials(self):
        """sources = 全部传入素材,content 为素材整篇原文。"""
        chunks = self._chunks()
        ans = _answer("套餐推荐", _scripted_model(), chunks)
        self.assertEqual({s.chunk_id for s in ans.sources},
                         {c.chunk_id for c in chunks})
        by_id = {c.chunk_id: c.content for c in chunks}
        for s in ans.sources:
            self.assertEqual(s.content, by_id[s.chunk_id])

    def test_sources_survive_json_parse_failure(self):
        """LLM 输出无法解析时也不丢溯源:sources 仍为全部素材。"""
        class _BadJsonModel:
            def invoke(self, messages):
                return MagicMock(content="不是JSON")

        chunks = self._chunks()
        ans = _answer("套餐推荐", _BadJsonModel(), chunks)
        self.assertEqual({s.chunk_id for s in ans.sources},
                         {c.chunk_id for c in chunks})

    def test_relevance_normalized_max_100_and_descending(self):
        """相关度归一化:最相关一篇=100,sources 按相关度降序。"""
        chunks = self._chunks()
        matched = [
            _doc_frags(chunks[0], relevance=50, frag_text="月费59元"),
            _doc_frags(chunks[1], relevance=20),
            _doc_frags(chunks[2], relevance=100, frag_text="10元5GB加油包"),
        ]
        ans = _answer("套餐推荐", _scripted_model(), chunks, matched=matched)
        rels = [s.relevance for s in ans.sources]
        self.assertEqual(rels[0], 100, "最相关一篇应归一化到 100")
        self.assertEqual(rels, sorted(rels, reverse=True), "sources 按相关度降序")
        self.assertEqual(ans.sources[0].chunk_id, "kb_0002#p1")

    def test_key_fragment_is_verbatim_substring(self):
        """keyFragment 来自 locate 的逐字片段,必为 content 子串。"""
        chunks = self._chunks()
        matched = [_doc_frags(chunks[0], relevance=90, frag_text="月费59元")]
        ans = _answer("套餐推荐", _scripted_model(), chunks[:1], matched=matched)
        kf = ans.sources[0].key_fragment
        self.assertEqual(kf, "月费59元")
        self.assertIn(kf, ans.sources[0].content)

    def test_relevance_fallback_when_no_matched(self):
        """matched 缺失时相关度按素材顺序兜底,仍保证最相关=100。"""
        chunks = self._chunks()
        ans = _answer("套餐推荐", _scripted_model(), chunks, matched=None)
        rels = [s.relevance for s in ans.sources]
        self.assertEqual(rels[0], 100)
        self.assertEqual(rels, sorted(rels, reverse=True))

    def test_consistency_fail_tightens_usability(self):
        """批量一致性校验不过 → usability 收紧为 verify_first,issues 入 reasons。"""
        class _InconsistentModel:
            def invoke(self, messages):
                msg = "\n".join(str(getattr(m, "content", "")) for m in messages)
                if "[TASK:anchor_check]" in msg:
                    return MagicMock(content=json.dumps(
                        {"consistent": False, "issues": ["资费数字无原文依据"]},
                        ensure_ascii=False))
                return MagicMock(content=json.dumps(
                    {"script": "您好,月费59元。", "handling_suggestion": "可办理",
                     "usability": {"level": "directly_usable",
                                   "reasons": [], "uncovered": []}},
                    ensure_ascii=False))

        ans = _answer("套餐推荐", _InconsistentModel(), self._chunks())
        self.assertEqual(ans.usability.level, "verify_first")
        self.assertTrue(any("核实" in r or "一致性" in r
                            for r in ans.usability.reasons))
        self.assertTrue(any("资费数字无原文依据" in r
                            for r in ans.usability.reasons))

    def test_generate_with_empty_chunks(self):
        """空片段列表不应崩溃，应返回有效的(但内容空的) FinalAnswer。"""
        ans = _answer("套餐推荐", _scripted_model(), [])
        self.assertIsInstance(ans, FinalAnswer)
        self.assertFalse(ans.degraded)

    def test_stale_detection_for_old_knowledge(self):
        """更新时间超过 stale_days(默认 365) 的片段应被标记为 stale。"""
        old_chunk = _chunk("old_c1", updated_at="2020-01-01")
        ans = _answer("套餐推荐", _scripted_model(), [old_chunk])
        for s in ans.sources:
            if s.chunk_id == "old_c1":
                self.assertTrue(s.stale, "2020年的知识应被标记为 stale")


# ══════════════════════════════════════════════════════════════════════════ #
#  T4  AnswerSubAgent 完整运行                                               #
# ══════════════════════════════════════════════════════════════════════════ #

class TestAnswerSubAgent(unittest.TestCase):

    def _agent(self):
        from kbagent.answer.agent import AnswerSubAgent
        return AnswerSubAgent(_scripted_model(), DEFAULT_CONFIG, Tracer())

    def _chunks(self) -> list:
        return [
            _chunk("kb_0001#p1", doc_id="kb_0001",
                   content="5G畅享套餐月费59元,含30GB流量。"),
            _chunk("kb_0001#p2", doc_id="kb_0001",
                   content="办理条件:实名客户,无欠费。"),
            _chunk("kb_0002#p1", doc_id="kb_0002",
                   content="10元5GB加油包,当月有效,立即生效。"),
        ]

    def test_run_returns_final_answer(self):
        ans = self._agent().run("套餐推荐", self._chunks(), "trace_001")
        self.assertIsInstance(ans, FinalAnswer)

    def test_run_sets_trace_id(self):
        ans = self._agent().run("套餐推荐", self._chunks(), "trace_xyz")
        self.assertEqual(ans.trace_id, "trace_xyz")

    def test_run_preserves_query(self):
        ans = self._agent().run("用户问套餐价格", self._chunks(), "t1")
        self.assertEqual(ans.query, "用户问套餐价格")

    def test_run_not_degraded(self):
        ans = self._agent().run("套餐推荐", self._chunks(), "t1")
        self.assertFalse(ans.degraded)

    def test_run_has_script_or_sources(self):
        ans = self._agent().run("套餐推荐", self._chunks(), "t1")
        self.assertTrue(ans.script or ans.sources,
                        "答案应至少包含话术或引用文档之一")

    def test_agent_selects_max_4_fragments(self):
        """AnswerSubAgent 使用 select_fragments(top_n=4)，sources 不超过 4 条。"""
        chunks = [_chunk(f"c{i}", doc_id=f"d{i}") for i in range(8)]
        ans = self._agent().run("套餐推荐", chunks, "t1")
        self.assertLessEqual(len(ans.sources), 4)

    def test_agent_populates_keyfragment_and_relevance(self):
        """完整运行:locate 产出的关键片段/相关度落到 sources,且逐字可溯源。"""
        chunks = self._chunks()
        ans = self._agent().run("5G套餐月费多少", chunks, "t_loc")
        by_id = {c.chunk_id: c.content for c in chunks}
        rels = [s.relevance for s in ans.sources]
        self.assertEqual(rels[0], 100, "最相关一篇应=100")
        self.assertEqual(rels, sorted(rels, reverse=True))
        for s in ans.sources:
            if s.key_fragment:
                self.assertIn(s.key_fragment, by_id[s.chunk_id],
                              "keyFragment 必为原文逐字子串")


# ══════════════════════════════════════════════════════════════════════════ #
#  T4b  可用性判定                                                            #
# ══════════════════════════════════════════════════════════════════════════ #

def _full_answer_model(payload: dict, consistent: bool = True,
                       issues: list = None):
    """构造返回指定 answer JSON 的 mock 模型;批量一致性校验按参数固定返回。"""
    issues = issues or []

    class _Model:
        def invoke(self, messages):
            msg = "\n".join(str(getattr(m, "content", "")) for m in messages)
            if "[TASK:anchor_check]" in msg:
                return MagicMock(content=json.dumps(
                    {"consistent": consistent, "issues": issues},
                    ensure_ascii=False))
            return MagicMock(content=json.dumps(payload, ensure_ascii=False))
    return _Model()


_BASE_PAYLOAD = {
    "script": "您好,异地是可以办理补换卡的。",
    "handling_suggestion": "引导用户使用APP办理。",
}


class TestStructuredContentAndUsability(unittest.TestCase):

    def _chunks(self) -> list:
        return [_chunk("kb_0001#p1", doc_id="kb_0001",
                       content="异地补换卡支持线上办理。")]

    def test_new_fields_populated(self):
        """话术/办理建议/可用性应完整落到 FinalAnswer。"""
        payload = dict(_BASE_PAYLOAD,
                       usability={"level": "directly_usable",
                                  "reasons": ["片段完整覆盖问题"],
                                  "uncovered": []})
        ans = _answer("异地补换卡", _full_answer_model(payload), self._chunks())
        self.assertEqual(ans.script, "您好,异地是可以办理补换卡的。")
        self.assertEqual(ans.handling_suggestion, "引导用户使用APP办理。")
        self.assertEqual(ans.usability.level, "directly_usable")
        self.assertEqual(ans.usability.reasons, ["片段完整覆盖问题"])

    def test_invalid_level_falls_back_to_verify(self):
        """LLM 自评 level 非法时,回退 verify_first 并记录原因。"""
        payload = dict(_BASE_PAYLOAD,
                       usability={"level": "maybe_ok", "reasons": [], "uncovered": []})
        ans = _answer("q", _full_answer_model(payload), self._chunks())
        self.assertEqual(ans.usability.level, "verify_first")
        self.assertTrue(any("未输出可用性自评" in r for r in ans.usability.reasons))

    def test_unparseable_output_is_not_usable(self):
        """LLM 输出无法解析(空 data)→ not_usable。"""
        class _BadModel:
            def invoke(self, messages):
                return MagicMock(content="抱歉,我无法回答")
        ans = _answer("q", _BadModel(), self._chunks())
        self.assertEqual(ans.usability.level, "not_usable")
        self.assertTrue(any("解析" in r or "为空" in r for r in ans.usability.reasons))

    def test_no_materials_is_not_usable(self):
        """素材为空 → not_usable(即使模型硬编了话术)。"""
        payload = dict(_BASE_PAYLOAD,
                       usability={"level": "directly_usable", "reasons": [], "uncovered": []})
        ans = _answer("q", _full_answer_model(payload), [])
        self.assertEqual(ans.usability.level, "not_usable")

    def test_rules_only_tighten_never_relax(self):
        """LLM 自评 directly_usable 但引用知识过旧 → 规则收紧为 verify_first。"""
        payload = dict(_BASE_PAYLOAD,
                       usability={"level": "directly_usable", "reasons": [], "uncovered": []})
        old_chunk = _chunk("kb_0001#p1", content="异地补换卡支持线上办理。",
                           updated_at="2020-01-01")
        ans = _answer("q", _full_answer_model(payload), [old_chunk])
        stale_sources = [s for s in ans.sources if s.stale]
        if stale_sources:
            self.assertEqual(ans.usability.level, "verify_first")
            self.assertTrue(any("过旧" in r for r in ans.usability.reasons))

    def test_consistency_fail_tightens_to_verify(self):
        """批量一致性校验不过 → 至少 verify_first。"""
        payload = dict(_BASE_PAYLOAD,
                       usability={"level": "directly_usable", "reasons": [], "uncovered": []})
        ans = _answer("q", _full_answer_model(payload, consistent=False,
                                              issues=["月费数字无依据"]),
                      self._chunks())
        self.assertIn(ans.usability.level, ("verify_first", "not_usable"))

    def test_uncovered_promoted_to_reasons(self):
        """LLM 报告 uncovered 时,level 至少 verify_first。"""
        payload = dict(_BASE_PAYLOAD,
                       usability={"level": "directly_usable", "reasons": [],
                                  "uncovered": ["跨省流量资费"]})
        ans = _answer("q", _full_answer_model(payload), self._chunks())
        self.assertEqual(ans.usability.uncovered, ["跨省流量资费"])
        self.assertEqual(ans.usability.level, "verify_first")

    def test_degrade_path_marks_not_usable(self):
        """主智能体降级路径 → usability=not_usable。"""
        from kbagent.main_agent import MainAgent
        from kbagent.shared.search import MockESClient
        agent = MainAgent(_scripted_model(), MockESClient(), enable_skills=False)
        ans = agent._degrade("随便问点啥", "test-trigger")
        self.assertTrue(ans.degraded)
        self.assertEqual(ans.usability.level, "not_usable")
        self.assertTrue(ans.usability.reasons)
        # 降级 sources 也应带 content 与相关度
        for s in ans.sources:
            self.assertTrue(s.content)


# ══════════════════════════════════════════════════════════════════════════ #
#  T5  FinalAnswer.render                                                    #
# ══════════════════════════════════════════════════════════════════════════ #

class TestFinalAnswerRender(unittest.TestCase):

    def _normal_answer(self, **kw) -> FinalAnswer:
        defaults = dict(
            trace_id="t1", query="q",
            script="您好,月费59元。",
            handling_suggestion="可为客户办理。",
            sources=[], degraded=False, elapsed_ms=100,
        )
        defaults.update(kw)
        return FinalAnswer(**defaults)

    def test_render_contains_script(self):
        ans = self._normal_answer(script="您好,月费59元,含30GB。")
        self.assertIn("月费59元", ans.render())

    def test_render_contains_handling_suggestion(self):
        ans = self._normal_answer(handling_suggestion="可为客户办理此业务。")
        self.assertIn("可为客户办理此业务。", ans.render())

    def test_render_degraded_shows_marker(self):
        ans = self._normal_answer(degraded=True)
        self.assertIn("降级", ans.render())

    def test_render_sources_section(self):
        ans = self._normal_answer(sources=[
            SourceRef(chunk_id="c1", doc_id="d1", doc_title="5G套餐说明",
                      relevance=100, key_fragment="月费59元",
                      content="月费59元含30GB", updated_at="2026-06-10")
        ])
        rendered = ans.render()
        self.assertIn("引用文档", rendered)
        self.assertIn("5G套餐说明", rendered)
        self.assertIn("c1", rendered)
        self.assertIn("100%", rendered)
        self.assertIn("月费59元", rendered)

    def test_render_stale_source_shows_warning(self):
        ans = self._normal_answer(sources=[
            SourceRef(chunk_id="c1", doc_id="d1", doc_title="旧文档",
                      relevance=80, content="内容", updated_at="2020-01-01",
                      stale=True)
        ])
        self.assertIn("过旧", ans.render())

    def test_render_no_sources_no_sources_section(self):
        ans = self._normal_answer(sources=[])
        self.assertNotIn("引用文档", ans.render())

    def test_render_sections_order(self):
        """渲染结果应按 话术 → 办理建议 → 引用文档 排列。"""
        ans = self._normal_answer(sources=[
            SourceRef(chunk_id="c1", doc_id="d1", doc_title="文档",
                      relevance=100, content="内容", updated_at="2026-01-01")
        ])
        rendered = ans.render()
        self.assertLess(rendered.index("坐席话术"), rendered.index("办理建议"))
        self.assertLess(rendered.index("办理建议"), rendered.index("引用文档"))


# ══════════════════════════════════════════════════════════════════════════ #
#  T6  locate_fragments — 文档内证据片段定位 + 相关度                          #
# ══════════════════════════════════════════════════════════════════════════ #

def _locate_model(answerable: bool, fragments: list, relevance: int = 0):
    """构造对 [TASK:locate_fragments] 返回固定 JSON 的 mock 模型。"""
    payload = {"answerable": answerable, "relevance": relevance,
               "fragments": fragments}

    class _Model:
        def invoke(self, messages):
            return MagicMock(content=json.dumps(payload, ensure_ascii=False))
    return _Model()


class TestLocateFragments(unittest.TestCase):

    def test_verbatim_fragment_located_with_offsets(self):
        """逐字片段被正确定位,start/end 偏移正确且 content[start:end]==text。"""
        from kbagent.answer.locate import locate_fragments
        content = "5G畅享套餐59元档:每月包含国内流量20GB、国内通话300分钟。超出后按5元/GB计费。"
        chunk = _chunk("c1", doc_id="d1", content=content)
        frag = "每月包含国内流量20GB、国内通话300分钟"
        df = locate_fragments(
            _locate_model(True, [{"text": frag, "reason": "回答了流量与通话"}], 90),
            "59元档含多少流量和通话", chunk)
        self.assertTrue(df.answerable)
        self.assertEqual(len(df.fragments), 1)
        f = df.fragments[0]
        self.assertEqual(f.text, frag)
        self.assertEqual(content[f.start:f.end], frag)
        self.assertEqual(f.reason, "回答了流量与通话")
        self.assertEqual((df.chunk_id, df.doc_id), ("c1", "d1"))
        self.assertEqual(df.relevance, 90)

    def test_hallucinated_fragment_dropped(self):
        """模型返回非原文片段 → 被丢弃,无可验证片段时 answerable 归 False。"""
        from kbagent.answer.locate import locate_fragments
        chunk = _chunk("c1", content="5G畅享套餐59元档每月包含国内流量20GB。")
        df = locate_fragments(
            _locate_model(True, [{"text": "该套餐赠送视频会员", "reason": "x"}], 80),
            "送不送视频会员", chunk)
        self.assertEqual(df.fragments, [])
        self.assertFalse(df.answerable)

    def test_not_answerable_returns_empty(self):
        from kbagent.answer.locate import locate_fragments
        chunk = _chunk("c1", content="宽带安装需预约。")
        df = locate_fragments(_locate_model(False, [], 0), "59元套餐多少钱", chunk)
        self.assertFalse(df.answerable)
        self.assertEqual(df.fragments, [])

    def test_whitespace_flexible_match(self):
        """片段与原文仅空白/换行不同 → 弹性兜底命中,回填的是原文(含换行)。"""
        from kbagent.answer.locate import locate_fragments
        content = "套餐内容:\n每月流量 20GB\n通话 300 分钟"
        chunk = _chunk("c1", content=content)
        df = locate_fragments(
            _locate_model(True, [{"text": "每月流量 20GB 通话 300 分钟", "reason": ""}], 70),
            "套餐含什么", chunk)
        self.assertEqual(len(df.fragments), 1)
        f = df.fragments[0]
        self.assertEqual(content[f.start:f.end], f.text)
        self.assertIn("每月流量", f.text)

    def test_max_fragments_capped(self):
        """单篇文档片段数受 MAX_FRAGMENTS_PER_DOC 上限约束。"""
        from kbagent.answer.locate import locate_fragments, MAX_FRAGMENTS_PER_DOC
        content = "。".join(f"第{i}条内容" for i in range(20))
        chunk = _chunk("c1", content=content)
        frags = [{"text": f"第{i}条内容", "reason": ""} for i in range(20)]
        df = locate_fragments(_locate_model(True, frags, 60), "内容", chunk)
        self.assertLessEqual(len(df.fragments), MAX_FRAGMENTS_PER_DOC)

    def test_relevance_clamped_to_100(self):
        """相关度超过 100 应被夹紧。"""
        from kbagent.answer.locate import locate_fragments
        chunk = _chunk("c1", content="5G套餐59元。")
        df = locate_fragments(
            _locate_model(True, [{"text": "59元", "reason": ""}], 999), "多少钱", chunk)
        self.assertEqual(df.relevance, 100)

    def test_relevance_capped_when_not_answerable(self):
        """无可验证片段(不可答)时相关度压到 39 以下。"""
        from kbagent.answer.locate import locate_fragments
        chunk = _chunk("c1", content="宽带安装需预约。")
        df = locate_fragments(_locate_model(False, [], 95), "59元套餐", chunk)
        self.assertLessEqual(df.relevance, 39)

    def test_agent_locates_all_chunks_into_sources(self):
        """AnswerSubAgent.run 对全部输入 chunks 定位,逐字片段落进 sources.key_fragment。"""
        from kbagent.answer.agent import AnswerSubAgent
        chunks = [
            _chunk("c1", doc_id="d1", content="5G畅享套餐月费59元,含30GB流量。"),
            _chunk("c2", doc_id="d2", content="宽带安装需预约,免费上门。"),
            _chunk("c3", doc_id="d3", content="10元5GB加油包,当月有效。"),
        ]
        agent = AnswerSubAgent(_scripted_model(), DEFAULT_CONFIG, Tracer())
        ans = agent.run("5G套餐月费多少", chunks, "t_loc")
        by_id = {c.chunk_id: c.content for c in chunks}
        for s in ans.sources:
            if s.key_fragment:
                self.assertIn(s.key_fragment, by_id[s.chunk_id])


if __name__ == "__main__":
    unittest.main(verbosity=2)
