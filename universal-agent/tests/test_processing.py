# -*- coding: utf-8 -*-
"""ProcessingSubAgent 测试

覆盖范围:
  T1  processing/tools — 7 个工具逐一单元测试
  T2  run_fallback_pipeline — 确定性保底流水线
  T3  ProcessingSubAgent — 完整子智能体运行

运行方式:
  PYTHONPATH=src python -m unittest tests.test_processing -v
"""
import asyncio
import json
import sys
import unittest

sys.path.insert(0, "src")

from kbagent.shared.config import DEFAULT_CONFIG
from kbagent.shared.models import Chunk
from kbagent.shared.search import MockESClient
from kbagent.shared.tracing import Tracer
from kbagent.shared.workspace import RunWorkspace, set_workspace


# ─────────────────────────────── helpers ────────────────────────────────── #

def _ws(chunks: list | None = None, query: str = "套餐推荐") -> RunWorkspace:
    ws = RunWorkspace(
        query=query, cfg=DEFAULT_CONFIG, es=MockESClient(), tracer=Tracer()
    )
    ws.stage = "processing"
    ws.data["chunks"] = list(chunks) if chunks else []
    set_workspace(ws)
    return ws


def _chunk(chunk_id: str = "c1", score: float = 0.8,
           content: str = "5G畅享套餐月费59元,含30GB流量。",
           category: str = "套餐", status: str = "在售",
           updated_at: str = "2026-06-10") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=f"d_{chunk_id}", doc_title=f"文档_{chunk_id}",
        content=content, category=category,
        extra={"status": status},
        score=score, updated_at=updated_at,
    )


def _chunks_suite() -> list:
    """标准测试片段集:混合在售/下架，包含金额/时限，分属不同 chunk_id。"""
    return [
        _chunk("c1", score=0.9, content="5G畅享套餐月费59元,含30GB流量。"),
        _chunk("c2", score=0.7, content="宽带新装需48小时内上门,月费30元。",
               category="宽带"),
        _chunk("c3", score=0.5, content="4G飞享套餐已停售。",
               status="下架"),
        _chunk("c4", score=0.9, content="客户可查询近6个月账单。",
               category="账单"),
        _chunk("c1", score=0.6, content="重复 c1 低分版。"),  # 用于测试去重
    ]


# ══════════════════════════════════════════════════════════════════════════ #
#  T1  数据处理工具单元测试                                                  #
# ══════════════════════════════════════════════════════════════════════════ #

class TestAnalyzeData(unittest.TestCase):

    def test_returns_count_and_categories(self):
        ws = _ws([_chunk("c1"), _chunk("c2", category="宽带")])
        from kbagent.processing.tools import analyze_data
        raw = analyze_data.func()
        data = json.loads(raw)
        self.assertEqual(data["count"], 2)
        self.assertIn("categories", data)

    def test_writes_analysis_to_workspace(self):
        ws = _ws([_chunk()])
        from kbagent.processing.tools import analyze_data
        analyze_data.func()
        self.assertIn("analysis", ws.data)

    def test_counts_statuses(self):
        ws = _ws([_chunk(status="在售"), _chunk("c2", status="下架")])
        from kbagent.processing.tools import analyze_data
        raw = analyze_data.func()
        data = json.loads(raw)
        self.assertIn("在售", data["statuses"])
        self.assertIn("下架", data["statuses"])

    def test_empty_chunks(self):
        _ws([])
        from kbagent.processing.tools import analyze_data
        raw = analyze_data.func()
        data = json.loads(raw)
        self.assertEqual(data["count"], 0)


class TestCleanData(unittest.TestCase):

    def test_strips_multiple_whitespace(self):
        c = _chunk(content="套餐    月费   59元\t含流量。")
        ws = _ws([c])
        from kbagent.processing.tools import clean_data
        clean_data.func()
        self.assertNotIn("    ", ws.data["chunks"][0].content)
        self.assertNotIn("\t", ws.data["chunks"][0].content)

    def test_strips_leading_trailing_whitespace(self):
        c = _chunk(content="  内容前后有空格  ")
        _ws([c])
        from kbagent.processing.tools import clean_data
        clean_data.func()
        chunk = _ws.__self__ if hasattr(_ws, "__self__") else None
        # 直接从工作区读
        ws = RunWorkspace.__new__(RunWorkspace)
        from kbagent.shared.workspace import get_workspace
        cleaned = get_workspace().data["chunks"][0].content
        self.assertEqual(cleaned, "内容前后有空格")

    def test_returns_count(self):
        _ws([_chunk("c1"), _chunk("c2")])
        from kbagent.processing.tools import clean_data
        raw = clean_data.func()
        data = json.loads(raw)
        self.assertEqual(data["cleaned"], 2)


class TestDenoiseData(unittest.TestCase):

    def test_removes_offshelf_chunks(self):
        ws = _ws([
            _chunk("c1", status="在售"),
            _chunk("c2", status="下架"),
            _chunk("c3", status="下架"),
        ])
        from kbagent.processing.tools import denoise_data
        denoise_data.func()
        remaining = [c.chunk_id for c in ws.data["chunks"]]
        self.assertIn("c1", remaining)
        self.assertNotIn("c2", remaining)
        self.assertNotIn("c3", remaining)

    def test_keeps_all_inservice_chunks(self):
        ws = _ws([_chunk(f"c{i}", status="在售") for i in range(3)])
        from kbagent.processing.tools import denoise_data
        denoise_data.func()
        self.assertEqual(len(ws.data["chunks"]), 3)

    def test_returns_before_after_counts(self):
        _ws([_chunk("c1"), _chunk("c2", status="下架")])
        from kbagent.processing.tools import denoise_data
        raw = denoise_data.func()
        data = json.loads(raw)
        self.assertEqual(data["before"], 2)
        self.assertEqual(data["after"], 1)


class TestDedupeData(unittest.TestCase):

    def test_keeps_highest_score_for_same_chunk_id(self):
        ws = _ws([
            _chunk("dup", score=0.5),
            _chunk("dup", score=0.9),
            _chunk("dup", score=0.3),
        ])
        from kbagent.processing.tools import dedupe_data
        dedupe_data.func()
        chunks = ws.data["chunks"]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].score, 0.9)

    def test_different_chunk_ids_all_kept(self):
        ws = _ws([_chunk("c1"), _chunk("c2"), _chunk("c3")])
        from kbagent.processing.tools import dedupe_data
        dedupe_data.func()
        ids = {c.chunk_id for c in ws.data["chunks"]}
        self.assertEqual(ids, {"c1", "c2", "c3"})

    def test_returns_before_after(self):
        _ws([_chunk("x", score=0.8), _chunk("x", score=0.5)])
        from kbagent.processing.tools import dedupe_data
        raw = dedupe_data.func()
        data = json.loads(raw)
        self.assertEqual(data["before"], 2)
        self.assertEqual(data["after"], 1)


class TestStructureData(unittest.TestCase):

    def test_extracts_fee_amounts(self):
        ws = _ws([_chunk(content="月费59元,次月1日生效。")])
        from kbagent.processing.tools import structure_data
        structure_data.func()
        self.assertIn("fees_yuan", ws.data["chunks"][0].extra)
        self.assertIn("59", ws.data["chunks"][0].extra["fees_yuan"])

    def test_extracts_deadline_hours(self):
        ws = _ws([_chunk(content="预约后48小时内上门安装。")])
        from kbagent.processing.tools import structure_data
        structure_data.func()
        self.assertIn("deadlines_hours", ws.data["chunks"][0].extra)
        self.assertIn("48", ws.data["chunks"][0].extra["deadlines_hours"])

    def test_no_extra_fields_removed(self):
        """structure_data 只增不删，原有 extra 字段应保留。"""
        c = _chunk(content="月费59元")
        c.extra["status"] = "在售"
        ws = _ws([c])
        from kbagent.processing.tools import structure_data
        structure_data.func()
        self.assertIn("status", ws.data["chunks"][0].extra)

    def test_content_without_fees_not_modified(self):
        ws = _ws([_chunk(content="请携带身份证办理。")])
        from kbagent.processing.tools import structure_data
        structure_data.func()
        # 没有金额，不应写入 fees_yuan
        self.assertNotIn("fees_yuan", ws.data["chunks"][0].extra)


class TestSortData(unittest.TestCase):

    def test_sorted_by_score_descending(self):
        ws = _ws([
            _chunk("c1", score=0.3),
            _chunk("c2", score=0.9),
            _chunk("c3", score=0.6),
        ])
        from kbagent.processing.tools import sort_data
        sort_data.func()
        scores = [c.score for c in ws.data["chunks"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_same_score_sorted_by_updated_at_desc(self):
        ws = _ws([
            _chunk("c1", score=0.8, updated_at="2025-01-01"),
            _chunk("c2", score=0.8, updated_at="2026-06-01"),
        ])
        from kbagent.processing.tools import sort_data
        sort_data.func()
        ids = [c.chunk_id for c in ws.data["chunks"]]
        self.assertEqual(ids[0], "c2", "相同得分时更新日期更新的应排前面")

    def test_returns_count(self):
        _ws([_chunk(), _chunk("c2")])
        from kbagent.processing.tools import sort_data
        raw = sort_data.func()
        data = json.loads(raw)
        self.assertEqual(data["sorted"], 2)


class TestApplyBusinessSkill(unittest.TestCase):

    def test_taocan_normalizes_content(self):
        """套餐技能应把 '月费' 归一为 '月费(每月)'。"""
        ws = _ws([_chunk(content="5G套餐月费59元。")])
        from kbagent.processing.tools import apply_business_skill
        apply_business_skill.func(category="套餐")
        content = ws.data["chunks"][0].content
        self.assertIn("月费(每月)", content)

    def test_marks_skill_applied_in_extra(self):
        ws = _ws([_chunk()])
        from kbagent.processing.tools import apply_business_skill
        apply_business_skill.func(category="套餐")
        self.assertEqual(ws.data["chunks"][0].extra.get("skill_applied"), "套餐")

    def test_invalid_category_returns_error(self):
        """非法 category 应返回 error，不抛异常，不修改 chunks。"""
        ws = _ws([_chunk()])
        from kbagent.processing.tools import apply_business_skill
        raw = apply_business_skill.func(category="未知类目")
        data = json.loads(raw)
        self.assertIn("error", data)
        # chunks 不应被修改
        self.assertNotIn("skill_applied", ws.data["chunks"][0].extra)

    def test_all_valid_categories_accepted(self):
        for cat in ("套餐", "宽带", "账单", "投诉"):
            ws = _ws([_chunk()])
            from kbagent.processing.tools import apply_business_skill
            raw = apply_business_skill.func(category=cat)
            data = json.loads(raw)
            self.assertNotIn("error", data,
                             f"合法类目 '{cat}' 不应返回 error")


# ══════════════════════════════════════════════════════════════════════════ #
#  T2  确定性保底流水线                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class TestFallbackPipeline(unittest.TestCase):

    def test_pipeline_removes_offshelf_and_dedupes(self):
        """保底流水线应去噪(删下架) + 去重 + 排序，不崩溃。"""
        ws = _ws([
            _chunk("c1", score=0.9, status="在售"),
            _chunk("c2", score=0.3, status="下架"),
            _chunk("c1", score=0.5, status="在售"),  # 重复
        ])
        from kbagent.processing.tools import run_fallback_pipeline
        run_fallback_pipeline()
        ids = [c.chunk_id for c in ws.data["chunks"]]
        self.assertIn("c1", ids)
        self.assertNotIn("c2", ids)
        self.assertEqual(ids.count("c1"), 1, "去重后 c1 应只出现一次")

    def test_pipeline_runs_on_empty_chunks(self):
        """空 chunk 列表不应导致异常。"""
        _ws([])
        from kbagent.processing.tools import run_fallback_pipeline
        try:
            run_fallback_pipeline()
        except Exception as exc:
            self.fail(f"空输入时保底流水线不应抛出异常: {exc}")

    def test_pipeline_preserves_content_without_truncation(self):
        """保底流水线不裁剪片段内容。"""
        original_content = "5G畅享套餐月费59元,含30GB流量。" * 3
        ws = _ws([_chunk(content=original_content)])
        from kbagent.processing.tools import run_fallback_pipeline
        run_fallback_pipeline()
        # 内容被 clean_data 压缩空白，但不应截断
        cleaned = ws.data["chunks"][0].content
        self.assertGreater(len(cleaned), 10)


# ══════════════════════════════════════════════════════════════════════════ #
#  T3  ProcessingSubAgent 完整运行                                           #
# ══════════════════════════════════════════════════════════════════════════ #

class TestProcessingSubAgent(unittest.TestCase):

    def _make_agent(self, enable_skills: bool = False):
        from kbagent.processing.agent import ProcessingSubAgent
        from kbagent.scripted_model import ScriptedChatModel
        return ProcessingSubAgent(ScriptedChatModel(), Tracer(),
                                  enable_skills=enable_skills)

    def _setup_ws(self) -> RunWorkspace:
        ws = RunWorkspace(
            query="套餐推荐", cfg=DEFAULT_CONFIG,
            es=MockESClient(), tracer=Tracer()
        )
        ws.stage = "processing"
        set_workspace(ws)
        return ws

    def _sample_chunks(self) -> list:
        return [
            _chunk("k1", score=0.9, content="5G畅享套餐月费59元,含30GB流量。"),
            _chunk("k2", score=0.7, content="加油包10元5GB,立即生效。"),
            _chunk("k3", score=0.4, content="旧套餐已停售。", status="下架"),
        ]

    def test_returns_list_of_chunks(self):
        self._setup_ws()
        agent = self._make_agent()
        chunks = asyncio.run(agent.run("套餐推荐", self._sample_chunks()))
        self.assertIsInstance(chunks, list)

    def test_offshelf_chunks_removed_by_processing(self):
        """处理后，下架片段应被去噪工具或保底流水线过滤。"""
        self._setup_ws()
        agent = self._make_agent()
        chunks = asyncio.run(agent.run("套餐推荐", self._sample_chunks()))
        statuses = {c.extra.get("status") for c in chunks}
        self.assertNotIn("下架", statuses,
                         "处理子智能体应过滤下架片段")

    def test_result_chunks_have_required_attrs(self):
        self._setup_ws()
        agent = self._make_agent()
        chunks = asyncio.run(agent.run("套餐推荐", self._sample_chunks()))
        for c in chunks:
            self.assertTrue(c.chunk_id, "chunk 必须有 chunk_id")
            self.assertTrue(c.content, "chunk 必须有 content")

    def test_tracer_logs_processing_snapshot(self):
        ws = self._setup_ws()
        agent = self._make_agent()
        agent.tracer = ws.tracer
        asyncio.run(agent.run("套餐推荐", self._sample_chunks()))
        events = [e.event for e in ws.tracer.events]
        self.assertIn("snapshot", events,
                      "处理子智能体应在 tracer 中记录 snapshot 事件")

    def test_empty_input_triggers_fallback(self):
        """子智能体没有产出时，保底流水线应接管，不崩溃。"""
        ws = self._setup_ws()
        agent = self._make_agent()
        try:
            result = asyncio.run(agent.run("套餐推荐", []))
            self.assertIsInstance(result, list)
        except Exception as exc:
            self.fail(f"空输入不应抛出异常: {exc}")

    def test_chunks_sorted_by_score_after_processing(self):
        """sort_data 工具运行后，处理结果应按 score 降序排列。"""
        self._setup_ws()
        agent = self._make_agent(enable_skills=False)
        chunks = asyncio.run(agent.run(
            "套餐推荐",
            [_chunk("c1", score=0.3), _chunk("c2", score=0.9),
             _chunk("c3", score=0.6)],
        ))
        if len(chunks) >= 2:
            scores = [c.score for c in chunks]
            self.assertEqual(scores, sorted(scores, reverse=True),
                             "处理后的 chunks 应按 score 降序排列")


if __name__ == "__main__":
    unittest.main(verbosity=2)
