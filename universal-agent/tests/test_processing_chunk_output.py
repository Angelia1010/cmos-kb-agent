"""Processing Top3 到 processed Chunk 输出边界的离线测试。"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kbagent.processing.agent import KnowledgeProcessingOrchestrator  # noqa: E402
from kbagent.processing.output import top3_to_processed_chunks  # noqa: E402
from kbagent.scripted_model import ScriptedChatModel  # noqa: E402
from kbagent.shared.knowledge_processing.models import ProcessedKnowledge  # noqa: E402
from kbagent.shared.models import Chunk  # noqa: E402
from kbagent.shared.workspace import RunWorkspace, workspace_scope  # noqa: E402


SENSITIVE_MARKER = "SYNTHETIC_RAW_METADATA_MUST_NOT_LEAK"


def _processed(
    knowledge_id: str,
    *,
    retrieval_rank: int,
    retrieval_score: float | None,
) -> ProcessedKnowledge:
    return ProcessedKnowledge(
        knowledge_id=knowledge_id,
        name=f"标题-{knowledge_id}",
        retrieval_rank=retrieval_rank,
        retrieval_score=retrieval_score,
        matched_atom_ids=[f"{knowledge_id}-A002", f"{knowledge_id}-A001"],
        source_routes=["keyword", "vector"],
        knowledge_type="业务说明",
        template_id="T001",
        metadata={"secret": SENSITIVE_MARKER},
        raw={"secret": SENSITIVE_MARKER},
        content_md=(
            f"# 标题-{knowledge_id}\n\n"
            "- 第一项\n"
            "- 第二项\n\n"
            "| 名称 | 数值 |\n"
            "| --- | --- |\n"
            "| 剩余流量 | 10GB |"
        ),
        included_atom_count=2,
        rerank_rank=retrieval_rank,
    )


def _candidate(prefix: str, index: int) -> dict:
    return {
        "knowledge_id": f"{prefix}-{index:03d}",
        "knowledge_name": f"流量知识{index}",
        "retrieval_rank": index,
        "retrieval_score": round(1 - index / 100, 2),
        "matched_atom_ids": [f"{prefix}-A-{index:03d}"],
        "source_routes": ["synthetic"],
        "knowledge_type": "业务说明",
        "template_id": "T001",
        "applicability": {"status": "1"},
        "atoms": [{
            "atom_id": f"{prefix}-A-{index:03d}",
            "param_name": "业务内容",
            "content": f"第{index}条流量办理说明",
            "arrange_seq_number": 1,
        }],
    }


def _workspace(candidates: list[dict]) -> RunWorkspace:
    return RunWorkspace(
        query="流量查询",
        data={
            "retrieval_query": "查询剩余流量和使用明细",
            "processing_context": {
                "region_id": "200",
                "region_name": "广东",
                "channel_code": "1",
                "request_time": "2026-09-02T10:00:00+08:00",
                "audience": "agent",
                "customer_type": "个人客户",
            },
            "knowledge_candidates": copy.deepcopy(candidates),
        },
    )


class _BrokenScriptedModel(ScriptedChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("synthetic rerank failure")


class TestTop3ToProcessedChunks(unittest.TestCase):
    def test_mapping_order_markdown_scores_and_whitelist(self):
        candidates = [
            _processed("K001", retrieval_rank=1, retrieval_score=0.10),
            _processed("K002", retrieval_rank=2, retrieval_score=0.90),
            _processed("K003", retrieval_rank=3, retrieval_score=None),
        ]
        original = copy.deepcopy(candidates)

        chunks = top3_to_processed_chunks(candidates)

        self.assertEqual(["K001", "K002", "K003"], [item.chunk_id for item in chunks])
        self.assertEqual(["K001", "K002", "K003"], [item.doc_id for item in chunks])
        self.assertEqual([0.10, 0.90, 0.0], [item.score for item in chunks])
        self.assertEqual([False, False, True], [
            item.extra["processing"]["score_missing"] for item in chunks
        ])
        self.assertEqual(candidates[0].content_md, chunks[0].content)
        self.assertIn("| 剩余流量 | 10GB |", chunks[0].content)
        self.assertTrue(all(isinstance(item, Chunk) for item in chunks))
        self.assertTrue(all(item.category == "" for item in chunks))
        self.assertTrue(all(item.position == {} for item in chunks))
        self.assertTrue(all(item.version == "v1.0" for item in chunks))
        self.assertTrue(all(item.updated_at == "" for item in chunks))
        self.assertTrue(all(item.source_chunk_ids == [] for item in chunks))

        expected_extra_fields = {
            "retrieval_rank",
            "rerank_rank",
            "included_atom_count",
            "matched_atom_ids",
            "source_routes",
            "knowledge_type",
            "template_id",
            "score_missing",
        }
        self.assertTrue(all(set(item.extra) == {"processing"} for item in chunks))
        self.assertTrue(all(
            set(item.extra["processing"]) == expected_extra_fields for item in chunks
        ))
        serialized = json.dumps([item.to_dict() for item in chunks], ensure_ascii=False)
        self.assertNotIn("raw", serialized)
        self.assertNotIn("metadata", serialized)
        self.assertNotIn(SENSITIVE_MARKER, serialized)
        self.assertEqual(original, candidates)

        chunks[0].extra["processing"]["matched_atom_ids"].append("NEW")
        chunks[0].extra["processing"]["source_routes"].append("NEW")
        chunks[0].position["para"] = 1
        self.assertEqual(original, candidates)
        self.assertNotIn("NEW", candidates[0].matched_atom_ids)
        self.assertNotIn("NEW", candidates[0].source_routes)
        self.assertIsNot(chunks[0].position, chunks[1].position)

    def test_empty_input_and_missing_id(self):
        self.assertEqual([], top3_to_processed_chunks([]))
        zero_score, missing_score = top3_to_processed_chunks([
            _processed("ZERO", retrieval_rank=1, retrieval_score=0.0),
            _processed("MISSING", retrieval_rank=2, retrieval_score=None),
        ])
        self.assertEqual(0.0, zero_score.score)
        self.assertFalse(zero_score.extra["processing"]["score_missing"])
        self.assertEqual(0.0, missing_score.score)
        self.assertTrue(missing_score.extra["processing"]["score_missing"])
        with self.assertRaisesRegex(ValueError, "knowledge_id"):
            top3_to_processed_chunks([
                _processed("", retrieval_rank=1, retrieval_score=0.5),
            ])


class TestOrchestratorProcessedChunks(unittest.IsolatedAsyncioTestCase):
    async def _run(self, count: int, model=None):
        ws = _workspace([_candidate("K", index) for index in range(1, count + 1)])
        with workspace_scope(ws):
            result = await KnowledgeProcessingOrchestrator(
                model or ScriptedChatModel()
            ).run()
        return ws, result

    def _assert_aligned(self, ws: RunWorkspace, result: list[ProcessedKnowledge]):
        chunks = ws.data["processed_chunks"]
        self.assertTrue(all(isinstance(item, Chunk) for item in chunks))
        self.assertTrue(all(isinstance(item, ProcessedKnowledge) for item in result))
        self.assertEqual(
            [item.knowledge_id for item in result],
            [item.chunk_id for item in chunks],
        )
        self.assertEqual(
            [item.content_md for item in result],
            [item.content for item in chunks],
        )
        self.assertEqual(len(result), len(chunks))

    async def test_normal_fallback_insufficient_and_empty_use_same_output(self):
        normal_ws, normal = await self._run(4)
        self.assertEqual(3, len(normal))
        self.assertFalse(normal_ws.data["processing_meta"].degraded)
        self._assert_aligned(normal_ws, normal)

        fallback_ws, fallback = await self._run(4, _BrokenScriptedModel())
        self.assertEqual(3, len(fallback))
        self.assertTrue(fallback_ws.data["processing_meta"].degraded)
        self._assert_aligned(fallback_ws, fallback)

        insufficient_ws, insufficient = await self._run(2)
        self.assertEqual(2, len(insufficient))
        self.assertTrue(insufficient_ws.data["processing_meta"].degraded)
        self._assert_aligned(insufficient_ws, insufficient)

        empty_ws, empty = await self._run(0)
        self.assertEqual([], empty)
        self.assertEqual([], empty_ws.data["processed_chunks"])
        self.assertTrue(empty_ws.data["processing_meta"].degraded)

    async def test_conversion_runs_once_and_keeps_four_stage_statistics(self):
        ws = _workspace([_candidate("K", index) for index in range(1, 5)])
        with workspace_scope(ws), patch(
            "kbagent.processing.agent.top3_to_processed_chunks",
            wraps=top3_to_processed_chunks,
        ) as adapter:
            result = await KnowledgeProcessingOrchestrator(ScriptedChatModel()).run()

        self.assertEqual(1, adapter.call_count)
        self.assertEqual(result, ws.data["top3_candidates"])
        self.assertEqual([
            "analyze", "filter", "build_markdown", "rerank",
        ], ws.data["processing_meta"].stage_order)

    async def test_repeated_run_and_conversion_error_do_not_leave_stale_chunks(self):
        ws = _workspace([_candidate("FIRST", index) for index in range(1, 5)])
        orchestrator = KnowledgeProcessingOrchestrator(ScriptedChatModel())
        with workspace_scope(ws):
            await orchestrator.run()
            self.assertTrue(ws.data["processed_chunks"])

            ws.data["knowledge_candidates"] = []
            await orchestrator.run()
            self.assertEqual([], ws.data["processed_chunks"])

            ws.data["knowledge_candidates"] = [
                _candidate("SECOND", index) for index in range(1, 5)
            ]
            with patch(
                "kbagent.processing.agent.top3_to_processed_chunks",
                side_effect=RuntimeError("synthetic conversion error"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic conversion error"):
                    await orchestrator.run()
            self.assertEqual([], ws.data["processed_chunks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
