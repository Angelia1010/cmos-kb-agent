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
from kbagent.shared.knowledge_processing.adapter import (  # noqa: E402
    normalize_knowledge_candidate,
)
from kbagent.shared.knowledge_processing.bridge import (  # noqa: E402
    index_source_chunks,
    retrieval_to_candidates,
)
from kbagent.shared.knowledge_processing.markdown import (  # noqa: E402
    build_knowledge_markdown,
)
from kbagent.shared.knowledge_processing.models import ProcessedKnowledge  # noqa: E402
from kbagent.shared.models import Chunk  # noqa: E402
from kbagent.shared.search import merged_to_chunks  # noqa: E402
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
        chunk_id=f"chunk-{knowledge_id}",
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


def _chunk(prefix: str, index: int, *, status: str = "1") -> Chunk:
    knowledge_id = f"{prefix}-{index:03d}"
    return Chunk(
        chunk_id=f"chunk-{knowledge_id}",
        doc_id=knowledge_id,
        doc_title=f"流量知识{index}",
        content=f"检索阶段内容-{index}",
        category="套餐",
        position={"rank": index},
        version=f"v{index}",
        updated_at=f"2026-09-{index:02d}",
        score=round(1 - index / 100, 2),
        source_chunk_ids=[f"source-{index}"],
        extra={
            "status": status,
            "matched_atom_ids": [f"{prefix}-A-{index:03d}"],
            "source_routes": ["synthetic"],
            "knowledge_type": "业务说明",
            "template_id": "T001",
            "atoms": [{
            "atom_id": f"{prefix}-A-{index:03d}",
            "param_name": "业务内容",
            "content": f"第{index}条流量办理说明",
            "arrange_seq_number": 1,
            }],
        },
    )


def _workspace(chunks: list[Chunk]) -> tuple[RunWorkspace, dict[str, Chunk]]:
    return RunWorkspace(
        query="流量查询",
        data={
            "chunks": chunks,
            "retrieval_query": "查询剩余流量和使用明细",
            "processing_context": {
                "region_id": "200",
                "region_name": "广东",
                "channel_code": "1",
                "request_time": "2026-09-02T10:00:00+08:00",
                "audience": "agent",
                "customer_type": "个人客户",
            },
            "knowledge_candidates": retrieval_to_candidates(chunks=chunks),
        },
    ), index_source_chunks(chunks)


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
        source_chunks = {
            candidate.chunk_id: Chunk(
                chunk_id=candidate.chunk_id,
                doc_id=f"doc-{candidate.knowledge_id}",
                doc_title=f"原始标题-{candidate.knowledge_id}",
                content=f"原始内容-{candidate.knowledge_id}",
                category="套餐",
                position={"index": index},
                version=f"v{index}",
                updated_at=f"2026-09-0{index}",
                score=float(index),
                source_chunk_ids=[f"source-{index}"],
                extra={"atoms": [{"content": SENSITIVE_MARKER}]},
            )
            for index, candidate in enumerate(candidates, 1)
        }
        source_original = copy.deepcopy(source_chunks)
        original = copy.deepcopy(candidates)

        chunks = top3_to_processed_chunks(
            [candidates[2], candidates[0], candidates[1]], source_chunks
        )

        self.assertEqual(
            ["chunk-K003", "chunk-K001", "chunk-K002"],
            [item.chunk_id for item in chunks],
        )
        self.assertEqual(["doc-K003", "doc-K001", "doc-K002"], [item.doc_id for item in chunks])
        self.assertEqual([3.0, 1.0, 2.0], [item.score for item in chunks])
        self.assertEqual(candidates[2].content_md, chunks[0].content)
        self.assertIn("| 剩余流量 | 10GB |", chunks[0].content)
        self.assertTrue(all(isinstance(item, Chunk) for item in chunks))
        self.assertTrue(all(item.category == "套餐" for item in chunks))
        self.assertEqual([{"index": 3}, {"index": 1}, {"index": 2}], [item.position for item in chunks])
        self.assertEqual(["v3", "v1", "v2"], [item.version for item in chunks])
        self.assertEqual(
            ["2026-09-03", "2026-09-01", "2026-09-02"],
            [item.updated_at for item in chunks],
        )
        self.assertEqual([["source-3"], ["source-1"], ["source-2"]], [item.source_chunk_ids for item in chunks])

        self.assertTrue(all(set(item.extra) == {"processing"} for item in chunks))
        self.assertTrue(all(
            set(item.extra["processing"]) == {"rerank_rank"} for item in chunks
        ))
        serialized = json.dumps([item.to_dict() for item in chunks], ensure_ascii=False)
        self.assertNotIn("atoms", serialized)
        self.assertNotIn("raw", serialized)
        self.assertNotIn("metadata", serialized)
        self.assertNotIn(SENSITIVE_MARKER, serialized)
        self.assertEqual(original, candidates)
        self.assertEqual(source_original, source_chunks)

    def test_empty_input_and_missing_id(self):
        self.assertEqual([], top3_to_processed_chunks([], {}))
        missing_id = _processed("MISSING", retrieval_rank=1, retrieval_score=0.5)
        missing_id.chunk_id = ""
        with self.assertRaisesRegex(ValueError, "chunk_id"):
            top3_to_processed_chunks([missing_id], {})
        unknown = _processed("UNKNOWN", retrieval_rank=1, retrieval_score=0.5)
        with self.assertRaisesRegex(ValueError, "找不到 chunk_id"):
            top3_to_processed_chunks([unknown], {})


class TestRetrievalChunkBridge(unittest.TestCase):
    def test_chunk_to_candidate_field_mapping(self):
        chunk = _chunk("MAP", 2)

        raw_candidate = retrieval_to_candidates(chunks=[chunk])[0]
        candidate = normalize_knowledge_candidate(raw_candidate)

        self.assertEqual(0, raw_candidate["source_index"])
        self.assertEqual(chunk.chunk_id, candidate.chunk_id)
        self.assertEqual(chunk.doc_id, candidate.knowledge_id)
        self.assertEqual(chunk.doc_title, candidate.name)
        self.assertEqual(chunk.score, candidate.retrieval_score)
        self.assertEqual(1, candidate.retrieval_rank)
        self.assertEqual(0, candidate.source_index)
        self.assertEqual("", candidate.content)
        self.assertEqual(chunk.extra["atoms"][0]["atom_id"], candidate.atoms[0].atom_id)
        self.assertEqual(chunk.extra["atoms"][0]["content"], candidate.atoms[0].content)
        self.assertEqual(chunk.extra["matched_atom_ids"], candidate.matched_atom_ids)
        self.assertEqual(chunk.extra["source_routes"], candidate.source_routes)
        self.assertEqual(chunk.extra["knowledge_type"], candidate.knowledge_type)
        self.assertEqual(chunk.extra["template_id"], candidate.template_id)

    def test_missing_atoms_stays_empty_and_enters_existing_warning_filter(self):
        chunk = _chunk("EMPTY", 1)
        chunk.extra.pop("atoms")
        candidate = normalize_knowledge_candidate(
            retrieval_to_candidates(chunks=[chunk])[0]
        )

        self.assertEqual([], candidate.atoms)
        self.assertEqual("", candidate.content)
        self.assertNotEqual(chunk.content, candidate.content)
        processed, warnings = build_knowledge_markdown([candidate])
        self.assertEqual([], processed)
        self.assertIn("empty_rendered_content", [warning.code for warning in warnings])

    def test_duplicate_chunk_id_is_rejected(self):
        first = _chunk("DUP", 1)
        second = _chunk("DUP", 2)
        second.chunk_id = first.chunk_id

        with self.assertRaisesRegex(ValueError, "重复 chunk_id"):
            index_source_chunks([first, second])
        with self.assertRaisesRegex(ValueError, "重复 chunk_id"):
            retrieval_to_candidates(chunks=[first, second])

    def test_main_agent_legacy_call_shape_uses_chunks_only(self):
        chunk = _chunk("COMPAT", 1)
        misleading_merged = [{"knowledgeId": "WRONG", "atoms": []}]

        candidates = retrieval_to_candidates(
            merged=misleading_merged,
            chunks=[chunk],
        )

        self.assertEqual([chunk.chunk_id], [item["chunk_id"] for item in candidates])
        self.assertEqual([chunk.doc_id], [item["knowledge_id"] for item in candidates])
        with self.assertRaisesRegex(ValueError, "缺少 Retrieval chunks"):
            retrieval_to_candidates(merged=misleading_merged)


class TestOrchestratorProcessedChunks(unittest.IsolatedAsyncioTestCase):
    async def _run(self, count: int, model=None):
        ws, source_chunks = _workspace([
            _chunk("K", index) for index in range(1, count + 1)
        ])
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
            [item.chunk_id for item in result],
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
        ws, source_chunks = _workspace([_chunk("K", index) for index in range(1, 5)])
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

    async def test_filtering_keeps_each_output_aligned_by_chunk_id(self):
        chunks = [
            _chunk("ALIGN", 1),
            _chunk("ALIGN", 2, status="下架"),
            _chunk("ALIGN", 3),
            _chunk("ALIGN", 4),
        ]
        ws, source_chunks = _workspace(chunks)

        with workspace_scope(ws):
            await KnowledgeProcessingOrchestrator(ScriptedChatModel()).run()

        outputs = ws.data["processed_chunks"]
        self.assertEqual(
            [chunks[0].chunk_id, chunks[2].chunk_id, chunks[3].chunk_id],
            [item.chunk_id for item in outputs],
        )
        self.assertNotIn(chunks[1].chunk_id, [item.chunk_id for item in outputs])
        for output in outputs:
            source = source_chunks[output.chunk_id]
            self.assertEqual(source.doc_id, output.doc_id)
            self.assertEqual(source.doc_title, output.doc_title)
            self.assertEqual(source.category, output.category)
            self.assertEqual(source.position, output.position)
            self.assertEqual(source.version, output.version)
            self.assertEqual(source.updated_at, output.updated_at)
            self.assertEqual(source.score, output.score)
            self.assertEqual(source.source_chunk_ids, output.source_chunk_ids)

    async def test_merged_results_to_processed_chunk_single_chain(self):
        merged = [{
            "knowledgeId": "MERGED-001",
            "knowledgeName": "融合知识",
            "status": "1",
            "category": "套餐",
            "atoms": [{
                "klgAttrAtomId": "ATOM-001",
                "paramName": "业务内容",
                "content": "真实原子内容",
                "arrangeSeqNumber": 1,
            }],
        }]
        chunks = merged_to_chunks(merged)
        original = copy.deepcopy(chunks)
        ws, source_chunks = _workspace(chunks)

        with workspace_scope(ws):
            await KnowledgeProcessingOrchestrator(ScriptedChatModel()).run()

        self.assertEqual(1, len(ws.data["processed_chunks"]))
        output = ws.data["processed_chunks"][0]
        self.assertEqual(chunks[0].chunk_id, output.chunk_id)
        self.assertEqual(chunks[0].doc_id, output.doc_id)
        self.assertIn("真实原子内容", output.content)
        self.assertEqual({"processing": {"rerank_rank": 1}}, output.extra)
        self.assertNotIn("atoms", json.dumps(output.extra, ensure_ascii=False))
        self.assertEqual(original, chunks)

    async def test_repeated_run_and_conversion_error_do_not_leave_stale_chunks(self):
        ws, source_chunks = _workspace([
            _chunk("FIRST", index) for index in range(1, 5)
        ])
        orchestrator = KnowledgeProcessingOrchestrator(ScriptedChatModel())
        with workspace_scope(ws):
            await orchestrator.run()
            self.assertTrue(ws.data["processed_chunks"])

            ws.data["knowledge_candidates"] = []
            await orchestrator.run()
            self.assertEqual([], ws.data["processed_chunks"])

            second_chunks = [_chunk("SECOND", index) for index in range(1, 5)]
            ws.data["chunks"] = second_chunks
            ws.data["knowledge_candidates"] = retrieval_to_candidates(chunks=second_chunks)
            with patch(
                "kbagent.processing.agent.top3_to_processed_chunks",
                side_effect=RuntimeError("synthetic conversion error"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic conversion error"):
                    await orchestrator.run()
            self.assertEqual([], ws.data["processed_chunks"])

    async def test_missing_workspace_chunks_fails_and_clears_stale_output(self):
        ws, _ = _workspace([_chunk("MISSING", 1)])
        ws.data.pop("chunks")
        ws.data["processed_chunks"] = [_chunk("STALE", 1)]

        with workspace_scope(ws):
            with self.assertRaisesRegex(RuntimeError, "缺少原始 chunks"):
                await KnowledgeProcessingOrchestrator(ScriptedChatModel()).run()

        self.assertNotIn("processed_chunks", ws.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
