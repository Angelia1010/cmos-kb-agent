from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kbagent.processing.agent import KnowledgeProcessingOrchestrator  # noqa: E402
from kbagent.processing.output import top3_to_processed_chunks  # noqa: E402
from kbagent.scripted_model import ScriptedChatModel  # noqa: E402
from kbagent.shared.knowledge_processing.adapter import (  # noqa: E402
    normalize_knowledge_candidate,
)
from kbagent.shared.knowledge_processing.applicability import (  # noqa: E402
    filter_candidates,
)
from kbagent.shared.knowledge_processing.bridge import (  # noqa: E402
    retrieval_to_candidates,
)
from kbagent.shared.knowledge_processing.markdown import (  # noqa: E402
    build_candidate_markdown,
)
from kbagent.shared.knowledge_processing.models import (  # noqa: E402
    KnowledgeCandidate,
    ProcessingContext,
)
from kbagent.shared.models import Chunk  # noqa: E402
from kbagent.shared.workspace import RunWorkspace, workspace_scope  # noqa: E402


def _raw_chunk(
    *,
    title: str = "流量业务汇总",
    content: str = "流量业务汇总：套餐介绍:业务正文",
    group_name: str | None = "套餐介绍",
) -> Chunk:
    return Chunk(
        chunk_id="vector-001",
        doc_id="K-001",
        doc_title=title,
        content=content,
        category="",
        score=0.0,
        extra={
            "source": "vector",
            "vector_channel": "vector",
            "raw": {
                "groupName": group_name,
                "startTime": "2020-01-01T00:00:00",
                "endTime": "2099-01-01T23:59:59",
                "provinceId": "931",
                "channelCode": ["1", "2", "esoph5"],
                "score": 0.88,
                "private": "RAW_MUST_NOT_LEAK",
            },
        },
    )


def _candidate(chunk: Chunk):
    return normalize_knowledge_candidate(
        retrieval_to_candidates(chunks=[chunk])[0]
    )


class TestRawVectorBridge(unittest.TestCase):
    def test_raw_vector_mapping_and_input_immutability(self):
        chunk = _raw_chunk()
        original = copy.deepcopy(chunk)

        raw = retrieval_to_candidates(chunks=[chunk])[0]
        candidate = normalize_knowledge_candidate(raw)

        self.assertEqual(chunk.content, raw["content"])
        self.assertEqual("套餐介绍", candidate.content_group_name)
        self.assertEqual([], candidate.atoms)
        self.assertEqual("2020-01-01T00:00:00", candidate.applicability.effective_start)
        self.assertEqual("2099-01-01T23:59:59", candidate.applicability.effective_end)
        self.assertEqual(["931"], candidate.region_ids)
        self.assertEqual(["1", "2", "esoph5"], candidate.channel_codes)
        self.assertEqual(0.0, candidate.retrieval_score)
        self.assertNotIn("RAW_MUST_NOT_LEAK", repr(raw))
        self.assertEqual(original, chunk)

    def test_atoms_take_priority_and_legacy_missing_atoms_stays_empty(self):
        atom_chunk = _raw_chunk()
        atom_chunk.extra["atoms"] = [{
            "klgAttrAtomId": "A1",
            "paramName": "套餐说明",
            "groupId": "详情",
            "content": "结构化正文",
        }]
        atom_raw = retrieval_to_candidates(chunks=[atom_chunk])[0]
        self.assertEqual("", atom_raw["content"])
        self.assertIsNone(atom_raw["content_group_name"])
        self.assertEqual(1, len(atom_raw["atoms"]))

        legacy = _raw_chunk()
        legacy.extra = {}
        legacy_raw = retrieval_to_candidates(chunks=[legacy])[0]
        self.assertEqual("", legacy_raw["content"])
        self.assertIsNone(legacy_raw["content_group_name"])
        self.assertEqual([], legacy_raw["atoms"])

    def test_raw_string_null_values_are_not_mapped(self):
        chunk = _raw_chunk(group_name="null")
        chunk.extra["raw"].update({
            "startTime": "null",
            "endTime": None,
            "provinceId": "NULL",
            "channelCode": [None, "null", "1"],
        })
        candidate = _candidate(chunk)
        self.assertEqual("", candidate.content_group_name)
        self.assertIsNone(candidate.applicability.effective_start)
        self.assertIsNone(candidate.applicability.effective_end)
        self.assertEqual([], candidate.region_ids)
        self.assertEqual(["1"], candidate.channel_codes)

    def test_title_cleanup_is_shared_by_atoms_and_object_inputs(self):
        atom_chunk = _raw_chunk(
            title='<span class="keywords">5G</span><br></br>流量包'
        )
        atom_chunk.extra["atoms"] = [{
            "klgAttrAtomId": "A1",
            "paramName": "说明",
            "groupId": "详情",
            "content": "正文",
        }]
        candidate = _candidate(atom_chunk)
        processed = build_candidate_markdown(candidate)
        self.assertEqual("5G 流量包", candidate.name)
        self.assertTrue(processed.content_md.startswith("# 5G 流量包\n"))
        self.assertNotIn("<span", processed.content_md)
        self.assertNotIn("<br", processed.content_md)

        object_candidate = normalize_knowledge_candidate(KnowledgeCandidate(
            knowledge_id="K-object",
            name="<b>对象</b><br>标题",
            content="正文",
        ))
        self.assertEqual("对象 标题", object_candidate.name)


class TestRawVectorFiltering(unittest.TestCase):
    def setUp(self):
        self.candidate = _candidate(_raw_chunk())

    def _decision(self, **context):
        kept, decisions, _ = filter_candidates(
            [self.candidate], ProcessingContext(**context)
        )
        return kept, decisions[0]

    def test_partial_or_missing_context_only_checks_available_dimensions(self):
        kept, decision = self._decision(
            request_time="2026-09-17T12:00:00+08:00"
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(1, len(kept))

        kept, decision = self._decision(
            region_id="931", request_time="2026-09-17T12:00:00+08:00"
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(1, len(kept))

        _, region = self._decision(
            region_id="999", request_time="2026-09-17T12:00:00+08:00"
        )
        self.assertIn("region_not_applicable", region.reasons)

        _, channel = self._decision(
            channel_code="blocked", request_time="2026-09-17T12:00:00+08:00"
        )
        self.assertIn("channel_not_applicable", channel.reasons)

    def test_time_filter_and_empty_business_content(self):
        _, future = self._decision(request_time="2019-12-31T23:59:59+08:00")
        self.assertIn("not_started", future.reasons)

        _, expired = self._decision(request_time="2100-01-01T00:00:00+08:00")
        self.assertIn("expired", expired.reasons)

        empty = _candidate(_raw_chunk(content=""))
        kept, decisions, _ = filter_candidates(
            [empty], ProcessingContext(request_time="2026-09-17T12:00:00+08:00")
        )
        self.assertEqual([], kept)
        self.assertEqual(["empty_content"], decisions[0].reasons)


class TestRawVectorMarkdown(unittest.TestCase):
    def test_group_prefix_table_period_and_processed_title(self):
        title = '<span class="keywords">任我看</span><br></br>视频流量业务汇总'
        content = (
            "任我看 视频流量业务汇总：套餐介绍:套餐简介："
            "|-|-|-|-|-|-|序号 | 文档名称 | 文档归属 | "
            "| 1 | 任我看 视频流量 | 全国 | "
            "| 2 | 任我看 视频流量 甘肃*已下线* | 甘肃 |"
        )
        chunk = _raw_chunk(title=title, content=content)
        chunk.extra["raw"]["groupName"] = (
            '<span class="keywords">套餐</span><br></br>介绍'
        )
        source_original = copy.deepcopy(chunk)
        candidate = _candidate(chunk)

        processed = build_candidate_markdown(candidate)
        markdown = processed.content_md

        self.assertEqual("任我看 视频流量业务汇总", processed.name)
        self.assertIn("# 任我看 视频流量业务汇总", markdown)
        self.assertIn("## 套餐 介绍", markdown)
        self.assertIn("| 序号 | 文档名称 | 文档归属 |", markdown)
        self.assertIn("| --- | --- | --- |", markdown)
        self.assertIn("| 2 | 任我看 视频流量 甘肃*已下线* | 甘肃 |", markdown)
        self.assertNotIn("|-|-|-", markdown)
        self.assertIn("## 有效期", markdown)
        self.assertIn("- 生效时间：2020-01-01T00:00:00", markdown)
        self.assertIn("- 失效时间：2099-01-01T23:59:59", markdown)
        self.assertEqual([], [
            warning.code for warning in processed.processing_warnings
            if warning.code == "flattened_table_unparsed"
        ])

        output = top3_to_processed_chunks(
            [processed], {chunk.chunk_id: chunk}
        )[0]
        self.assertEqual("任我看 视频流量业务汇总", output.doc_title)
        self.assertEqual(markdown, output.content)
        self.assertNotIn("raw", output.extra)
        self.assertNotIn("RAW_MUST_NOT_LEAK", repr(output.to_dict()))
        self.assertEqual(source_original, chunk)

    def test_invalid_flattened_table_is_preserved_with_warning(self):
        chunk = _raw_chunk(
            content="流量业务汇总：套餐介绍:说明：|-|-|-|A | B | | only-one |"
        )
        processed = build_candidate_markdown(_candidate(chunk))
        self.assertIn("|-|-|-|A | B | | only-one |", processed.content_md)
        self.assertIn("flattened_table_unparsed", [
            warning.code for warning in processed.processing_warnings
        ])

    def test_valid_markdown_table_is_not_rewritten(self):
        table = "| A\\|B | C |\n| --- | --- |\n| 1 | 2 |"
        chunk = _raw_chunk(
            content=f"流量业务汇总：套餐介绍:{table}"
        )
        processed = build_candidate_markdown(_candidate(chunk))
        self.assertIn(table, processed.content_md)
        self.assertNotIn("flattened_table_unparsed", [
            warning.code for warning in processed.processing_warnings
        ])


class TestRawVectorOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_atoms_and_raw_candidates_share_the_existing_pipeline(self):
        raw_chunk = _raw_chunk(
            content=(
                "流量业务汇总：套餐介绍:套餐简介："
                "|-|-|-|-|名称 | 值 | | 月费 | 20元 |"
            )
        )
        chunks = [raw_chunk]
        for index in range(1, 4):
            chunks.append(Chunk(
                chunk_id=f"atom-{index}",
                doc_id=f"A-{index}",
                doc_title=f"普通知识{index}",
                content="检索拼接内容不应使用",
                category="",
                score=1.0 - index / 10,
                extra={"atoms": [{
                    "klgAttrAtomId": f"ATOM-{index}",
                    "paramName": "说明",
                    "groupId": "详情",
                    "content": f"普通正文{index}",
                }]},
            ))
        ws = RunWorkspace(
            query="流量业务汇总月费",
            data={
                "chunks": chunks,
                "retrieval_query": "流量业务汇总月费",
                "processing_context": {
                    "request_time": "2026-09-17T12:00:00+08:00",
                },
                "knowledge_candidates": retrieval_to_candidates(chunks=chunks),
            },
        )

        with workspace_scope(ws):
            await KnowledgeProcessingOrchestrator(ScriptedChatModel()).run()

        processed = ws.data["processed_knowledge_candidates"]
        self.assertEqual(4, len(processed))
        raw_processed = next(item for item in processed if item.chunk_id == raw_chunk.chunk_id)
        self.assertIn("| 名称 | 值 |", raw_processed.content_md)
        self.assertIn("## 有效期", raw_processed.content_md)
        self.assertIn(raw_chunk.chunk_id, [item.chunk_id for item in ws.data["top3_candidates"]])
        output = next(
            item for item in ws.data["processed_chunks"]
            if item.chunk_id == raw_chunk.chunk_id
        )
        self.assertEqual(raw_processed.content_md, output.content)
        self.assertEqual(raw_processed.name, output.doc_title)


if __name__ == "__main__":
    unittest.main(verbosity=2)
