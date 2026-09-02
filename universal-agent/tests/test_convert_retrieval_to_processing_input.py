"""真实 Retrieval JSON 转 Processing 测试输入的局部测试。"""
from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, "src")

from scripts.convert_retrieval_to_processing_input import (  # noqa: E402
    convert_payload,
    write_conversion,
)
from kbagent.shared.knowledge_processing.adapter import (  # noqa: E402
    normalize_knowledge_candidates,
)


def _record(knowledge_id, documents):
    return {
        "knowledgeId": knowledge_id,
        "success": True,
        "response": {
            "bean": {},
            "beans": [],
            "object": {"document": documents, "querySplit": [], "total": len(documents)},
            "rtnCode": "0",
            "rtnMsg": "ok",
        },
    }


class TestRetrievalToProcessingConversion(unittest.TestCase):
    def _test_directory(self) -> Path:
        path = Path(__file__).resolve().parent / f".retrieval_conversion_test_{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(self._remove_test_directory, path)
        return path

    @staticmethod
    def _remove_test_directory(path: Path) -> None:
        if path.exists():
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
            path.rmdir()

    def test_real_document_fields_are_mapped_without_flattening_content(self):
        structured_content = [{"fileName": "说明文件", "fileId": "F001"}]
        payload = [_record("K001", [
            {
                "klgAttrAtomId": "A001",
                "groupId": "G001",
                "paramName": "业务内容",
                "paramType": "3",
                "content": "<table><tr><td>原始表格</td></tr></table>",
                "except": "",
                "annotation": {"annotation": "说明", "isImport": "0"},
                "arrangeSeqNumber": "100",
                "wkuntt": "4",
                "statusCode": "1",
                "channelCode": "app,web",
            },
            {
                "klgAttrAtomId": "A002",
                "groupId": "G001",
                "paramName": "附件",
                "paramType": "9",
                "content": structured_content,
                "arrangeSeqNumber": "200",
                "statusCode": "1",
                "channelCode": "app",
            },
        ])]

        result = convert_payload(payload)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate["knowledge_id"], "K001")
        self.assertEqual(candidate["knowledge_name"], "未命名知识-K001")
        self.assertEqual(candidate["retrieval_rank"], 1)
        self.assertEqual(candidate["retrieval_score"], 1.0)
        self.assertEqual(candidate["applicability"], {})
        self.assertEqual(len(candidate["atoms"]), 2)
        first, second = candidate["atoms"]
        self.assertEqual(first["atom_id"], "A001")
        self.assertEqual(first["group_id"], "G001")
        self.assertEqual(first["param_type"], "3")
        self.assertEqual(first["arrange_seq_number"], 100)
        self.assertEqual(first["wkuntt"], "4")
        self.assertEqual(first["except_rules"], "")
        self.assertNotIn("except", first)
        self.assertEqual(first["annotation"], {"annotation": "说明", "isImport": "0"})
        self.assertEqual(first["applicability"], {
            "status": "1", "channel_codes": ["app", "web"],
        })
        self.assertEqual(second["content"], structured_content)

    def test_duplicates_are_aggregated_and_output_reloads_and_normalizes(self):
        payload = [
            _record("K001", [{
                "klgAttrAtomId": "A001", "paramName": "内容一", "content": "A",
                "arrangeSeqNumber": "1", "score": 0.2,
            }]),
            _record("K001", [
                {
                    "klgAttrAtomId": "A001", "paramName": "重复", "content": "不应重复",
                    "arrangeSeqNumber": "1", "score": 0.5,
                },
                {
                    "klgAttrAtomId": "A002", "paramName": "内容二", "content": "B",
                    "arrangeSeqNumber": "2", "score": 0.8,
                },
            ]),
            _record(None, [{"content": "C"}]),
        ]

        root = self._test_directory()
        input_path = root / "输入.json"
        output_path = root / "输出.json"
        report_path = root / "报告.md"
        original = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        input_path.write_text(original, encoding="utf-8")
        result = write_conversion(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            query="流量查询",
        )

        reloaded = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["query"], "流量查询")
        candidates = reloaded["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual([item["retrieval_rank"] for item in candidates], [1, 2])
        self.assertEqual(len({item["knowledge_id"] for item in candidates}), 2)
        self.assertEqual(candidates[0]["retrieval_score"], 0.8)
        self.assertEqual([atom["atom_id"] for atom in candidates[0]["atoms"]], [
            "A001", "A002",
        ])
        self.assertEqual(candidates[1]["knowledge_id"], "MOCK-KNOWLEDGE-003")
        self.assertEqual(candidates[1]["atoms"][0]["atom_id"], "MOCK-KNOWLEDGE-003-ATOM-001")
        normalization = normalize_knowledge_candidates(candidates)
        self.assertEqual(len(normalization.candidates), 2)
        self.assertEqual(input_path.read_text(encoding="utf-8"), original)
        self.assertIn("流量查询", report_path.read_text(encoding="utf-8"))
        self.assertEqual(result.stats["duplicate_knowledge_id_count"], 1)
        self.assertEqual(result.stats["duplicate_atom_id_count"], 1)


if __name__ == "__main__":
    unittest.main()
