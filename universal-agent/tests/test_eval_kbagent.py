"""Regression tests for the case-level Top3 retrieval hit metric."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_kbagent.py"
SPEC = importlib.util.spec_from_file_location("eval_kbagent", SCRIPT)
assert SPEC and SPEC.loader
eval_kbagent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_kbagent)


class Top3CombinedHitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            doc_chars=3000, llm_url="unused", llm_timeout=1,
            insecure=False, pass_b1_min=1, pass_b2_min=1, pass_b3_min=1,
        )

    def verdict(self, case_id: str, gold: str, relevant: str) -> dict:
        record = {
            "case_id": case_id,
            "query": "这项业务有年龄限制吗？",
            "gold_titles": ["目标业务"],
            "object": {
                "sources": [{"docId": "d1", "docTitle": "候选文档",
                             "content": "候选正文"}],
            },
        }
        judgments = [{
            "doc": "D1", "gold_match": gold,
            "query_relevant": relevant, "reason": "测试",
        }]
        with patch.object(eval_kbagent, "_judge_with_retry",
                          return_value=(judgments, None)):
            return eval_kbagent.judge_case(record, self.args)

    def test_either_gold_or_answerable_document_is_a_hit(self) -> None:
        gold_case = self.verdict("gold", "yes", "no")
        answer_case = self.verdict("answer", "no", "yes")
        partial_case = self.verdict("partial", "no", "partial")

        self.assertTrue(gold_case["retrieval"]["top3_hit"])
        self.assertTrue(answer_case["retrieval"]["top3_hit"])
        self.assertFalse(partial_case["retrieval"]["top3_hit"])
        self.assertFalse(answer_case["retrieval"]["gold_hit"])
        self.assertTrue(answer_case["retrieval"]["any_yes"])

    def test_one_hit_among_three_documents_counts_once(self) -> None:
        record = {
            "case_id": "third", "query": "这项业务有年龄限制吗？",
            "gold_titles": ["目标业务"],
            "object": {"sources": [
                {"docTitle": f"候选{i}", "content": "正文"}
                for i in range(1, 4)
            ]},
        }
        judgments = [
            {"doc": "D1", "gold_match": "no", "query_relevant": "no"},
            {"doc": "D2", "gold_match": "no", "query_relevant": "partial"},
            {"doc": "D3", "gold_match": "no", "query_relevant": "yes"},
        ]
        with patch.object(eval_kbagent, "_judge_with_retry",
                          return_value=(judgments, None)):
            verdict = eval_kbagent.judge_case(record, self.args)
        self.assertTrue(verdict["retrieval"]["top3_hit"])
        self.assertEqual(0.5, verdict["retrieval"]["precision_at_3"])

    def test_report_recomputes_metric_from_legacy_verdicts(self) -> None:
        verdicts = [
            self.verdict("gold", "yes", "no"),
            self.verdict("answer", "no", "yes"),
            self.verdict("partial", "no", "partial"),
        ]
        for verdict in verdicts:
            del verdict["retrieval"]["top3_hit"]
        with tempfile.TemporaryDirectory() as tmp:
            summary = eval_kbagent.build_report(verdicts, Path(tmp))
            self.assertEqual(
                0.6667,
                summary["A_retrieval"][
                    "A0_top3综合命中率(gold_match=yes或query_relevant=yes)"],
            )
            self.assertEqual(
                len(eval_kbagent._DETAIL_HEADER),
                len(eval_kbagent._detail_row(verdicts[0], [])),
            )


if __name__ == "__main__":
    unittest.main()
