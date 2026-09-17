"""Top3 回答充分性校验器测试。"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from dataclasses import fields
from typing import get_args

sys.path.insert(0, "src")

from langchain_core.messages import AIMessage

from kbagent.processing.verifier import Top3AnswerabilityVerifier
from kbagent.scripted_model import ScriptedChatModel
from kbagent.shared.knowledge_processing.models import (
    ProcessedKnowledge,
    RetrievalFeedback,
    RetryStrategy,
    Top3VerificationResult,
    VerificationReasonCode,
    VerificationStatus,
)


class _StaticModel:
    def __init__(self, response: str = "{}", *, delay: float = 0, error: Exception | None = None):
        self.response = response
        self.delay = delay
        self.error = error
        self.calls: list[list[object]] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.response)


def _candidate(index: int, content: str | None = None) -> ProcessedKnowledge:
    return ProcessedKnowledge(
        knowledge_id=f"knowledge-{index}",
        chunk_id=f"chunk-{index}",
        name=f"套餐知识 {index}",
        content_md=content or f"# 套餐知识 {index}\n\n套餐资费为 {index * 10} 元。",
        retrieval_rank=index,
    )


def _response(
    *,
    status="passed",
    reason_codes=None,
    evidence_ids=None,
    feedback=None,
):
    return json.dumps({
        "status": status,
        "reason_codes": [] if reason_codes is None else reason_codes,
        "summary": "校验结果说明",
        "evidence_ids": ["E001"] if evidence_ids is None else evidence_ids,
        "retrieval_feedback": feedback,
    }, ensure_ascii=False)


class TestVerifierModels(unittest.TestCase):
    def test_exact_fields_and_literal_values(self):
        self.assertEqual(
            [field.name for field in fields(RetrievalFeedback)],
            ["suggested_query", "missing_aspects", "suggested_keywords", "retry_strategy"],
        )
        self.assertEqual(
            [field.name for field in fields(Top3VerificationResult)],
            [
                "status",
                "reason_codes",
                "summary",
                "evidence_chunk_ids",
                "retrieval_feedback",
            ],
        )
        self.assertEqual(get_args(VerificationStatus), ("passed", "failed", "unknown"))
        self.assertEqual(len(get_args(VerificationReasonCode)), 8)
        self.assertEqual(len(get_args(RetryStrategy)), 4)

    def test_retry_strategy_is_required(self):
        with self.assertRaises(TypeError):
            RetrievalFeedback(
                suggested_query="套餐资费",
                missing_aspects=["缺少套餐资费"],
                suggested_keywords=["套餐资费"],
            )

    def test_status_consistency_and_reason_strategy_conflicts(self):
        feedback = RetrievalFeedback(
            suggested_query="套餐办理条件",
            missing_aspects=["缺少套餐办理条件"],
            suggested_keywords=["办理条件"],
            retry_strategy="supplement_missing_aspects",
        )
        with self.assertRaises(ValueError):
            Top3VerificationResult(
                "unknown", ["missing_key_fact"], "异常", [], None
            )
        with self.assertRaises(ValueError):
            Top3VerificationResult(
                "failed", ["verifier_timeout"], "失败", [], feedback
            )
        with self.assertRaises(ValueError):
            Top3VerificationResult(
                "failed", ["off_topic"], "偏题", [], feedback
            )
        with self.assertRaises(ValueError):
            Top3VerificationResult(
                "passed", [], "通过", [], None
            )


class TestTop3AnswerabilityVerifier(unittest.TestCase):
    def test_passed_maps_and_deduplicates_evidence_ids(self):
        model = _StaticModel(_response(evidence_ids=["E001", "E001", "E999"]))
        result = asyncio.run(Top3AnswerabilityVerifier(model).verify(
            "套餐资费是多少", [_candidate(1), _candidate(2)]
        ))

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.reason_codes, [])
        self.assertEqual(result.evidence_chunk_ids, ["chunk-1"])
        self.assertIsNone(result.retrieval_feedback)
        prompt = "\n".join(str(message.content) for message in model.calls[0])
        self.assertIn("E001", prompt)
        self.assertNotIn("chunk-1", prompt)

    def test_failed_returns_clean_retrieval_feedback(self):
        feedback = {
            "suggested_query": "套餐办理条件和适用渠道",
            "missing_aspects": ["缺少套餐的办理条件和适用渠道"],
            "suggested_keywords": ["办理条件", "信息", "办理条件", "适用渠道"],
            "retry_strategy": "supplement_missing_aspects",
        }
        model = _StaticModel(_response(
            status="failed",
            reason_codes=["partial_intent_coverage"],
            evidence_ids=["E002"],
            feedback=feedback,
        ))
        result = asyncio.run(Top3AnswerabilityVerifier(model).verify(
            "套餐资费和办理条件", [_candidate(1), _candidate(2)]
        ))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason_codes, ["partial_intent_coverage"])
        self.assertEqual(result.evidence_chunk_ids, ["chunk-2"])
        self.assertEqual(
            result.retrieval_feedback.suggested_keywords,
            ["办理条件", "适用渠道"],
        )

    def test_invalid_outputs_become_unknown(self):
        cases = {
            "invalid_json": "not json",
            "unknown_from_model": _response(status="unknown"),
            "missing_retry_strategy": json.dumps({
                "status": "failed",
                "reason_codes": ["missing_key_fact"],
                "summary": "缺少资费",
                "evidence_ids": ["E001"],
                "retrieval_feedback": {
                    "suggested_query": "套餐资费",
                    "missing_aspects": ["缺少套餐具体资费"],
                    "suggested_keywords": ["套餐资费"],
                },
            }, ensure_ascii=False),
            "invalid_retry_strategy": _response(
                status="failed",
                reason_codes=["missing_key_fact"],
                feedback={
                    "suggested_query": "套餐资费",
                    "missing_aspects": ["缺少套餐具体资费"],
                    "suggested_keywords": ["套餐资费"],
                    "retry_strategy": "invalid",
                },
            ),
            "conflicting_strategy": _response(
                status="failed",
                reason_codes=["off_topic"],
                feedback={
                    "suggested_query": "套餐资费",
                    "missing_aspects": ["当前结果偏离套餐资费主题"],
                    "suggested_keywords": ["套餐资费"],
                    "retry_strategy": "supplement_missing_aspects",
                },
            ),
        }
        for name, response in cases.items():
            with self.subTest(name=name):
                result = asyncio.run(Top3AnswerabilityVerifier(
                    _StaticModel(response)
                ).verify("套餐资费", [_candidate(1)]))
                self.assertEqual(result.status, "unknown")
                self.assertEqual(result.reason_codes, ["verifier_invalid_output"])
                self.assertIsNone(result.retrieval_feedback)

    def test_timeout_and_model_error_become_unknown(self):
        timeout = asyncio.run(Top3AnswerabilityVerifier(
            _StaticModel(delay=0.03), timeout_seconds=0.001
        ).verify("套餐资费", [_candidate(1)]))
        model_error = asyncio.run(Top3AnswerabilityVerifier(
            _StaticModel(error=RuntimeError("unavailable"))
        ).verify("套餐资费", [_candidate(1)]))

        self.assertEqual(timeout.status, "unknown")
        self.assertEqual(timeout.reason_codes, ["verifier_timeout"])
        self.assertEqual(model_error.status, "unknown")
        self.assertEqual(model_error.reason_codes, ["verifier_model_error"])
        self.assertIsNone(timeout.retrieval_feedback)
        self.assertIsNone(model_error.retrieval_feedback)

    def test_empty_or_invalid_candidates_use_deterministic_feedback(self):
        for candidates in ([], [ProcessedKnowledge(chunk_id="", content_md="")]):
            with self.subTest(candidates=candidates):
                model = _StaticModel()
                result = asyncio.run(Top3AnswerabilityVerifier(model).verify(
                    "查询套餐办理条件", candidates
                ))
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason_codes, ["no_valid_candidates"])
                self.assertEqual(result.evidence_chunk_ids, [])
                self.assertEqual(
                    result.retrieval_feedback.retry_strategy,
                    "broaden_semantic_recall",
                )
                self.assertIn("套餐", result.retrieval_feedback.suggested_query)
                self.assertTrue(result.retrieval_feedback.missing_aspects)
                self.assertTrue(result.retrieval_feedback.suggested_keywords)
                self.assertEqual(model.calls, [])

    def test_scripted_model_supports_offline_pass_and_fail(self):
        model = ScriptedChatModel()
        passed = asyncio.run(Top3AnswerabilityVerifier(model).verify(
            "套餐资费", [_candidate(1, "# 套餐资费\n\n套餐资费为 10 元。")]
        ))
        failed = asyncio.run(Top3AnswerabilityVerifier(model).verify(
            "宽带安装地址", [_candidate(1, "# 套餐资费\n\n套餐资费为 10 元。")]
        ))

        self.assertEqual(passed.status, "passed")
        self.assertEqual(passed.evidence_chunk_ids, ["chunk-1"])
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.reason_codes, ["off_topic"])
        self.assertEqual(
            failed.retrieval_feedback.retry_strategy,
            "replace_off_topic_results",
        )


if __name__ == "__main__":
    unittest.main()
