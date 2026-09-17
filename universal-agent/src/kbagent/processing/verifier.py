"""独立的 Top3 回答充分性校验器。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..shared import lexicon
from ..shared.knowledge_processing.models import (
    ProcessedKnowledge,
    ProcessingContext,
    RetrievalFeedback,
    Top3VerificationResult,
)
from .prompts import TOP3_ANSWERABILITY_SYSTEM_PROMPT

_RESULT_FIELDS = frozenset({
    "status",
    "reason_codes",
    "summary",
    "evidence_ids",
    "retrieval_feedback",
})
_FEEDBACK_FIELDS = frozenset({
    "suggested_query",
    "missing_aspects",
    "suggested_keywords",
    "retry_strategy",
})
_GENERIC_KEYWORDS = frozenset({"信息", "内容", "知识", "问题", "相关", "业务"})


class _InvalidVerifierOutput(ValueError):
    pass


def _context_payload(context: ProcessingContext | None) -> dict[str, Any]:
    if context is None:
        return {}
    return {
        "region_id": context.region_id,
        "region_name": context.region_name,
        "channel_code": context.channel_code,
        "request_time": context.request_time,
        "audience": context.audience,
        "customer_type": context.customer_type,
    }


def _stable_keywords(values: Sequence[Any]) -> list[str]:
    keywords: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise _InvalidVerifierOutput("suggested_keywords 必须是字符串列表")
        keyword = value.strip()
        if keyword and keyword not in _GENERIC_KEYWORDS and keyword not in keywords:
            keywords.append(keyword)
    return keywords


def _empty_candidate_keywords(query: str) -> list[str]:
    extracted = lexicon.extract_keywords(query)
    try:
        keywords = _stable_keywords(extracted)
    except _InvalidVerifierOutput:
        keywords = []
    return keywords or [f"{query}业务规则"]


def _no_valid_candidates_result(query: str) -> Top3VerificationResult:
    return Top3VerificationResult(
        status="failed",
        reason_codes=["no_valid_candidates"],
        summary="当前没有可用于验证的有效候选知识，需要扩大检索范围。",
        evidence_chunk_ids=[],
        retrieval_feedback=RetrievalFeedback(
            suggested_query=query,
            missing_aspects=[f"缺少能够回答“{query}”的有效候选知识"],
            suggested_keywords=_empty_candidate_keywords(query),
            retry_strategy="broaden_semantic_recall",
        ),
    )


def _unknown_result(reason_code: str, summary: str) -> Top3VerificationResult:
    return Top3VerificationResult(
        status="unknown",
        reason_codes=[reason_code],
        summary=summary,
        evidence_chunk_ids=[],
        retrieval_feedback=None,
    )


def _prepare_evidence(
    candidates: Sequence[ProcessedKnowledge],
) -> list[tuple[str, str, ProcessedKnowledge]]:
    evidence: list[tuple[str, str, ProcessedKnowledge]] = []
    seen_chunk_ids: set[str] = set()
    for candidate in list(candidates)[:3]:
        if not isinstance(candidate, ProcessedKnowledge):
            continue
        chunk_id = str(candidate.chunk_id or "").strip()
        content = str(candidate.content_md or "").strip()
        if not chunk_id or not content or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        evidence_id = f"E{len(evidence) + 1:03d}"
        evidence.append((evidence_id, chunk_id, candidate))
    return evidence


def _build_user_prompt(
    query: str,
    retrieval_query: str | None,
    context: ProcessingContext | None,
    evidence: Sequence[tuple[str, str, ProcessedKnowledge]],
    max_chars_per_candidate: int,
) -> str:
    payload = {
        "query": query,
        "retrieval_query": retrieval_query,
        "context": _context_payload(context),
        "candidates": [
            {
                "evidence_id": evidence_id,
                "title": str(candidate.name or ""),
                "content_md": str(candidate.content_md or "")[:max_chars_per_candidate],
            }
            for evidence_id, _, candidate in evidence
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"TOP3_VERIFICATION_INPUT_BEGIN\n{serialized}\nTOP3_VERIFICATION_INPUT_END"


def _parse_feedback(value: Any) -> RetrievalFeedback | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _FEEDBACK_FIELDS:
        raise _InvalidVerifierOutput("retrieval_feedback 结构非法")
    keywords = value.get("suggested_keywords")
    if not isinstance(keywords, list):
        raise _InvalidVerifierOutput("suggested_keywords 必须是列表")
    return RetrievalFeedback(
        suggested_query=value.get("suggested_query"),
        missing_aspects=value.get("missing_aspects"),
        suggested_keywords=_stable_keywords(keywords),
        retry_strategy=value.get("retry_strategy"),
    )


def _parse_model_result(
    raw: str,
    evidence: Sequence[tuple[str, str, ProcessedKnowledge]],
) -> Top3VerificationResult:
    try:
        data = json.loads(str(raw).strip())
        if not isinstance(data, dict) or set(data) != _RESULT_FIELDS:
            raise _InvalidVerifierOutput("Verifier JSON 结构非法")
        if data.get("status") not in {"passed", "failed"}:
            raise _InvalidVerifierOutput("模型只能返回 passed 或 failed")
        reason_codes = data.get("reason_codes")
        evidence_ids = data.get("evidence_ids")
        if not isinstance(reason_codes, list) or not isinstance(evidence_ids, list):
            raise _InvalidVerifierOutput("reason_codes 和 evidence_ids 必须是列表")
        if any(not isinstance(value, str) for value in evidence_ids):
            raise _InvalidVerifierOutput("evidence_ids 必须是字符串列表")
        if "no_valid_candidates" in reason_codes:
            raise _InvalidVerifierOutput("非空候选不能返回 no_valid_candidates")

        evidence_map = {evidence_id: chunk_id for evidence_id, chunk_id, _ in evidence}
        mapped_chunk_ids = list(dict.fromkeys(
            evidence_map[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_map
        ))
        result = Top3VerificationResult(
            status=data.get("status"),
            reason_codes=reason_codes,
            summary=data.get("summary"),
            evidence_chunk_ids=mapped_chunk_ids,
            retrieval_feedback=_parse_feedback(data.get("retrieval_feedback")),
        )
    except _InvalidVerifierOutput:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise _InvalidVerifierOutput(str(exc)) from exc
    return result


class Top3AnswerabilityVerifier:
    """使用注入模型判断当前 Top3 是否足以回答原始 query。"""

    def __init__(
        self,
        model: Any,
        *,
        timeout_seconds: float = 15.0,
        max_chars_per_candidate: int = 6000,
    ) -> None:
        self._model = model
        self._timeout_seconds = max(0.001, float(timeout_seconds))
        self._max_chars_per_candidate = max(1, int(max_chars_per_candidate))

    async def verify(
        self,
        query: str,
        candidates: Sequence[ProcessedKnowledge],
        *,
        retrieval_query: str | None = None,
        context: ProcessingContext | None = None,
    ) -> Top3VerificationResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        query = query.strip()
        evidence = _prepare_evidence(candidates)
        if not evidence:
            return _no_valid_candidates_result(query)

        user_prompt = _build_user_prompt(
            query,
            retrieval_query,
            context,
            evidence,
            self._max_chars_per_candidate,
        )
        try:
            response = await asyncio.wait_for(
                self._model.ainvoke([
                    SystemMessage(content=TOP3_ANSWERABILITY_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            return _unknown_result(
                "verifier_timeout", "Verifier 模型调用超时，无法完成可靠判断。"
            )
        except Exception:  # noqa: BLE001 - 模型服务异常必须收敛为 unknown
            return _unknown_result(
                "verifier_model_error", "Verifier 模型调用失败，无法完成可靠判断。"
            )

        try:
            return _parse_model_result(str(getattr(response, "content", response)), evidence)
        except _InvalidVerifierOutput:
            return _unknown_result(
                "verifier_invalid_output", "Verifier 模型输出无法通过结构化解析和校验。"
            )
