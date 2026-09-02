"""基于临时证据编号的两阶段知识重排。"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from ..shared.knowledge_processing.models import (
    KnowledgeProcessingOptions,
    ProcessedKnowledge,
    ProcessingContext,
    ProcessingWarning,
    RerankResult,
)
from ..shared.knowledge_processing.eligibility import rerank_ineligibility_reason
from .prompts import RERANK_BATCH_SYSTEM_PROMPT, RERANK_GLOBAL_SYSTEM_PROMPT

_SECTION_RE = re.compile(r"(?m)(?=^#{1,6}\s+)")


def _stable_candidates(candidates: Sequence[ProcessedKnowledge]) -> List[ProcessedKnowledge]:
    return [item for _, item in sorted(
        enumerate(candidates), key=lambda pair: (pair[1].retrieval_rank, pair[0])
    )]


def _prompt_markdown(content_md: str, limit: int) -> str:
    """只保留能完整放入限额的章节，绝不从章节中间截断。"""
    content_md = str(content_md or "").strip()
    if len(content_md) <= limit:
        return content_md
    sections = [section.strip() for section in _SECTION_RE.split(content_md) if section.strip()]
    selected: List[str] = []
    used = 0
    omitted = 0
    for section in sections:
        extra = len(section) + (2 if selected else 0)
        if used + extra <= limit:
            selected.append(section)
            used += extra
        else:
            omitted += 1
    if not selected and sections:
        # 标题是定位证据所必需的，但也不截断它。
        title = sections[0]
        if "\n" not in title:
            selected.append(title)
            omitted = max(0, omitted - 1)
    if omitted:
        selected.append(f"[已按完整章节省略 {omitted} 节]")
    return "\n\n".join(selected)


def _context_payload(context: ProcessingContext) -> Dict[str, Any]:
    """仅输出重排明确需要的标准上下文字段。"""
    return {
        "region_id": context.region_id,
        "region_name": context.region_name,
        "channel_code": context.channel_code,
        "request_time": context.request_time,
        "audience": context.audience,
        "customer_type": context.customer_type,
    }


def _build_user_prompt(
    query: str,
    context: ProcessingContext,
    retrieval_query: Optional[str],
    evidence: Sequence[Tuple[str, ProcessedKnowledge]],
    top_k: int,
    options: KnowledgeProcessingOptions,
) -> str:
    candidates = []
    for evidence_id, candidate in evidence:
        candidates.append({
            "evidence_id": evidence_id,
            "title": candidate.name,
            "content_md": _prompt_markdown(
                candidate.content_md, options.prompt_max_chars_per_candidate
            ),
        })
    payload = {
        "query": query,
        "context": _context_payload(context),
        "retrieval_query": retrieval_query,
        "top_k": top_k,
        "candidates": candidates,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"RERANK_INPUT_BEGIN\n{serialized}\nRERANK_INPUT_END"


async def _invoke_ranker(
    model: Any,
    system: str,
    user: str,
    timeout_seconds: float,
) -> str:
    response = await asyncio.wait_for(
        model.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]),
        timeout=timeout_seconds,
    )
    return str(getattr(response, "content", response))


def _parse_ranked_ids(
    raw: str,
    allowed: Sequence[str],
    expected_count: int,
    stage: str,
) -> Tuple[List[str], List[ProcessingWarning], bool]:
    warnings: List[ProcessingWarning] = []
    try:
        data = json.loads(str(raw).strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return [], [ProcessingWarning(
            code="rerank_invalid_json",
            message=f"{stage} 重排未返回严格 JSON",
            field=stage,
        )], False
    if not isinstance(data, dict) or set(data) != {"ranked_ids"} or not isinstance(data.get("ranked_ids"), list):
        return [], [ProcessingWarning(
            code="rerank_invalid_schema",
            message=f"{stage} 重排 JSON 结构非法",
            field=stage,
        )], False
    allowed_set = set(allowed)
    valid: List[str] = []
    seen = set()
    had_invalid = False
    for item in data["ranked_ids"]:
        if not isinstance(item, str):
            had_invalid = True
            warnings.append(ProcessingWarning(
                code="rerank_non_string_id", message=f"{stage} 包含非字符串编号", field=stage
            ))
            continue
        if item in seen:
            had_invalid = True
            warnings.append(ProcessingWarning(
                code="rerank_duplicate_id", message=f"{stage} 包含重复编号 {item}", field=stage
            ))
            continue
        seen.add(item)
        if item not in allowed_set:
            had_invalid = True
            warnings.append(ProcessingWarning(
                code="rerank_unknown_id", message=f"{stage} 包含未知编号 {item}", field=stage
            ))
            continue
        valid.append(item)
    if len(data["ranked_ids"]) != expected_count or len(valid) != expected_count:
        had_invalid = True
        warnings.append(ProcessingWarning(
            code="rerank_wrong_count",
            message=f"{stage} 期望 {expected_count} 条，实际得到 {len(valid)} 条有效结果",
            field=stage,
        ))
    return valid[:expected_count], warnings, not had_invalid


def _fill_by_rank(
    selected_ids: Sequence[str],
    evidence: Sequence[Tuple[str, ProcessedKnowledge]],
    count: int,
) -> List[str]:
    result = list(selected_ids[:count])
    for evidence_id, _ in evidence:
        if len(result) >= count:
            break
        if evidence_id not in result:
            result.append(evidence_id)
    return result


def _assign_rerank_ranks(
    candidates: Sequence[ProcessedKnowledge],
) -> List[ProcessedKnowledge]:
    ranked: List[ProcessedKnowledge] = []
    for rank, candidate in enumerate(candidates, 1):
        copied = copy.deepcopy(candidate)
        copied.rerank_rank = rank
        ranked.append(copied)
    return ranked


def _stable_unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _warning_reasons(values: Sequence[ProcessingWarning]) -> List[str]:
    return _stable_unique([warning.code for warning in values])


def _finalization_metadata(
    *,
    model_attempted: bool,
    model_ids: Sequence[str],
    selected_ids: Sequence[str],
    model_complete: bool,
    upstream_fallback_used: bool,
    eligible_count: int,
    final_top_k: int,
    stage_reasons: Sequence[str] = (),
) -> Dict[str, Any]:
    """在唯一位置计算最终模式、降级状态、补位数量和原因。"""
    if not model_attempted:
        insufficient = eligible_count < final_top_k
        return {
            "mode": "insufficient_candidates" if insufficient else "not_needed",
            "degraded": insufficient,
            "fallback_count": 0,
            "fallback_reasons": ["insufficient_candidates"] if insufficient else [],
        }

    model_id_set = set(model_ids)
    fallback_count = sum(evidence_id not in model_id_set for evidence_id in selected_ids)
    reasons: List[str] = list(stage_reasons)
    if fallback_count:
        reasons.append("retrieval_rank_supplement")
    if not model_ids:
        reasons.append("global_model_failed")
        mode = "fallback"
    else:
        if not model_complete:
            reasons.append("incomplete_model_result")
        if upstream_fallback_used:
            reasons.append("batch_fallback_used")
        mode = "model_with_fallback" if reasons else "model"
    return {
        "mode": mode,
        "degraded": mode != "model",
        "fallback_count": fallback_count,
        "fallback_reasons": _stable_unique(reasons),
    }


async def rerank_candidates(
    model: Any,
    query: str,
    context: ProcessingContext,
    retrieval_query: Optional[str],
    candidates: Sequence[ProcessedKnowledge],
    options: KnowledgeProcessingOptions,
) -> RerankResult:
    """每批 Top5、全局最多 25 复排，最终返回 Top3。"""
    ordered = _stable_candidates(candidates)
    warnings: List[ProcessingWarning] = []
    eligible: List[ProcessedKnowledge] = []
    for candidate in ordered:
        ineligibility_reason = rerank_ineligibility_reason(candidate)
        if ineligibility_reason is None:
            eligible.append(candidate)
        elif ineligibility_reason == "missing_knowledge_id":
            warnings.append(ProcessingWarning(
                code="rerank_missing_knowledge_id",
                message="缺少知识 ID，已排除于重排",
                source_index=candidate.source_index,
                field="knowledge_id",
            ))
        else:
            warnings.append(ProcessingWarning(
                code="rerank_empty_rendered_content",
                message="候选没有可渲染的业务正文，已排除于重排",
                source_index=candidate.source_index,
                knowledge_id=candidate.knowledge_id,
                field="content_md",
            ))
    evidence_pairs = [(f"E{index:03d}", candidate) for index, candidate in enumerate(eligible, 1)]
    evidence_map = {evidence_id: candidate.knowledge_id for evidence_id, candidate in evidence_pairs}
    by_evidence = dict(evidence_pairs)
    details: Dict[str, Any] = {"batches": [], "global": {}, "eligible_count": len(eligible)}
    if len(eligible) <= options.final_top_k:
        final_meta = _finalization_metadata(
            model_attempted=False,
            model_ids=[],
            selected_ids=[pair[0] for pair in evidence_pairs],
            model_complete=True,
            upstream_fallback_used=False,
            eligible_count=len(eligible),
            final_top_k=options.final_top_k,
        )
        if final_meta["degraded"]:
            warnings.append(ProcessingWarning(
                code="rerank_insufficient_candidates",
                message=f"有效候选不足 {options.final_top_k} 条，已返回全部",
                field="rerank",
            ))
        details["global"] = {
            "selected_ids": [pair[0] for pair in evidence_pairs],
            **final_meta,
        }
        details["fallback_reasons"] = list(final_meta["fallback_reasons"])
        return RerankResult(
            _assign_rerank_ranks(eligible), evidence_map, details, warnings, final_meta["degraded"]
        )

    pool_ids: List[str] = []
    batch_fallback_used = False
    for batch_index, start in enumerate(range(0, len(evidence_pairs), options.batch_size), 1):
        batch = evidence_pairs[start:start + options.batch_size]
        expected = min(options.batch_top_k, len(batch))
        stage = f"batch_{batch_index}"
        prompt = _build_user_prompt(
            query, context, retrieval_query, batch, expected, options
        )
        raw = ""
        valid: List[str] = []
        complete = False
        batch_stage_warnings: List[ProcessingWarning] = []
        try:
            raw = await _invoke_ranker(
                model,
                RERANK_BATCH_SYSTEM_PROMPT,
                prompt,
                options.rerank_timeout_seconds,
            )
            valid, parse_warnings, complete = _parse_ranked_ids(
                raw, [item[0] for item in batch], expected, stage
            )
            warnings.extend(parse_warnings)
            batch_stage_warnings.extend(parse_warnings)
        except asyncio.TimeoutError:
            warning = ProcessingWarning(
                code="rerank_timeout",
                message=f"{stage} 模型调用超过 {options.rerank_timeout_seconds:g} 秒，已降级",
                field=stage,
            )
            warnings.append(warning)
            batch_stage_warnings.append(warning)
        except Exception as exc:  # noqa: BLE001 - 模型故障必须降级
            warning = ProcessingWarning(
                code="rerank_model_error", message=f"{stage} 模型调用失败: {exc}", field=stage
            )
            warnings.append(warning)
            batch_stage_warnings.append(warning)
        selected = _fill_by_rank(valid, batch, expected)
        batch_reasons = _warning_reasons(batch_stage_warnings)
        if len(selected) > len(valid):
            batch_reasons.append("retrieval_rank_supplement")
        batch_reasons = _stable_unique(batch_reasons)
        if not complete:
            batch_fallback_used = True
        pool_ids.extend(selected)
        details["batches"].append({
            "batch_index": batch_index,
            "input_ids": [item[0] for item in batch],
            "model_ids": valid,
            "selected_ids": selected,
            "complete": complete,
            "fallback_reasons": batch_reasons,
        })

    pool_ids = pool_ids[:options.global_pool_size]
    pool = [(evidence_id, by_evidence[evidence_id]) for evidence_id in pool_ids]
    expected_global = min(options.final_top_k, len(pool))
    global_prompt = _build_user_prompt(
        query, context, retrieval_query, pool, expected_global, options
    )
    global_valid: List[str] = []
    global_complete = False
    global_stage_warnings: List[ProcessingWarning] = []
    try:
        raw = await _invoke_ranker(
            model,
            RERANK_GLOBAL_SYSTEM_PROMPT,
            global_prompt,
            options.rerank_timeout_seconds,
        )
        global_valid, parse_warnings, global_complete = _parse_ranked_ids(
            raw, pool_ids, expected_global, "global"
        )
        warnings.extend(parse_warnings)
        global_stage_warnings.extend(parse_warnings)
    except asyncio.TimeoutError:
        warning = ProcessingWarning(
            code="rerank_timeout",
            message=f"global 模型调用超过 {options.rerank_timeout_seconds:g} 秒，已降级",
            field="global",
        )
        warnings.append(warning)
        global_stage_warnings.append(warning)
    except Exception as exc:  # noqa: BLE001
        warning = ProcessingWarning(
            code="rerank_model_error", message=f"global 模型调用失败: {exc}", field="global"
        )
        warnings.append(warning)
        global_stage_warnings.append(warning)

    if not global_valid:
        # 全局完全失败时必须从全部有效候选中降级，不只限于批内池。
        final_ids = [evidence_id for evidence_id, _ in evidence_pairs[:options.final_top_k]]
    else:
        final_ids = _fill_by_rank(global_valid, evidence_pairs, options.final_top_k)
    final_meta = _finalization_metadata(
        model_attempted=True,
        model_ids=global_valid,
        selected_ids=final_ids,
        model_complete=global_complete,
        upstream_fallback_used=batch_fallback_used,
        eligible_count=len(eligible),
        final_top_k=options.final_top_k,
        stage_reasons=_warning_reasons(global_stage_warnings),
    )
    details["global"] = {
        "pool_ids": pool_ids,
        "model_ids": global_valid,
        "selected_ids": final_ids,
        "complete": global_complete,
        **final_meta,
    }
    details["fallback_reasons"] = _stable_unique([
        *(
            reason
            for batch_detail in details["batches"]
            for reason in batch_detail["fallback_reasons"]
        ),
        *final_meta["fallback_reasons"],
    ])
    return RerankResult(
        candidates=_assign_rerank_ranks([
            by_evidence[evidence_id] for evidence_id in final_ids
        ]),
        evidence_map=evidence_map,
        details=details,
        warnings=warnings,
        degraded=final_meta["degraded"],
    )
