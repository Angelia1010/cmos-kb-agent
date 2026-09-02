"""知识级和原子级适用性判定。"""
from __future__ import annotations

import copy
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import (
    Applicability,
    FilterDecision,
    KnowledgeCandidate,
    ProcessingContext,
    ProcessingWarning,
)
from .eligibility import has_renderable_candidate_content

_INACTIVE = {
    "下架", "已下架", "停用", "已停用", "失效", "已失效", "废弃", "删除",
    "inactive", "disabled", "expired", "offline", "deleted", "invalid", "false", "0",
}
_ACTIVE = {
    "生效", "已生效", "有效", "启用", "已启用", "上线", "已上线",
    "active", "enabled", "valid", "online", "true", "1",
}
try:
    _BUSINESS_TIMEZONE = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:  # Windows 可能未配置系统 IANA 时区库
    _BUSINESS_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _datetime_value(value: Any, *, end_of_day: bool = False) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.max if end_of_day else time.min)
    else:
        text_value = str(value).strip()
        if not text_value:
            return None
        if text_value.endswith(("Z", "z")):
            text_value = text_value[:-1] + "+00:00"
        try:
            if len(text_value) == 10:
                parsed_date = date.fromisoformat(text_value)
                parsed = datetime.combine(parsed_date, time.max if end_of_day else time.min)
            else:
                parsed = datetime.fromisoformat(text_value)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_BUSINESS_TIMEZONE)
    return parsed.astimezone(_BUSINESS_TIMEZONE)


def _now_in_business_timezone() -> datetime:
    return datetime.now(_BUSINESS_TIMEZONE)


def _request_datetime(context: ProcessingContext) -> datetime:
    # 接口约定的 request_time 优先于旧 as_of。
    return (
        _datetime_value(context.request_time)
        or _datetime_value(context.as_of)
        or _now_in_business_timezone()
    )


def _today(context: ProcessingContext) -> date:
    """兼容旧的日期调用，日期一律以业务时区计算。"""
    return _request_datetime(context).date()


def _matches_allowed(value: str | None, allowed: Sequence[str]) -> bool:
    if not allowed or not value:
        return True
    normalized = value.strip().casefold()
    return any(str(item).strip().casefold() in {normalized, "*", "全部", "全国"} for item in allowed)


def _matches_excluded(value: str | None, excluded: Sequence[str]) -> bool:
    if not value or not excluded:
        return False
    normalized = value.strip().casefold()
    return any(str(item).strip().casefold() in {normalized, "*", "全部", "全国"} for item in excluded)


def _value_summary(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _applicability_warning(
    code: str,
    message: str,
    field: str,
    value: Any,
    *,
    knowledge_id: str | None,
    source_index: int,
) -> ProcessingWarning:
    return ProcessingWarning(
        code=code,
        message=message,
        source_index=source_index,
        knowledge_id=knowledge_id,
        field=field,
        details={"warning_code": code, "raw_value": _value_summary(value)},
    )


def collect_applicability_warnings(
    applicability: Applicability,
    *,
    knowledge_id: str | None,
    source_index: int,
    field_prefix: str = "applicability",
) -> List[ProcessingWarning]:
    """记录无法判定的适用性字段，但不将其转为过滤原因。"""
    warnings: List[ProcessingWarning] = []
    status = applicability.status
    if status not in (None, ""):
        if not isinstance(status, (str, int, float, bool)):
            warnings.append(_applicability_warning(
                "invalid_applicability_field", "适用性状态字段格式异常",
                f"{field_prefix}.status", status,
                knowledge_id=knowledge_id, source_index=source_index,
            ))
        elif str(status).strip().casefold() not in (_ACTIVE | _INACTIVE):
            warnings.append(_applicability_warning(
                "unknown_status", "无法识别适用性状态，已保留候选",
                f"{field_prefix}.status", status,
                knowledge_id=knowledge_id, source_index=source_index,
            ))

    start_value = applicability.effective_start or applicability.start_at
    if start_value not in (None, "") and _datetime_value(start_value) is None:
        warnings.append(_applicability_warning(
            "invalid_start_time", "开始时间无法解析，已保留候选",
            f"{field_prefix}.effective_start", start_value,
            knowledge_id=knowledge_id, source_index=source_index,
        ))
    end_value = applicability.effective_end or applicability.end_at
    if end_value not in (None, "") and _datetime_value(end_value, end_of_day=True) is None:
        warnings.append(_applicability_warning(
            "invalid_end_time", "结束时间无法解析，已保留候选",
            f"{field_prefix}.effective_end", end_value,
            knowledge_id=knowledge_id, source_index=source_index,
        ))

    for field in (
        "region_ids", "regions", "channel_codes", "channels",
        "excluded_regions", "excluded_channels",
    ):
        value = getattr(applicability, field)
        invalid = not isinstance(value, (list, tuple, set)) or any(
            isinstance(item, (dict, list, tuple, set)) for item in value
        )
        if invalid:
            warnings.append(_applicability_warning(
                "invalid_applicability_field", "适用性范围字段格式异常，已保留候选",
                f"{field_prefix}.{field}", value,
                knowledge_id=knowledge_id, source_index=source_index,
            ))
    if not isinstance(applicability.conditions, dict):
        warnings.append(_applicability_warning(
            "invalid_applicability_field", "适用性条件字段格式异常，已保留候选",
            f"{field_prefix}.conditions", applicability.conditions,
            knowledge_id=knowledge_id, source_index=source_index,
        ))
    return warnings


def evaluate_applicability(
    applicability: Applicability,
    context: ProcessingContext,
) -> List[str]:
    """返回不适用原因；空列表表示当前上下文可用。"""
    reasons: List[str] = []
    status = str(applicability.status or "").strip().casefold()
    if status in _INACTIVE:
        reasons.append("inactive_status")
    request_time = _request_datetime(context)
    start = _datetime_value(applicability.effective_start or applicability.start_at)
    end = _datetime_value(applicability.effective_end or applicability.end_at, end_of_day=True)
    if start and request_time < start:
        reasons.append("not_started")
    if end and request_time > end:
        reasons.append("expired")
    region_id = context.region_id
    region_name = context.region_name
    # 旧 region 只作为名称型兼容值，绝不用于 region_ids 比较。
    legacy_region_name = context.region if not region_id and not region_name else None
    effective_region_name = region_name or legacy_region_name
    channel_code = context.channel_code or context.channel
    region_ids = applicability.region_ids if isinstance(applicability.region_ids, (list, tuple, set)) else ()
    regions = applicability.regions if isinstance(applicability.regions, (list, tuple, set)) else ()
    channel_codes = applicability.channel_codes if isinstance(applicability.channel_codes, (list, tuple, set)) else ()
    channels = applicability.channels if isinstance(applicability.channels, (list, tuple, set)) else ()
    excluded_regions = (
        applicability.excluded_regions
        if isinstance(applicability.excluded_regions, (list, tuple, set)) else ()
    )
    excluded_channels = (
        applicability.excluded_channels
        if isinstance(applicability.excluded_channels, (list, tuple, set)) else ()
    )
    if region_id and not _matches_allowed(region_id, region_ids):
        reasons.append("region_not_applicable")
    if effective_region_name and not _matches_allowed(effective_region_name, regions):
        reasons.append("region_not_applicable")
    if _matches_excluded(effective_region_name, excluded_regions):
        reasons.append("region_excluded")
    if not _matches_allowed(channel_code, channel_codes):
        reasons.append("channel_not_applicable")
    if not _matches_allowed(context.channel or channel_code, channels):
        reasons.append("channel_not_applicable")
    if _matches_excluded(context.channel or channel_code, excluded_channels):
        reasons.append("channel_excluded")

    context_values: Dict[str, Any] = {
        **context.attributes,
        "region": effective_region_name,
        "region_id": region_id,
        "region_name": effective_region_name,
        "channel": context.channel or channel_code,
        "channel_code": channel_code,
        "audience": context.audience,
        "customer_type": context.customer_type,
    }
    conditions = applicability.conditions if isinstance(applicability.conditions, dict) else {}
    for key, expected in conditions.items():
        if key not in context_values or context_values[key] is None:
            continue  # 上下文不足时不激进过滤
        actual = context_values[key]
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                reasons.append(f"condition_mismatch:{key}")
        elif isinstance(expected, dict):
            if "not" in expected and actual == expected["not"]:
                reasons.append(f"condition_excluded:{key}")
            elif "in" in expected and actual not in expected["in"]:
                reasons.append(f"condition_mismatch:{key}")
        elif actual != expected:
            reasons.append(f"condition_mismatch:{key}")
    return list(dict.fromkeys(reasons))


def filter_candidates(
    candidates: Sequence[KnowledgeCandidate],
    context: ProcessingContext,
) -> Tuple[List[KnowledgeCandidate], List[FilterDecision], List[ProcessingWarning]]:
    """深复制后过滤，不修改标准输入对象。"""
    accepted: List[KnowledgeCandidate] = []
    decisions: List[FilterDecision] = []
    warnings: List[ProcessingWarning] = []
    for candidate in candidates:
        reasons: List[str] = []
        warnings.extend(collect_applicability_warnings(
            candidate.applicability,
            knowledge_id=candidate.knowledge_id,
            source_index=candidate.source_index,
        ))
        # 缺少 ID 不影响内容处理，只在重排边界排除。
        # Adapter 和 rerank 都会对此产生明确告警。
        reasons.extend(evaluate_applicability(candidate.applicability, context))
        kept_atoms = []
        filtered_atom_count = 0
        for atom in candidate.atoms:
            warnings.extend(collect_applicability_warnings(
                atom.applicability,
                knowledge_id=candidate.knowledge_id,
                source_index=candidate.source_index,
                field_prefix=f"atoms[{atom.source_index}].applicability",
            ))
            atom_reasons = evaluate_applicability(atom.applicability, context)
            if atom_reasons:
                filtered_atom_count += 1
                warnings.append(ProcessingWarning(
                    code="atom_not_applicable",
                    message=f"原子已按适用性过滤: {', '.join(atom_reasons)}",
                    source_index=candidate.source_index,
                    knowledge_id=candidate.knowledge_id,
                    field="atoms",
                    details={"atom_id": atom.atom_id, "reasons": atom_reasons},
                ))
            else:
                kept_atoms.append(copy.deepcopy(atom))
        renderable_candidate = copy.deepcopy(candidate)
        renderable_candidate.atoms = kept_atoms
        if not has_renderable_candidate_content(renderable_candidate):
            reasons.append("empty_content")
        is_accepted = not reasons
        decisions.append(FilterDecision(
            accepted=is_accepted,
            reasons=reasons,
            knowledge_id=candidate.knowledge_id,
            source_index=candidate.source_index,
            kept_atom_count=len(kept_atoms),
            filtered_atom_count=filtered_atom_count,
        ))
        if is_accepted:
            copied = copy.deepcopy(candidate)
            copied.atoms = kept_atoms
            accepted.append(copied)
    return accepted, decisions, warnings


filter_knowledge_candidates = filter_candidates
