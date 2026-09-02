"""外部候选数据到标准知识处理对象的唯一转换边界。"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    Applicability,
    KnowledgeAtom,
    KnowledgeCandidate,
    NormalizationResult,
    ProcessingContext,
    ProcessingWarning,
)


def _mapping(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    if isinstance(raw, Applicability):
        return copy.deepcopy(asdict(raw))
    raise TypeError(f"期望标准字典，实际为 {type(raw).__name__}")


def _warn(
    warnings: List[ProcessingWarning],
    code: str,
    message: str,
    source_index: Optional[int] = None,
    field: Optional[str] = None,
    knowledge_id: Optional[str] = None,
    **details: Any,
) -> None:
    warnings.append(ProcessingWarning(
        code=code,
        message=message,
        source_index=source_index,
        field=field,
        knowledge_id=knowledge_id,
        details=details,
    ))


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value).strip() or None


def _string_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    values: Iterable[Any]
    if isinstance(value, str):
        values = [v for v in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    result: List[str] = []
    for item in values:
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return result


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _iso_time_text(value: Any) -> Optional[str]:
    """保留 ISO 时间的时区信息，并将 Z 规范为显式 UTC 偏移。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _text(value)
    if not text:
        return None
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    # 纯日期必须保留 date 语义，结束日才能在判定层展开到当天末尾。
    if len(candidate) == 10:
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(candidate).isoformat()
    except ValueError:
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError:
            # Adapter 不丢弃无法识别的时间，判定层将按未知处理。
            return text


def _value_summary(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _normalize_applicability(
    raw: Any,
    warnings: List[ProcessingWarning],
    source_index: int,
    knowledge_id: Optional[str] = None,
    owner: Optional[Dict[str, Any]] = None,
) -> Applicability:
    raw_snapshot = raw.raw if isinstance(raw, Applicability) else raw
    try:
        data = _mapping(raw) if raw is not None else {}
    except Exception as exc:  # noqa: BLE001 - Adapter 必须隔离单条异常
        _warn(warnings, "invalid_applicability", f"适用性字段无法转换: {exc}", source_index)
        data = {}
    if isinstance(raw, Applicability):
        # 内部模型仍保留早期字段；只在已实例化对象边界同步，不对外部字典猜测。
        data["effective_start"] = raw.effective_start or raw.start_at
        data["effective_end"] = raw.effective_end or raw.end_at

    owner = owner or {}

    def value(field: str) -> Any:
        selected = data.get(field)
        if selected in (None, "", [], {}):
            selected = owner.get(field)
        return selected

    def normalized_list(field: str) -> List[str]:
        raw_value = value(field)
        invalid = isinstance(raw_value, dict) or (
            isinstance(raw_value, (list, tuple, set))
            and any(isinstance(item, (dict, list, tuple, set)) for item in raw_value)
        )
        if invalid:
            _warn(
                warnings,
                "invalid_applicability_field",
                "适用性列表字段格式异常，已按无法判定处理并保留候选",
                source_index,
                field,
                knowledge_id,
                warning_code="invalid_applicability_field",
                raw_value=_value_summary(raw_value),
            )
            return []
        return _string_list(raw_value)

    conditions = value("conditions")
    if not isinstance(conditions, dict):
        conditions = {}
    conditions = copy.deepcopy(conditions)
    effective_start = _iso_time_text(value("effective_start"))
    effective_end = _iso_time_text(value("effective_end"))
    region_ids = normalized_list("region_ids")
    channel_codes = normalized_list("channel_codes")
    return Applicability(
        status=_text(value("status")),
        effective_start=effective_start,
        effective_end=effective_end,
        region_ids=region_ids,
        channel_codes=channel_codes,
        start_at=effective_start,
        end_at=effective_end,
        regions=normalized_list("regions"),
        channels=normalized_list("channels"),
        excluded_regions=normalized_list("excluded_regions"),
        excluded_channels=normalized_list("excluded_channels"),
        conditions=conditions,
        raw=copy.deepcopy(raw_snapshot),
    )


def normalize_candidate_applicability(
    candidate: KnowledgeCandidate,
    warnings: Optional[List[ProcessingWarning]] = None,
) -> KnowledgeCandidate:
    """将已实例化候选的适用性同步到内部统一字段。

    顶层回退只用于正式调用链传入的 KnowledgeCandidate 内部模型；
    外部字典只读取嵌套 applicability，不会猜测顶层历史字段。
    """
    target_warnings = warnings if warnings is not None else []
    normalized = copy.deepcopy(candidate)
    owner = {
        "status": normalized.status,
        "effective_start": normalized.start_at,
        "effective_end": normalized.end_at,
        "region_ids": normalized.region_ids,
        "regions": normalized.regions,
        "channel_codes": normalized.channel_codes,
        "channels": normalized.channels,
    }
    normalized.applicability = _normalize_applicability(
        normalized.applicability,
        target_warnings,
        normalized.source_index,
        normalized.knowledge_id,
        owner=owner,
    )
    applicability = normalized.applicability
    normalized.status = applicability.status
    normalized.start_at = applicability.start_at
    normalized.end_at = applicability.end_at
    normalized.region_ids = list(applicability.region_ids)
    normalized.regions = list(applicability.regions)
    normalized.channel_codes = list(applicability.channel_codes)
    normalized.channels = list(applicability.channels)
    return normalized


def normalize_processing_context(raw: Any) -> ProcessingContext:
    if isinstance(raw, ProcessingContext):
        context = copy.deepcopy(raw)
        context.region = context.region or context.region_name or context.region_id
        context.channel = context.channel or context.channel_code
        context.request_time = _iso_time_text(context.request_time)
        context.as_of = context.request_time or _iso_time_text(context.as_of)
        context.audience_defaulted = context.audience_defaulted or not bool(
            str(context.audience or "").strip()
        )
        context.audience = context.audience or "agent"
        return context
    data = copy.deepcopy(raw) if isinstance(raw, dict) else {}
    region_id = _text(data.get("region_id"))
    region_name = _text(data.get("region_name"))
    channel_code = _text(data.get("channel_code"))
    request_time = _iso_time_text(data.get("request_time"))
    audience = _text(data.get("audience"))
    return ProcessingContext(
        region_id=region_id,
        region_name=region_name,
        channel_code=channel_code,
        request_time=request_time,
        audience=audience or "agent",
        audience_defaulted=not bool(audience),
        region=region_name or region_id,
        channel=channel_code,
        as_of=request_time,
        customer_type=_text(data.get("customer_type")),
        attributes={},
        raw=copy.deepcopy(raw),
    )


def _normalize_atom(
    raw: Any,
    source_index: int,
    warnings: List[ProcessingWarning],
) -> KnowledgeAtom:
    if isinstance(raw, KnowledgeAtom):
        atom = copy.deepcopy(raw)
        atom.source_index = source_index
        atom.param_name = atom.param_name or atom.title
        atom.title = atom.param_name
        atom.group_id = atom.group_id or atom.group
        atom.group = atom.group_id
        atom.arrange_seq_number = (
            atom.arrange_seq_number if atom.arrange_seq_number is not None else atom.order
        )
        atom.order = atom.arrange_seq_number if atom.arrange_seq_number is not None else source_index
        atom.wkuntt = atom.wkuntt or atom.unit
        atom.unit = atom.wkuntt
        return atom
    data = _mapping(raw)
    atom_id = _text(data.get("atom_id"))
    param_name = _text(data.get("param_name"))
    if not param_name:
        param_name = "未命名字段"
        _warn(
            warnings, "missing_param_name", "缺少原子字段名，已生成占位名称",
            source_index, "param_name",
        )
    param_type = _text(data.get("param_type"))
    content = data.get("content")
    group_id = _text(data.get("group_id")) or ""
    arrange_raw = data.get("arrange_seq_number")
    arrange_seq_number = _optional_int(arrange_raw)
    if arrange_raw not in (None, "") and arrange_seq_number is None:
        _warn(
            warnings, "invalid_arrange_seq_number",
            "非法原子排序值已按原始相对顺序排后", source_index,
            "arrange_seq_number",
        )
    wkuntt = _text(data.get("wkuntt"))
    except_rules = data.get("except_rules")
    annotation = data.get("annotation")
    applicability_raw = data.get("applicability")
    known = {
        "atom_id", "group_id", "param_name", "param_type", "content", "except_rules",
        "annotation", "arrange_seq_number", "wkuntt", "applicability",
    }
    return KnowledgeAtom(
        atom_id=atom_id,
        param_name=param_name,
        param_type=param_type,
        group_id=group_id,
        arrange_seq_number=arrange_seq_number,
        wkuntt=wkuntt,
        title=param_name,
        content=copy.deepcopy(content if content is not None else ""),
        group=group_id,
        order=arrange_seq_number if arrange_seq_number is not None else source_index,
        unit=wkuntt,
        except_rules=copy.deepcopy(except_rules),
        annotation=copy.deepcopy(annotation),
        applicability=_normalize_applicability(applicability_raw, warnings, source_index),
        source_index=source_index,
        metadata={k: copy.deepcopy(v) for k, v in data.items() if k not in known},
        raw=copy.deepcopy(raw),
    )


def normalize_knowledge_atom(raw: Any, source_index: int = 0) -> KnowledgeAtom:
    warnings: List[ProcessingWarning] = []
    return _normalize_atom(raw, source_index, warnings)


def _normalize_candidate(
    raw: Any,
    source_index: int,
    warnings: List[ProcessingWarning],
) -> KnowledgeCandidate:
    if isinstance(raw, KnowledgeCandidate):
        candidate = copy.deepcopy(raw)
        candidate.source_index = source_index
        if not candidate.retrieval_rank:
            candidate.retrieval_rank = source_index + 1
        if isinstance(candidate.atoms, (list, tuple)):
            normalized_atoms: List[KnowledgeAtom] = []
            for atom_index, atom in enumerate(candidate.atoms):
                try:
                    normalized_atoms.append(_normalize_atom(atom, atom_index, warnings))
                except Exception as exc:  # noqa: BLE001 - 与字典输入保持单原子异常隔离
                    _warn(
                        warnings, "atom_conversion_error", f"原子转换失败，已跳过: {exc}",
                        source_index, "atoms", candidate.knowledge_id, atom_index=atom_index,
                    )
            candidate.atoms = normalized_atoms
        else:
            candidate.atoms = []
            _warn(
                warnings, "invalid_atoms", "atoms 非列表，已规范为空列表",
                source_index, "atoms", candidate.knowledge_id,
            )
        return normalize_candidate_applicability(candidate, warnings)
    data = _mapping(raw)
    knowledge_id = _text(data.get("knowledge_id"))
    name = _text(data.get("knowledge_name"))
    rank_raw = data.get("retrieval_rank")
    rank = _int(rank_raw, source_index + 1)
    if rank_raw is None or rank <= 0:
        rank = source_index + 1
        if rank_raw is None:
            _warn(
                warnings, "missing_retrieval_rank", "缺少排名，已按输入顺序生成",
                source_index, "retrieval_rank",
            )
        else:
            _warn(warnings, "invalid_retrieval_rank", "非法排名已按输入顺序生成", source_index, "retrieval_rank")
    if not name:
        name = f"未命名知识-{source_index + 1:03d}"
        _warn(warnings, "missing_name", "缺少知识名称，已生成稳定占位标题", source_index, "name", knowledge_id)
    if not knowledge_id:
        _warn(warnings, "missing_knowledge_id", "缺少知识 ID，该候选不会进入重排", source_index, "knowledge_id")

    retrieval_score_raw = data.get("retrieval_score")
    retrieval_score = _optional_float(retrieval_score_raw)
    if retrieval_score_raw not in (None, "") and retrieval_score is None:
        _warn(warnings, "invalid_retrieval_score", "无法解析 retrieval_score，已保留空值", source_index, "retrieval_score")
    matched_atom_ids = _string_list(data.get("matched_atom_ids"))
    source_routes = _string_list(data.get("source_routes"))
    knowledge_type = _text(data.get("knowledge_type"))
    template_id = _text(data.get("template_id"))

    atoms_raw = data.get("atoms")
    atoms: List[KnowledgeAtom] = []
    if atoms_raw is None:
        atoms_raw = []
    if isinstance(atoms_raw, (list, tuple)):
        for atom_index, atom_raw in enumerate(atoms_raw):
            try:
                atoms.append(_normalize_atom(atom_raw, atom_index, warnings))
            except Exception as exc:  # noqa: BLE001
                _warn(
                    warnings, "atom_conversion_error", f"原子转换失败，已跳过: {exc}",
                    source_index, "atoms", knowledge_id, atom_index=atom_index,
                )
    else:
        _warn(warnings, "invalid_atoms", "atoms 非列表，已规范为空列表", source_index, "atoms", knowledge_id)

    content = data.get("content")
    applicability_raw = data.get("applicability")
    applicability = _normalize_applicability(
        applicability_raw, warnings, source_index, knowledge_id
    )
    known = {
        "knowledge_id", "knowledge_name", "content", "retrieval_rank", "retrieval_score",
        "matched_atom_ids", "source_routes", "knowledge_type", "template_id",
        "applicability", "atoms",
    }
    candidate = KnowledgeCandidate(
        knowledge_id=knowledge_id,
        name=name,
        content=copy.deepcopy(content if content is not None else ""),
        atoms=atoms,
        retrieval_rank=rank,
        retrieval_score=retrieval_score,
        matched_atom_ids=matched_atom_ids,
        source_routes=source_routes,
        knowledge_type=knowledge_type,
        template_id=template_id,
        status=applicability.status,
        start_at=applicability.start_at,
        end_at=applicability.end_at,
        regions=list(applicability.regions),
        channels=list(applicability.channels),
        region_ids=list(applicability.region_ids),
        channel_codes=list(applicability.channel_codes),
        applicability=applicability,
        source_index=source_index,
        metadata={k: copy.deepcopy(v) for k, v in data.items() if k not in known},
        raw=copy.deepcopy(raw),
    )
    return normalize_candidate_applicability(candidate, warnings)


def normalize_knowledge_candidate(raw: Any, source_index: int = 0) -> KnowledgeCandidate:
    warnings: List[ProcessingWarning] = []
    return _normalize_candidate(raw, source_index, warnings)


def _candidate_sequence(raw: Any) -> tuple[List[Any], Optional[str]]:
    if raw is None:
        return [], None
    if isinstance(raw, (list, tuple)):
        return list(raw), None
    return [], "invalid"


def normalize_knowledge_candidates(raw: Any) -> NormalizationResult:
    """批量规范化，任何单条失败都不会影响其他候选。"""
    warnings: List[ProcessingWarning] = []
    items, wrapper = _candidate_sequence(raw)
    if wrapper == "invalid":
        _warn(warnings, "invalid_candidates", "候选集必须是标准候选列表")
        return NormalizationResult([], warnings)
    candidates: List[KnowledgeCandidate] = []
    for source_index, item in enumerate(items):
        try:
            candidates.append(_normalize_candidate(item, source_index, warnings))
        except Exception as exc:  # noqa: BLE001 - 核心契约：单条异常隔离
            _warn(
                warnings,
                "candidate_conversion_error",
                f"候选转换失败，已跳过: {exc}",
                source_index,
            )
    return NormalizationResult(candidates, warnings)
