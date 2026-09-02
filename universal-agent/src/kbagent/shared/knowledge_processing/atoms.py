"""知识原子的例外、适用性、分组、排序与去重。"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Sequence, Tuple

from .annotations import filter_annotation_for_audience, strip_annotation_fields
from .applicability import evaluate_applicability
from .models import KnowledgeAtom, ProcessingContext, ProcessingWarning
from .richtext import render_richtext


_CONDITION_ALIASES = {
    "region_id": "region_id",
    "regionId": "region_id",
    "region_ids": "region_id",
    "regionIds": "region_id",
    "region_name": "region_name",
    "regionName": "region_name",
    "regions": "region_name",
    "region": "region_name",
    "province": "region_name",
    "channel_code": "channel_code",
    "channelCode": "channel_code",
    "channel_codes": "channel_code",
    "channelCodes": "channel_code",
    "channel": "channel",
    "channels": "channel",
    "customer_type": "customer_type",
    "customerType": "customer_type",
    "audience": "audience",
}
_CONDITION_CONTAINER_KEYS = ("when", "condition", "conditions")
_EXCLUDE_TYPES = {"exclude", "excluded", "remove", "not_applicable", "not-applicable"}


def parse_except_rule(
    rule: Any,
    warnings: List[ProcessingWarning] | None = None,
    atom_id: str | None = None,
) -> Any:
    """只负责解析 JSON 形式，不解释业务匹配或覆盖语义。"""
    if not isinstance(rule, str) or not rule.strip():
        return copy.deepcopy(rule)
    try:
        parsed = json.loads(rule)
    except json.JSONDecodeError as exc:
        if warnings is not None:
            warnings.append(ProcessingWarning(
                code="invalid_except_rules_json",
                message="except_rules JSON 解析失败，已保留原始内容",
                field="except_rules",
                details={"atom_id": atom_id, "error": str(exc)},
            ))
        return rule
    return parsed if isinstance(parsed, (dict, list)) else rule


def _context_value(context: ProcessingContext, canonical_key: str) -> Any:
    if canonical_key == "region_name":
        return context.region_name or (
            context.region if not context.region_id and not context.region_name else None
        )
    if canonical_key == "channel":
        return context.channel
    if hasattr(context, canonical_key):
        return getattr(context, canonical_key)
    return context.attributes.get(canonical_key)


def _condition_items(rule: Dict[str, Any]) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    nested = next((rule.get(key) for key in _CONDITION_CONTAINER_KEYS if key in rule), None)
    if isinstance(nested, dict):
        items.extend(
            (_CONDITION_ALIASES[key], expected)
            for key, expected in nested.items()
            if key in _CONDITION_ALIASES
        )
    for key, expected in rule.items():
        if key in _CONDITION_ALIASES:
            items.append((_CONDITION_ALIASES[key], expected))
    return items


def _matches_expected(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if "in" in expected:
            return actual in expected["in"]
        if "eq" in expected:
            return actual == expected["eq"]
        if "not" in expected:
            return actual != expected["not"]
        return False
    if isinstance(expected, (list, tuple, set)):
        return actual in expected
    return actual == expected


def match_except_conditions(rule: Any, context: ProcessingContext) -> bool:
    """只匹配约定的业务上下文字段；覆盖值和控制字段绝不参与匹配。"""
    if not isinstance(rule, dict) or rule.get("disabled") is True or rule.get("enabled") is False:
        return False
    conditions = _condition_items(rule)
    if not conditions:
        return False
    for canonical_key, expected in conditions:
        actual = _context_value(context, canonical_key)
        if actual is None or not _matches_expected(actual, expected):
            return False
    return True


def _is_explicit_exclusion(rule: Dict[str, Any]) -> bool:
    if rule.get("exclude") is True or rule.get("excluded") is True:
        return True
    rule_type = str(rule.get("type") or rule.get("action") or "").strip().casefold()
    return rule_type in _EXCLUDE_TYPES


def _has_override_result(rule: Dict[str, Any]) -> bool:
    return any(key in rule for key in ("content", "value", "annotation", "unit", "wkuntt"))


def apply_except_override(atom: KnowledgeAtom, rule: Dict[str, Any]) -> KnowledgeAtom | None:
    """应用命中规则的结果字段；仅显式排除规则返回 None。"""
    if _is_explicit_exclusion(rule):
        return None
    overridden = copy.deepcopy(atom)
    if "content" in rule:
        overridden.content = copy.deepcopy(rule["content"])
    elif "value" in rule:
        overridden.content = copy.deepcopy(rule["value"])
    if "annotation" in rule:
        overridden.annotation = copy.deepcopy(rule["annotation"])
    unit_key = "wkuntt" if "wkuntt" in rule else "unit" if "unit" in rule else None
    if unit_key:
        unit = copy.deepcopy(rule[unit_key])
        overridden.wkuntt = unit
        overridden.unit = unit
    return overridden


def _rule_items(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def except_rules_match(
    rules: Any,
    context: ProcessingContext,
    warnings: List[ProcessingWarning] | None = None,
    atom_id: str | None = None,
) -> bool:
    parsed = parse_except_rule(rules, warnings, atom_id)
    return any(match_except_conditions(rule, context) for rule in _rule_items(parsed))


def _dedupe_key(atom: KnowledgeAtom) -> str:
    def stable(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError, RecursionError):
            return str(value)

    # 指纹覆盖所有会改变最终 Markdown 的规范化字段。
    return "value:" + "\x1f".join((
        atom.group_id or atom.group,
        atom.param_name or atom.title,
        atom.param_type or "",
        str(atom.arrange_seq_number) if atom.arrange_seq_number is not None else "",
        render_richtext(atom.content),
        stable(atom.except_rules),
        stable(atom.annotation),
        atom.unit or "",
        atom.wkuntt or "",
    ))


def process_atoms(
    atoms: Sequence[KnowledgeAtom],
    context: ProcessingContext,
) -> Tuple[List[KnowledgeAtom], List[ProcessingWarning]]:
    warnings: List[ProcessingWarning] = []
    kept: List[KnowledgeAtom] = []
    seen = set()
    for atom in atoms:
        reasons = evaluate_applicability(atom.applicability, context)
        if reasons:
            warnings.append(ProcessingWarning(
                code="atom_not_applicable",
                message=f"原子不适用: {', '.join(reasons)}",
                field="atoms",
                details={"atom_id": atom.atom_id, "reasons": reasons},
            ))
            continue
        working = copy.deepcopy(atom)
        parsed_rules = parse_except_rule(atom.except_rules, warnings, atom.atom_id)
        matched_rule = next(
            (rule for rule in _rule_items(parsed_rules) if match_except_conditions(rule, context)),
            None,
        )
        if matched_rule is not None:
            overridden = apply_except_override(working, matched_rule)
            if overridden is None:
                warnings.append(ProcessingWarning(
                    code="atom_excepted",
                    message="原子命中显式排除规则，已排除",
                    field="except_rules",
                    details={"atom_id": atom.atom_id},
                ))
                continue
            if _has_override_result(matched_rule):
                working = overridden
                warnings.append(ProcessingWarning(
                    code="atom_except_overridden",
                    message="原子已应用命中的例外覆盖",
                    field="except_rules",
                    details={"atom_id": atom.atom_id},
                ))
        working.annotation = filter_annotation_for_audience(
            working.annotation, context, warnings, atom_id=working.atom_id,
            preserve_visibility=True,
        )
        working.except_rules = strip_annotation_fields(working.except_rules)
        working.metadata = strip_annotation_fields(working.metadata)
        working.raw = strip_annotation_fields(working.raw)
        key = _dedupe_key(working)
        if key in seen:
            warnings.append(ProcessingWarning(
                code="duplicate_atom",
                message="重复原子已去除",
                field="atoms",
                details={"atom_id": atom.atom_id},
            ))
            continue
        seen.add(key)
        kept.append(working)

    # group_id 仅用于归组，不参与组间字典序。组间按最小展示序号、再按首次出现排序；
    # 组内按展示序号稳定排序，缺少序号的原子保持输入相对顺序。
    grouped: Dict[str, List[Tuple[int, KnowledgeAtom]]] = {}
    for position, atom in enumerate(kept):
        grouped.setdefault(atom.group_id or atom.group, []).append((position, atom))

    def group_key(item: Tuple[str, List[Tuple[int, KnowledgeAtom]]]) -> Tuple[bool, int, int]:
        entries = item[1]
        sequences = [
            entry.arrange_seq_number for _, entry in entries
            if entry.arrange_seq_number is not None
        ]
        return (not sequences, min(sequences) if sequences else 0, entries[0][0])

    ordered: List[KnowledgeAtom] = []
    for _, entries in sorted(grouped.items(), key=group_key):
        entries.sort(key=lambda item: (
            item[1].arrange_seq_number is None,
            item[1].arrange_seq_number if item[1].arrange_seq_number is not None else 0,
            item[0],
        ))
        ordered.extend(atom for _, atom in entries)
    return ordered, warnings


def group_atoms(atoms: Sequence[KnowledgeAtom]) -> List[Tuple[str, List[KnowledgeAtom]]]:
    groups: Dict[str, List[KnowledgeAtom]] = {}
    for atom in atoms:
        groups.setdefault(atom.group_id or atom.group or "详情", []).append(atom)
    return list(groups.items())


def render_rule(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            if key in {"id", "uuid", "style", "url", "src", "metadata"}:
                continue
            rendered = render_rule(item)
            if rendered:
                pairs.append(f"{key}: {rendered}")
        return "；".join(pairs)
    if isinstance(value, (list, tuple)):
        return "；".join(filter(None, (render_rule(item) for item in value)))
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return render_rule(parsed)
    return render_richtext(value)
