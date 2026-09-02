"""Annotation 的结构化可见性规范化与受众过滤。"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Sequence

from .models import ProcessingContext, ProcessingWarning

_ALIASES = {
    "visibleTo": "visible_to",
    "audiences": "visible_to",
    "isVisible": "visible",
    "is_visible": "visible",
    "isPrivate": "private",
    "is_private": "private",
    "isPersonal": "personal",
    "is_personal": "personal",
    "isInternal": "internal",
    "is_internal": "internal",
    "agentOnly": "agent_only",
    "customerVisible": "customer_visible",
    "agentVisible": "agent_visible",
}
_VISIBILITY_KEYS = {
    "audience", "visibility", "visible_to", "visible", "private", "personal", "internal",
    "agent_only", "customer_visible", "agent_visible",
}
_ANNOTATION_FIELD_KEYS = {"annotation", "annotations", "note", "remark"}
_PAYLOAD_KEYS = ("content", "text", "value", "description", "message")
_PUBLIC = {"public", "customer", "customers", "all", "external", "公开", "客户", "面客"}
_AGENT = {"agent", "agents", "internal", "staff", "坐席", "内部", "员工"}
_HIDDEN = {
    "private", "personal", "hidden", "invisible", "none", "deny", "不可见", "个人", "私有",
}


def _canonical_mapping(value: Dict[str, Any]) -> Dict[str, Any]:
    canonical = copy.deepcopy(value)
    for alias, target in _ALIASES.items():
        if target not in canonical and alias in canonical:
            canonical[target] = canonical[alias]
        canonical.pop(alias, None)
    return canonical


def _tokens(value: Any) -> List[str]:
    values: Sequence[Any]
    if isinstance(value, str):
        values = [part for part in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif value is None:
        values = []
    else:
        values = [value]
    return [str(item).strip().casefold() for item in values if str(item).strip()]


def _visibility(mapping: Dict[str, Any], audience: str) -> tuple[bool, str, bool]:
    """返回 (是否可见, 原因, 是否存在结构化可见性标记)。"""
    marked = any(key in mapping for key in _VISIBILITY_KEYS)
    if mapping.get("private") is True or mapping.get("personal") is True:
        return False, "private_or_personal", marked
    if mapping.get("visible") is False:
        return False, "explicitly_hidden", marked

    tokens = _tokens(mapping.get("visible_to"))
    tokens.extend(_tokens(mapping.get("audience")))
    tokens.extend(_tokens(mapping.get("visibility")))
    token_set = set(tokens)
    if token_set & _HIDDEN:
        return False, "explicitly_hidden", marked
    if mapping.get("internal") is True or mapping.get("agent_only") is True:
        token_set.add("agent")
    if mapping.get("customer_visible") is True:
        token_set.add("customer")
    if mapping.get("agent_visible") is True:
        token_set.add("agent")

    if token_set:
        if audience == "customer":
            allowed = bool(token_set & _PUBLIC)
        else:
            allowed = bool(token_set & (_PUBLIC | _AGENT))
        return allowed, "allowed" if allowed else "audience_not_allowed", marked
    if mapping.get("visible") is True:
        return True, "explicitly_visible", marked
    # 文档默认运行模式是 agent；面客时无标记 annotation 采用安全隐藏。
    return audience == "agent", "visibility_unspecified", marked


def _payload(mapping: Dict[str, Any]) -> Any:
    for key in _PAYLOAD_KEYS:
        if key in mapping:
            return copy.deepcopy(mapping[key])
    remaining = {
        key: copy.deepcopy(value)
        for key, value in mapping.items()
        if key not in _VISIBILITY_KEYS
    }
    return remaining or None


def filter_annotation_for_audience(
    annotation: Any,
    context: ProcessingContext,
    warnings: List[ProcessingWarning],
    *,
    atom_id: str | None = None,
    preserve_visibility: bool = False,
) -> Any:
    """返回可进入 Markdown/下游对象的安全 annotation 副本。"""
    if annotation in (None, "", [], {}):
        return None
    audience = str(context.audience or "").strip().casefold()
    if context.audience_defaulted and not any(
        warning.code == "annotation_audience_defaulted" for warning in warnings
    ):
        warnings.append(ProcessingWarning(
            code="annotation_audience_defaulted",
            message="未提供 annotation 受众，已按项目默认坐席模式处理",
            field="annotation",
            details={"atom_id": atom_id, "audience": "agent"},
        ))
    if audience not in {"agent", "customer"}:
        audience = "agent"
        warnings.append(ProcessingWarning(
            code="annotation_audience_defaulted",
            message="缺少或无法识别 annotation 受众，已按项目默认坐席模式处理",
            field="annotation",
            details={"atom_id": atom_id, "audience": "agent"},
        ))
    if isinstance(annotation, (list, tuple)):
        visible = [
            filtered
            for item in annotation
            if (filtered := filter_annotation_for_audience(
                item, context, warnings, atom_id=atom_id,
                preserve_visibility=preserve_visibility,
            )) not in (None, "", [], {})
        ]
        return visible or None
    if isinstance(annotation, str) and annotation.strip().startswith(("{", "[")):
        try:
            parsed = json.loads(annotation)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return filter_annotation_for_audience(
                parsed, context, warnings, atom_id=atom_id,
                preserve_visibility=preserve_visibility,
            )
    if not isinstance(annotation, dict):
        if audience == "agent":
            return copy.deepcopy(annotation)
        warnings.append(ProcessingWarning(
            code="annotation_filtered",
            message="无结构化可见性标记的 annotation 已从面客内容中过滤",
            field="annotation",
            details={"atom_id": atom_id, "audience": audience, "reason": "visibility_unspecified"},
        ))
        return None

    mapping = _canonical_mapping(annotation)
    visible, reason, marked = _visibility(mapping, audience)
    if not visible:
        warnings.append(ProcessingWarning(
            code="annotation_filtered",
            message="annotation 因受众或可见性限制已过滤",
            field="annotation",
            details={"atom_id": atom_id, "audience": audience, "reason": reason},
        ))
        return None
    payload = _payload(mapping)
    if not preserve_visibility or payload in (None, "", [], {}):
        return payload
    tokens = set(
        _tokens(mapping.get("visible_to"))
        + _tokens(mapping.get("audience"))
        + _tokens(mapping.get("visibility"))
    )
    normalized_visibility = (
        "public" if audience == "customer" or tokens & _PUBLIC else "agent"
    )
    return {"visibility": normalized_visibility, "content": payload}


def strip_annotation_fields(value: Any) -> Any:
    """从 raw/except 副本中移除 annotation 及其别名，避免旁路泄露。"""
    if isinstance(value, dict):
        return {
            key: strip_annotation_fields(item)
            for key, item in value.items()
            if key not in _ANNOTATION_FIELD_KEYS
        }
    if isinstance(value, list):
        return [strip_annotation_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_annotation_fields(item) for item in value)
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            return strip_annotation_fields(parsed)
    return copy.deepcopy(value)
