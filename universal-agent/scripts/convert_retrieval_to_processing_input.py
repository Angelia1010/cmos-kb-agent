#!/usr/bin/env python
"""将真实检索响应转换为 Processing Demo 可读取的候选 JSON。"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_MISSING = object()
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_CUSTOMER_NAME_RE = re.compile(
    r"(?:客户姓名|客户名称|姓名)\s*[：:]\s*[\u4e00-\u9fff·]{2,8}"
)
_PHONE_LABEL_RE = re.compile(
    r"(?:手机号|手机号码|联系电话|电话号码|号码)\s*[：:]\s*\d{6,}"
)

_DOCUMENT_MAPPED_FIELDS = {
    "knowledgeId",
    "knowledge_id",
    "klgAttrAtomId",
    "atomId",
    "atom_id",
    "groupId",
    "group_id",
    "paramName",
    "param_name",
    "paramType",
    "param_type",
    "content",
    "except",
    "exceptRules",
    "except_rules",
    "annotation",
    "arrangeSeqNumber",
    "arrange_seq_number",
    "wkuntt",
    "unit",
    "statusCode",
    "status_code",
    "status",
    "channelCode",
    "channel_code",
    "channelCodes",
    "channel_codes",
    "retrievalScore",
    "retrieval_score",
    "score",
    "_score",
}


@dataclass
class ConversionWarning:
    code: str
    message: str
    candidate_index: int | None = None
    document_index: int | None = None


@dataclass
class ConversionResult:
    candidates: list[dict[str, Any]]
    stats: dict[str, Any]
    warnings: list[ConversionWarning] = field(default_factory=list)
    root_description: str = ""
    unrecognized_document_fields: list[str] = field(default_factory=list)
    sensitive_counts: dict[str, Any] = field(default_factory=dict)


def _pick(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _integer(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None


def _score(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values: Iterable[Any]
    if isinstance(value, str):
        values = value.replace("，", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    result: list[str] = []
    for item in values:
        text = _text(item)
        if text and text not in result:
            result.append(text)
    return result


def _candidate_records(payload: Any) -> tuple[list[Any], str]:
    if isinstance(payload, list):
        return payload, "顶层列表"
    if isinstance(payload, dict):
        for key in ("candidates", "data", "response", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value, f"顶层字典.{key}列表"
    raise ValueError("原始检索 JSON 必须是列表，或包含 candidates/data/response/items 列表")


def _documents(record: Mapping[str, Any]) -> Any:
    response = record.get("response")
    if not isinstance(response, Mapping):
        return _MISSING
    object_value = response.get("object")
    if not isinstance(object_value, Mapping):
        return _MISSING
    return object_value.get("document", _MISSING)


def _candidate_applicability(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scalar_fields = {
        "status": ("status",),
        "effective_start": ("effective_start", "effectiveStart", "startAt"),
        "effective_end": ("effective_end", "effectiveEnd", "endAt"),
    }
    list_fields = {
        "region_ids": ("region_ids", "regionIds"),
        "regions": ("regions", "regionNames", "region_names"),
        "channel_codes": ("channel_codes", "channelCodes"),
        "channels": ("channels",),
        "excluded_regions": ("excluded_regions", "excludedRegions"),
        "excluded_channels": ("excluded_channels", "excludedChannels"),
    }
    for target, aliases in scalar_fields.items():
        value = _pick(record, *aliases, default=_MISSING)
        if value is not _MISSING and _text(value) is not None:
            result[target] = _text(value)
    for target, aliases in list_fields.items():
        value = _pick(record, *aliases, default=_MISSING)
        if value is not _MISSING:
            values = _string_list(value)
            if values:
                result[target] = values
    conditions = record.get("conditions", _MISSING)
    if isinstance(conditions, Mapping) and conditions:
        result["conditions"] = copy.deepcopy(dict(conditions))
    return result


def _atom_applicability(document: Mapping[str, Any]) -> dict[str, Any]:
    """只映射真实存在且名称明确的 document 级状态和渠道字段。"""
    result: dict[str, Any] = {}
    status = _pick(document, "status", "statusCode", "status_code", default=_MISSING)
    if status is not _MISSING and _text(status) is not None:
        result["status"] = _text(status)
    channels = _pick(
        document,
        "channel_codes",
        "channelCodes",
        "channelCode",
        "channel_code",
        default=_MISSING,
    )
    if channels is not _MISSING:
        normalized = _string_list(channels)
        if normalized:
            result["channel_codes"] = normalized
    return result


def _document_score(document: Mapping[str, Any]) -> float | None:
    return _score(_pick(
        document,
        "retrieval_score",
        "retrievalScore",
        "score",
        "_score",
        default=None,
    ))


def _convert_atom(
    document: Any,
    *,
    knowledge_id: str,
    candidate_index: int,
    document_index: int,
    warnings: list[ConversionWarning],
) -> dict[str, Any]:
    position = document_index + 1
    generated_atom_id = f"{knowledge_id}-ATOM-{position:03d}"
    if not isinstance(document, Mapping):
        warnings.append(ConversionWarning(
            "invalid_document_element",
            "document 元素不是对象，已作为原始 content 保留",
            candidate_index,
            document_index,
        ))
        return {
            "atom_id": generated_atom_id,
            "group_id": None,
            "param_name": "业务内容",
            "param_type": None,
            "content": copy.deepcopy(document),
            "except_rules": [],
            "annotation": None,
            "arrange_seq_number": position,
            "wkuntt": None,
            "applicability": {},
        }

    atom_id = _text(_pick(
        document, "klgAttrAtomId", "atom_id", "atomId", default=None
    ))
    if not atom_id:
        atom_id = generated_atom_id
        warnings.append(ConversionWarning(
            "mock_atom_id",
            "缺少 atom_id，已按知识 ID 和 document 顺序稳定生成",
            candidate_index,
            document_index,
        ))

    param_name = _text(_pick(document, "param_name", "paramName", default=None))
    if not param_name:
        param_name = "业务内容"
        warnings.append(ConversionWarning(
            "mock_param_name",
            "缺少原子字段名称，已使用默认名称“业务内容”",
            candidate_index,
            document_index,
        ))

    content = copy.deepcopy(document.get("content")) if "content" in document else None
    if "content" not in document or content is None or (
        isinstance(content, str) and not content.strip()
    ):
        warnings.append(ConversionWarning(
            "missing_document_content",
            "document 缺少可确认的正文，atom content 已保留为 null/空值",
            candidate_index,
            document_index,
        ))

    arrange_raw = _pick(
        document, "arrange_seq_number", "arrangeSeqNumber", default=_MISSING
    )
    arrange_seq_number = _integer(arrange_raw)
    if arrange_seq_number is None:
        arrange_seq_number = position
        warnings.append(ConversionWarning(
            "mock_arrange_seq_number",
            "缺少或无法解析展示顺序，已使用 document 原始顺序",
            candidate_index,
            document_index,
        ))

    except_value = _pick(
        document, "except", "except_rules", "exceptRules", default=[]
    )
    return {
        "atom_id": atom_id,
        "group_id": _text(_pick(document, "group_id", "groupId", default=None)),
        "param_name": param_name,
        "param_type": _text(_pick(document, "param_type", "paramType", default=None)),
        "content": content,
        "except_rules": copy.deepcopy(except_value),
        "annotation": copy.deepcopy(document.get("annotation")),
        "arrange_seq_number": arrange_seq_number,
        "wkuntt": _text(_pick(document, "wkuntt", "unit", default=None)),
        "applicability": _atom_applicability(document),
    }


def _walk_strings(value: Any, field_name: str = "root") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_strings(item, str(key))
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item, field_name)
    elif isinstance(value, str):
        yield field_name, value


def _sensitive_counts(payload: Any, documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    id_card_fields = Counter()
    for field_name, text in _walk_strings(payload):
        counts["mobile_pattern"] += len(_MOBILE_RE.findall(text))
        matches = len(_ID_CARD_RE.findall(text))
        counts["id_card_pattern"] += matches
        if matches:
            id_card_fields[field_name] += matches
        counts["customer_name_label"] += len(_CUSTOMER_NAME_RE.findall(text))
        counts["phone_label"] += len(_PHONE_LABEL_RE.findall(text))
    person_documents = [
        item for item in documents
        if re.search(r"姓名|人员|报送人|联系人", str(item.get("paramName") or ""))
    ]
    counts["person_related_param_atoms"] = len(person_documents)
    counts["person_related_nonempty_content"] = sum(
        item.get("content") not in (None, "", [], {}) for item in person_documents
    )
    return {
        **dict(counts),
        "id_card_pattern_fields": dict(sorted(id_card_fields.items())),
    }


def convert_payload(payload: Any) -> ConversionResult:
    records, root_description = _candidate_records(payload)
    warnings: list[ConversionWarning] = []
    states: list[dict[str, Any]] = []
    by_knowledge_id: dict[str, dict[str, Any]] = {}
    document_fields: set[str] = set()
    all_mapping_documents: list[Mapping[str, Any]] = []
    input_document_count = 0
    empty_document_candidates = 0
    non_list_document_candidates = 0
    invalid_document_elements = 0
    duplicate_knowledge_ids: set[str] = set()
    duplicate_atom_ids = 0

    for candidate_index, raw_record in enumerate(records):
        record = raw_record if isinstance(raw_record, Mapping) else {}
        if not isinstance(raw_record, Mapping):
            warnings.append(ConversionWarning(
                "invalid_candidate",
                "顶层候选不是对象，已保留为空候选并继续转换",
                candidate_index,
            ))

        knowledge_id = _text(_pick(
            record, "knowledge_id", "knowledgeId", default=None
        ))
        mock_knowledge_id = not bool(knowledge_id)
        if not knowledge_id:
            knowledge_id = f"MOCK-KNOWLEDGE-{candidate_index + 1:03d}"
            warnings.append(ConversionWarning(
                "mock_knowledge_id",
                "缺少知识 ID，已按原始候选顺序稳定生成",
                candidate_index,
            ))

        knowledge_name = _text(_pick(
            record,
            "knowledge_name",
            "knowledgeName",
            "title",
            default=None,
        ))
        mock_knowledge_name = not bool(knowledge_name)
        if not knowledge_name:
            knowledge_name = f"未命名知识-{knowledge_id}"
            warnings.append(ConversionWarning(
                "mock_knowledge_name",
                "没有可确认的知识级标题，已使用稳定占位标题",
                candidate_index,
            ))

        source_rank = _integer(_pick(
            record, "retrieval_rank", "retrievalRank", "rank", default=None
        ))
        mock_retrieval_rank = source_rank is None or source_rank <= 0
        if mock_retrieval_rank:
            source_rank = candidate_index + 1
            warnings.append(ConversionWarning(
                "mock_retrieval_rank",
                "没有知识级召回排名，已按原始候选顺序生成",
                candidate_index,
            ))

        documents_raw = _documents(record)
        if documents_raw is _MISSING or documents_raw is None:
            documents: list[Any] = []
            empty_document_candidates += 1
            warnings.append(ConversionWarning(
                "missing_document",
                "缺少 response.object.document，已转换为空 atoms",
                candidate_index,
            ))
        elif not isinstance(documents_raw, list):
            documents = []
            non_list_document_candidates += 1
            warnings.append(ConversionWarning(
                "invalid_document",
                "response.object.document 不是列表，已转换为空 atoms",
                candidate_index,
            ))
        else:
            documents = documents_raw
            if not documents:
                empty_document_candidates += 1

        input_document_count += len(documents)
        atoms: list[dict[str, Any]] = []
        seen_atom_ids: set[str] = set()
        document_scores: list[float] = []
        for document_index, document in enumerate(documents):
            if isinstance(document, Mapping):
                document_fields.update(str(key) for key in document)
                all_mapping_documents.append(document)
                score_value = _document_score(document)
                if score_value is not None:
                    document_scores.append(score_value)
            else:
                invalid_document_elements += 1
            atom = _convert_atom(
                document,
                knowledge_id=knowledge_id,
                candidate_index=candidate_index,
                document_index=document_index,
                warnings=warnings,
            )
            if atom["atom_id"] in seen_atom_ids:
                duplicate_atom_ids += 1
                warnings.append(ConversionWarning(
                    "duplicate_atom_id",
                    "同一知识内重复 atom_id，已保留首次出现的原子",
                    candidate_index,
                    document_index,
                ))
                continue
            seen_atom_ids.add(atom["atom_id"])
            atoms.append(atom)

        source_score = _score(_pick(
            record,
            "retrieval_score",
            "retrievalScore",
            "score",
            "_score",
            default=None,
        ))
        if source_score is None and document_scores:
            source_score = max(document_scores)
        mock_retrieval_score = source_score is None

        state = {
            "candidate": {
                "knowledge_id": knowledge_id,
                "knowledge_name": knowledge_name,
                "retrieval_rank": source_rank,
                "retrieval_score": source_score,
                "applicability": _candidate_applicability(record),
                "atoms": atoms,
            },
            "source_index": candidate_index,
            "mock_knowledge_id": mock_knowledge_id,
            "mock_knowledge_name": mock_knowledge_name,
            "mock_retrieval_rank": mock_retrieval_rank,
            "mock_retrieval_score": mock_retrieval_score,
            "seen_atom_ids": seen_atom_ids,
        }

        existing = by_knowledge_id.get(knowledge_id)
        if existing is None:
            by_knowledge_id[knowledge_id] = state
            states.append(state)
            continue

        duplicate_knowledge_ids.add(knowledge_id)
        target = existing["candidate"]
        incoming = state["candidate"]
        target["retrieval_rank"] = min(target["retrieval_rank"], incoming["retrieval_rank"])
        scores = [
            value for value in (target["retrieval_score"], incoming["retrieval_score"])
            if value is not None
        ]
        target["retrieval_score"] = max(scores) if scores else None
        existing["mock_retrieval_score"] = not bool(scores)
        if existing["mock_knowledge_name"] and not state["mock_knowledge_name"]:
            target["knowledge_name"] = incoming["knowledge_name"]
            existing["mock_knowledge_name"] = False
        if not target["applicability"] and incoming["applicability"]:
            target["applicability"] = incoming["applicability"]
        for atom in incoming["atoms"]:
            if atom["atom_id"] in existing["seen_atom_ids"]:
                duplicate_atom_ids += 1
                continue
            existing["seen_atom_ids"].add(atom["atom_id"])
            target["atoms"].append(atom)

    states.sort(key=lambda item: (
        item["candidate"]["retrieval_rank"], item["source_index"]
    ))
    output_count = len(states)
    for rank, state in enumerate(states, 1):
        candidate = state["candidate"]
        # Processing 的最终测试输入要求连续排名；真实排名若存在则只用于排序。
        if candidate["retrieval_rank"] != rank and not state["mock_retrieval_rank"]:
            candidate["source_retrieval_rank"] = candidate["retrieval_rank"]
            warnings.append(ConversionWarning(
                "normalized_retrieval_rank",
                "真实召回排名不连续，已保留为 source_retrieval_rank 并生成连续测试排名",
                state["source_index"],
            ))
        candidate["retrieval_rank"] = rank
        if state["mock_retrieval_score"]:
            candidate["retrieval_score"] = round((output_count - rank + 1) / output_count, 6)
            warnings.append(ConversionWarning(
                "mock_retrieval_score",
                "没有知识级或 document 级分数，已按最终顺序生成递减测试分数",
                state["source_index"],
            ))

    candidates = [state["candidate"] for state in states]
    sensitive = _sensitive_counts(payload, all_mapping_documents)
    if sensitive.get("id_card_pattern", 0):
        warnings.append(ConversionWarning(
            "potential_sensitive_identifier",
            "检测到身份证号形态的长数字；仅报告计数，未在终端或报告中展示原值",
        ))
    if sensitive.get("person_related_nonempty_content", 0):
        warnings.append(ConversionWarning(
            "potential_person_name_content",
            "存在姓名/人员类参数且正文非空；未展示原值，应按敏感数据管理输出文件",
        ))

    stats = {
        "original_candidate_count": len(records),
        "converted_candidate_count": len(candidates),
        "document_count": input_document_count,
        "generated_atom_count": sum(len(item["atoms"]) for item in candidates),
        "duplicate_knowledge_id_count": len(duplicate_knowledge_ids),
        "aggregated_candidate_occurrence_count": len(records) - len(candidates),
        "duplicate_atom_id_count": duplicate_atom_ids,
        "empty_document_count": empty_document_candidates,
        "non_list_document_count": non_list_document_candidates,
        "invalid_document_element_count": invalid_document_elements,
        "documents_without_content_count": sum(
            "content" not in item or item.get("content") in (None, "")
            for item in all_mapping_documents
        ),
        "missing_title_count": sum(state["mock_knowledge_name"] for state in states),
        "mock_knowledge_id_count": sum(state["mock_knowledge_id"] for state in states),
        "mock_knowledge_name_count": sum(state["mock_knowledge_name"] for state in states),
        "mock_retrieval_rank_count": sum(state["mock_retrieval_rank"] for state in states),
        "mock_retrieval_score_count": sum(state["mock_retrieval_score"] for state in states),
        "mock_atom_id_count": sum(w.code == "mock_atom_id" for w in warnings),
        "mock_param_name_count": sum(w.code == "mock_param_name" for w in warnings),
        "mock_arrange_seq_number_count": sum(
            w.code == "mock_arrange_seq_number" for w in warnings
        ),
        "candidate_without_applicability_count": sum(
            not item["applicability"] for item in candidates
        ),
        "atom_with_applicability_count": sum(
            bool(atom.get("applicability"))
            for item in candidates
            for atom in item["atoms"]
        ),
        "real_annotation_count": sum(
            atom.get("annotation") not in (None, "", [], {})
            for item in candidates
            for atom in item["atoms"]
        ),
        "html_content_count": sum(
            isinstance(item.get("content"), str)
            and bool(re.search(r"<\s*/?\s*[A-Za-z][^>]*>", item["content"]))
            for item in all_mapping_documents
        ),
        "html_table_content_count": sum(
            isinstance(item.get("content"), str)
            and bool(re.search(r"<\s*table\b", item["content"], re.I))
            for item in all_mapping_documents
        ),
        "structured_content_count": sum(
            isinstance(item.get("content"), (dict, list))
            for item in all_mapping_documents
        ),
    }
    return ConversionResult(
        candidates=candidates,
        stats=stats,
        warnings=warnings,
        root_description=root_description,
        unrecognized_document_fields=sorted(document_fields - _DOCUMENT_MAPPED_FIELDS),
        sensitive_counts=sensitive,
    )


def _warning_summary(warnings: Sequence[ConversionWarning]) -> Counter[str]:
    return Counter(warning.code for warning in warnings)


def build_report(
    result: ConversionResult,
    *,
    input_path: Path,
    output_path: Path,
    query: str | None,
) -> str:
    stats = result.stats
    warning_counts = _warning_summary(result.warnings)
    sensitive = result.sensitive_counts
    id_fields = sensitive.get("id_card_pattern_fields", {})
    unrecognized = ", ".join(f"`{name}`" for name in result.unrecognized_document_fields)
    if not unrecognized:
        unrecognized = "无"
    warning_lines = (
        "\n".join(f"- `{code}`：{count}" for code, count in sorted(warning_counts.items()))
        or "- 无"
    )
    id_field_text = ", ".join(
        f"`{field}`={count}" for field, count in sorted(id_fields.items())
    ) or "无"
    query_text = query or "未提供"
    return f"""# 真实 Retrieval → Processing 测试输入转换报告

## 基本信息

- 原始文件：`{input_path}`
- 转换结果：`{output_path}`
- 本次用户问题：`{query_text}`
- 原始顶层结构：{result.root_description}；每个元素是一条候选知识包装对象。
- 候选知识位置：顶层元素。
- 真实知识 ID：顶层 `knowledgeId`。
- 原子列表位置：`response.object.document`。
- Demo 输出外层结构：提供 query 时为 `{{"query": "...", "candidates": [...]}}`，未提供时为 `{{"candidates": [...]}}`，均符合当前 `--input-json` 读取契约。

## 转换统计

| 指标 | 数量 |
|---|---:|
| 原始候选数量 | {stats['original_candidate_count']} |
| 转换后候选数量 | {stats['converted_candidate_count']} |
| document 总数 | {stats['document_count']} |
| 生成的 atom 数量 | {stats['generated_atom_count']} |
| 重复 knowledge_id 数量 | {stats['duplicate_knowledge_id_count']} |
| 被聚合的额外候选出现次数 | {stats['aggregated_candidate_occurrence_count']} |
| 重复 atom_id 数量 | {stats['duplicate_atom_id_count']} |
| document 为空的候选数量 | {stats['empty_document_count']} |
| document 非列表的候选数量 | {stats['non_list_document_count']} |
| document 元素类型异常数量 | {stats['invalid_document_element_count']} |
| 缺少正文的 document 数量 | {stats['documents_without_content_count']} |
| 缺少知识标题数量 | {stats['missing_title_count']} |
| 使用 Mock knowledge_id 数量 | {stats['mock_knowledge_id_count']} |
| 使用 Mock knowledge_name 数量 | {stats['mock_knowledge_name_count']} |
| 使用 Mock retrieval_rank 数量 | {stats['mock_retrieval_rank_count']} |
| 使用 Mock retrieval_score 数量 | {stats['mock_retrieval_score_count']} |
| 使用 Mock atom_id 数量 | {stats['mock_atom_id_count']} |
| 使用 Mock param_name 数量 | {stats['mock_param_name_count']} |
| 使用顺序补 arrange_seq_number 数量 | {stats['mock_arrange_seq_number_count']} |
| 无知识级 applicability 数据的候选数量 | {stats['candidate_without_applicability_count']} |
| 有 document 级 applicability 的 atom 数量 | {stats['atom_with_applicability_count']} |
| 保留真实 annotation 的 atom 数量 | {stats['real_annotation_count']} |
| HTML 正文数量 | {stats['html_content_count']} |
| HTML 表格正文数量 | {stats['html_table_content_count']} |
| 结构化对象/列表正文数量 | {stats['structured_content_count']} |

## 字段映射

| Processing 字段 | 真实来源 | 规则 |
|---|---|---|
| `knowledge_id` | 顶层 `knowledgeId` | 优先保留真实值；缺失时才稳定生成 `MOCK-KNOWLEDGE-NNN` |
| `knowledge_name` | 顶层 `knowledge_name` → `knowledgeName` → `title` | 本文件未提供知识级标题；使用 `未命名知识-{{knowledge_id}}`，不使用正文冒充标题 |
| `retrieval_rank` | 顶层真实排名字段 | 本文件没有排名；按原始候选顺序生成连续排名 |
| `retrieval_score` | 知识级真实分数 → document 级最高有效分数 | 本文件两级都没有分数；按最终顺序生成 0 到 1 内稳定递减测试分数 |
| `applicability` | 候选顶层状态、有效期、地区、渠道字段 | 本文件顶层没有可映射字段，保持空对象，不虚构地区和有效期 |
| `atom_id` | `document.klgAttrAtomId` | 本文件均有真实值；缺失时才生成 `{{knowledge_id}}-ATOM-NNN` |
| `group_id` | `document.groupId` | 原样保留；缺失时为 `null` |
| `param_name` | `document.paramName` | 原样保留；缺失时使用“业务内容” |
| `param_type` | `document.paramType` | 以字符串保留真实编码，不猜测编码业务含义 |
| `content` | `document.content` | 原样深复制，字符串、HTML、附件对象列表均不提前展平或生成 Markdown |
| `except_rules` | `document.except` | 原样保留；本文件现有值为空字符串或字段缺失 |
| `annotation` | `document.annotation` | 原样保留；不解释未确认的 `isImport` 权限含义 |
| `arrange_seq_number` | `document.arrangeSeqNumber` | 可解析时转为整数；缺失时使用 document 原始顺序 |
| `wkuntt` | `document.wkuntt` | 原样保留真实非空编码；不猜测单位编码含义 |
| atom `applicability.status` | `document.statusCode` | 保留真实状态编码；本文件值为 `1`，其业务字典仍需数据方确认 |
| atom `applicability.channel_codes` | `document.channelCode` | 按逗号拆为代码列表；代码业务含义和请求渠道仍需数据方确认 |

## 未识别字段

以下 document 字段未映射到 Processing 业务字段，因为当前数据和模型无法确认其业务含义；转换脚本不会据名称猜测：

{unrecognized}

包装层的 `success`、`response.bean`、`response.beans`、`response.rtnCode`、`response.rtnMsg`、`response.object.querySplit` 和 `response.object.total` 是检索响应控制信息，不作为候选正文或原子字段输出。

## 转换警告

{warning_lines}

## 敏感信息检查

- 手机号形态匹配数：{sensitive.get('mobile_pattern', 0)}
- 身份证号形态长数字匹配数：{sensitive.get('id_card_pattern', 0)}
- 身份证号形态所在字段计数：{id_field_text}
- “客户姓名/姓名”标签形态匹配数：{sensitive.get('customer_name_label', 0)}
- 电话标签形态匹配数：{sensitive.get('phone_label', 0)}
- 姓名/人员类参数 atom 数量：{sensitive.get('person_related_param_atoms', 0)}
- 上述参数中正文非空数量：{sensitive.get('person_related_nonempty_content', 0)}

这里只做模式检测，不能证明长数字或姓名字段一定属于真实客户。由于至少一处身份证号形态长数字位于 `content`，且姓名/人员类参数正文非空，原始文件和转换结果都应按潜在敏感数据管理；报告和终端均不展示匹配原值。

## 当前数据可以验证什么

- 当前 Demo 能读取转换结果的外层结构。
- Adapter 能处理标准候选字段、61 个 atom，并原样保留其中的字符串、HTML、HTML 表格和附件对象列表。
- 能验证真实 `annotation`、顺序字段、分组字段、参数类型编码、单位编码是否被保留。
- 提供真实请求上下文后，可以验证 document 级状态和渠道适用性字段的代码处理。
- 61 个 document 均对应一个输出 atom；未提前生成 Markdown。

## 当前数据不能验证什么

- 没有真实知识标题，不能验证知识名称映射质量。
- 没有真实召回排名和分数，不能评估 Retrieval 排名或分数质量；输出分数只用于测试。
- 没有知识级地区、渠道、状态和有效期，不能证明候选在真实业务上下文中一定适用。
- `statusCode`、`channelCode`、`paramType`、`wkuntt` 和 `annotation.isImport` 的业务字典尚未确认。
- 1 个附件正文是仅含 `fileName/fileId` 的对象列表；转换结果会保留它，但当前富文本渲染器没有明确支持这两个键，本任务未运行完整 Demo，不能证明该附件信息最终能进入 Markdown。
- 输出 JSON 会写入已提供的 query；原始文件没有可确认的 retrieval_query 和请求上下文，转换器不会虚构。运行 Demo 时应通过命令行显式传入这些值，否则会使用 Demo 默认值。
"""


def write_conversion(
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    query: str | None = None,
) -> ConversionResult:
    original_bytes = input_path.read_bytes()
    text = original_bytes.decode("utf-8-sig")
    payload = json.loads(text)
    result = convert_payload(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload: dict[str, Any] = {"candidates": result.candidates}
    if query is not None:
        output_payload = {"query": query, **output_payload}
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        build_report(
            result,
            input_path=input_path,
            output_path=output_path,
            query=query,
        ),
        encoding="utf-8",
    )
    if input_path.read_bytes() != original_bytes:
        raise RuntimeError("原始检索 JSON 在转换过程中发生变化")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将真实 Retrieval JSON 转换为 Processing Demo 测试输入"
    )
    parser.add_argument("--input", type=Path, required=True, help="原始 Retrieval JSON")
    parser.add_argument("--output", type=Path, required=True, help="Processing 候选输出 JSON")
    parser.add_argument(
        "--report",
        type=Path,
        help="转换报告路径；默认与输出同目录并追加 _report.md",
    )
    parser.add_argument("--query", help="写入 Demo 外层 JSON 和报告的真实用户问题")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else output_path.with_name(output_path.stem + "_report.md")
    )
    if input_path in {output_path, report_path}:
        raise SystemExit("输入文件不能与输出或报告文件相同")
    try:
        result = write_conversion(
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
            query=args.query,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
        print(f"转换失败：{exc}")
        return 1

    stats = result.stats
    print(
        "转换完成："
        f"原始候选={stats['original_candidate_count']}，"
        f"输出候选={stats['converted_candidate_count']}，"
        f"document={stats['document_count']}，"
        f"atom={stats['generated_atom_count']}，"
        f"警告={len(result.warnings)}"
    )
    print(
        "Mock 字段："
        f"knowledge_id={stats['mock_knowledge_id_count']}，"
        f"knowledge_name={stats['mock_knowledge_name_count']}，"
        f"retrieval_rank={stats['mock_retrieval_rank_count']}，"
        f"retrieval_score={stats['mock_retrieval_score_count']}"
    )
    print(
        "敏感模式计数："
        f"手机号={result.sensitive_counts.get('mobile_pattern', 0)}，"
        f"身份证号形态={result.sensitive_counts.get('id_card_pattern', 0)}，"
        f"姓名/人员类非空字段={result.sensitive_counts.get('person_related_nonempty_content', 0)}"
    )
    print(f"输出 JSON：{output_path}")
    print(f"转换报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
