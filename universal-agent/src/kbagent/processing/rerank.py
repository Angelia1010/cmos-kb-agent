"""基于临时证据编号的两阶段知识重排。"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass
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
from ..shared.knowledge_processing.richtext import render_richtext
from .prompts import RERANK_BATCH_SYSTEM_PROMPT, RERANK_GLOBAL_SYSTEM_PROMPT

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_ASCII_TERM_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]+")
_PROMPT_PREFIX = "RERANK_INPUT_BEGIN\n"
_PROMPT_SUFFIX = "\nRERANK_INPUT_END"
_GLOBAL_PROMPT_SAFETY_MARGIN_CHARS = 256
_INTRO_HEADING_ALIASES = frozenset({
    "业务简介", "套餐简介", "产品简介",
    "业务概述", "套餐概述", "产品概述",
})


def _stable_candidates(candidates: Sequence[ProcessedKnowledge]) -> List[ProcessedKnowledge]:
    return [item for _, item in sorted(
        enumerate(candidates), key=lambda pair: (pair[1].retrieval_rank, pair[0])
    )]


@dataclass
class _MarkdownUnit:
    """不可再拆分的 Markdown 投影单元。"""

    text: str
    heading: str
    order: int
    kind: str
    matched_atom_id: Optional[str] = None
    relevance: int = 0
    priority: int = 0


@dataclass(frozen=True)
class _HeadingRecord:
    line_index: int
    level: int
    line: str
    label: str


@dataclass
class _CandidateProjection:
    evidence_id: str
    candidate: ProcessedKnowledge
    title: str
    units: List[_MarkdownUnit]
    remaining_units: List[_MarkdownUnit]
    source_unit_count: int
    source_content_chars: int
    initial_unit_count: int
    selection_method: str
    initial_body_budget_chars: int = 0

    def content(self) -> str:
        return "\n\n".join(unit.text for unit in self.units)


def _project_title(value: Any, limit: int) -> str:
    title = str(value or "").strip()
    if len(title) <= limit:
        return title
    if limit == 1:
        return "…"
    return title[:limit - 1].rstrip() + "…"


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3


def _is_table_separator(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(_TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def _heading_prefix(headings: Dict[int, str]) -> Tuple[str, str]:
    # H1 是知识标题，已经单独传给模型；正文投影只重复 H2-H6 上下文。
    values = [headings[level] for level in sorted(headings) if level > 1]
    return "\n".join(values), " ".join(
        re.sub(r"^#{1,6}\s+", "", value).strip() for value in values
    )


def _with_heading(prefix: str, body: str) -> str:
    return f"{prefix}\n\n{body}" if prefix else body


def _heading_label(value: str) -> str:
    """规范标题文本用于精确分类，不对业务语义做模糊猜测。"""
    text = re.sub(r"\s+#+\s*$", "", str(value or "").strip())
    text = re.sub(r"\s+", " ", text).strip().rstrip("：:").strip()
    return text.casefold()


def _markdown_heading_records(lines: Sequence[str]) -> List[_HeadingRecord]:
    """提取代码围栏外的ATX标题及其原始位置。"""
    records: List[_HeadingRecord] = []
    fence_char: Optional[str] = None
    fence_length = 0
    for line_index, line in enumerate(lines):
        fence_match = _FENCE_RE.match(line)
        if fence_char is not None:
            if (
                fence_match
                and fence_match.group(1)[0] == fence_char
                and len(fence_match.group(1)) >= fence_length
            ):
                fence_char = None
                fence_length = 0
            continue
        if fence_match:
            token = fence_match.group(1)
            fence_char = token[0]
            fence_length = len(token)
            continue
        heading_match = _HEADING_RE.match(line.strip())
        if not heading_match:
            continue
        records.append(_HeadingRecord(
            line_index=line_index,
            level=len(heading_match.group(1)),
            line=line.strip(),
            label=_heading_label(heading_match.group(2)),
        ))
    return records


def _split_intro_body_units(
    lines: Sequence[str],
    *,
    heading_line: str,
    heading_label: str,
    start_order: int,
) -> List[_MarkdownUnit]:
    """把简介正文切成完整段落、完整代码块和完整表格行。"""
    units: List[_MarkdownUnit] = []
    order = start_order
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        fence_match = _FENCE_RE.match(lines[index])
        if fence_match:
            token = fence_match.group(1)
            block = [lines[index].rstrip()]
            index += 1
            while index < len(lines):
                current = lines[index]
                block.append(current.rstrip())
                closing = _FENCE_RE.match(current)
                index += 1
                if (
                    closing
                    and closing.group(1)[0] == token[0]
                    and len(closing.group(1)) >= len(token)
                ):
                    break
            units.append(_MarkdownUnit(
                text=_with_heading(heading_line, "\n".join(block)),
                heading=heading_label,
                order=order,
                kind="intro_code_block",
            ))
            order += 1
            continue

        if _is_table_row(lines[index]):
            table_lines: List[str] = []
            while index < len(lines) and _is_table_row(lines[index]):
                table_lines.append(lines[index].rstrip())
                index += 1
            if len(table_lines) >= 2 and _is_table_separator(table_lines[1]):
                fixed = table_lines[:2]
                rows = table_lines[2:]
                rendered_rows = ["\n".join([*fixed, row]) for row in rows] or [
                    "\n".join(table_lines)
                ]
            else:
                rendered_rows = table_lines
            for row in rendered_rows:
                units.append(_MarkdownUnit(
                    text=_with_heading(heading_line, row),
                    heading=heading_label,
                    order=order,
                    kind="intro_table_row",
                ))
                order += 1
            continue

        paragraph: List[str] = []
        while index < len(lines):
            current = lines[index]
            if not current.strip() or _is_table_row(current) or _FENCE_RE.match(current):
                break
            paragraph.append(current.rstrip())
            index += 1
        if paragraph:
            units.append(_MarkdownUnit(
                text=_with_heading(heading_line, "\n".join(paragraph)),
                heading=heading_label,
                order=order,
                kind="intro_paragraph",
            ))
            order += 1
        else:
            index += 1
    return units


def _heading_and_intro_units(
    candidate: ProcessedKnowledge,
) -> Tuple[List[_MarkdownUnit], int, str]:
    """生成H1-H3大纲，并仅为精确命中的简介/概述章节附带正文。"""
    lines = str(candidate.content_md or "").strip().splitlines()
    records = _markdown_heading_records(lines)
    outline_units = [
        _MarkdownUnit(
            text=record.line,
            heading=record.label,
            order=order,
            kind=f"heading_h{record.level}",
        )
        for order, record in enumerate(record for record in records if record.level <= 3)
    ]
    intro_units: List[_MarkdownUnit] = []
    covered_until = -1
    for record_index, record in enumerate(records):
        if (
            record.level > 3
            or record.label not in _INTRO_HEADING_ALIASES
            or record.line_index < covered_until
        ):
            continue
        section_end = len(lines)
        for next_record in records[record_index + 1:]:
            if next_record.level <= record.level:
                section_end = next_record.line_index
                break
        intro_units.extend(_split_intro_body_units(
            lines[record.line_index + 1:section_end],
            heading_line=record.line,
            heading_label=record.label,
            start_order=len(outline_units) + len(intro_units),
        ))
        covered_until = section_end

    units = [*outline_units, *intro_units]
    for priority, unit in enumerate(units):
        unit.priority = priority
    return units, len(units), "headings_then_intro_sections"


def _split_markdown_units(content_md: str) -> List[_MarkdownUnit]:
    """按标题、完整段落和完整表格行切分，不截断段落或表格单元格。"""
    lines = str(content_md or "").strip().splitlines()
    headings: Dict[int, str] = {}
    units: List[_MarkdownUnit] = []
    order = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        heading_match = _HEADING_RE.match(line.strip())
        if heading_match:
            level = len(heading_match.group(1))
            headings = {key: value for key, value in headings.items() if key < level}
            headings[level] = line.strip()
            index += 1
            continue
        if not line.strip():
            index += 1
            continue

        prefix, heading_text = _heading_prefix(headings)
        if _is_table_row(line):
            table_lines: List[str] = []
            while index < len(lines) and _is_table_row(lines[index]):
                table_lines.append(lines[index].rstrip())
                index += 1
            if len(table_lines) >= 2 and _is_table_separator(table_lines[1]):
                fixed = table_lines[:2]
                rows = table_lines[2:]
                if rows:
                    for row in rows:
                        units.append(_MarkdownUnit(
                            text=_with_heading(prefix, "\n".join([*fixed, row])),
                            heading=heading_text,
                            order=order,
                            kind="table_row",
                        ))
                        order += 1
                else:
                    units.append(_MarkdownUnit(
                        text=_with_heading(prefix, "\n".join(table_lines)),
                        heading=heading_text,
                        order=order,
                        kind="table",
                    ))
                    order += 1
            else:
                for row in table_lines:
                    units.append(_MarkdownUnit(
                        text=_with_heading(prefix, row),
                        heading=heading_text,
                        order=order,
                        kind="table_row",
                    ))
                    order += 1
            continue

        paragraph: List[str] = []
        while index < len(lines):
            current = lines[index]
            if not current.strip() or _HEADING_RE.match(current.strip()) or _is_table_row(current):
                break
            paragraph.append(current.rstrip())
            index += 1
        if paragraph:
            units.append(_MarkdownUnit(
                text=_with_heading(prefix, "\n".join(paragraph)),
                heading=heading_text,
                order=order,
                kind="paragraph",
            ))
            order += 1
        else:
            # 防御性推进，避免遇到未知行形态时停滞。
            index += 1
    return units


def _keyword_terms(value: Any) -> set[str]:
    text = str(value or "").casefold()
    terms = {match.group(0) for match in _ASCII_TERM_RE.finditer(text)}
    for match in _CJK_TERM_RE.finditer(text):
        segment = match.group(0)
        if len(segment) == 1:
            terms.add(segment)
            continue
        terms.update(segment[index:index + 2] for index in range(len(segment) - 1))
        if len(segment) <= 8:
            terms.add(segment)
    return terms


def _relevance(query: str, unit: _MarkdownUnit) -> int:
    query_terms = _keyword_terms(query)
    if not query_terms:
        return 0
    heading_overlap = len(query_terms & _keyword_terms(unit.heading))
    body_overlap = len(query_terms & _keyword_terms(unit.text))
    exact_bonus = 5 if str(query or "").strip() in unit.text else 0
    return heading_overlap * 3 + body_overlap + exact_bonus


def _unit_key(unit: _MarkdownUnit) -> str:
    return re.sub(r"\s+", " ", unit.text).strip().casefold()


def _matched_atom_units(candidate: ProcessedKnowledge) -> List[_MarkdownUnit]:
    by_id = {
        str(atom.atom_id): atom
        for atom in candidate.atoms
        if atom.atom_id is not None and str(atom.atom_id).strip()
    }
    result: List[_MarkdownUnit] = []
    for matched_id in candidate.matched_atom_ids:
        atom = by_id.get(str(matched_id))
        if atom is None:
            continue
        body = render_richtext(atom.content, [], f"matched_atom[{matched_id}].content").strip()
        if not body:
            continue
        unit = atom.wkuntt or atom.unit
        if unit and not body.rstrip().endswith(str(unit)):
            body = f"{body} {unit}"
        title = str(atom.param_name or atom.title or "详情").strip()
        atom_markdown = f"### {title}\n\n{body}" if title else body
        for item in _split_markdown_units(atom_markdown):
            item.matched_atom_id = str(matched_id)
            result.append(item)
    return result


def _serialized_content_chars(value: str) -> int:
    """返回 JSON 字符串值除引号外的序列化字符数。"""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) - 2


def _units_content(units: Sequence[_MarkdownUnit]) -> str:
    return "\n\n".join(unit.text for unit in units)


def _units_serialized_chars(units: Sequence[_MarkdownUnit]) -> int:
    return _serialized_content_chars(_units_content(units))


def _ordered_content_units(
    candidate: ProcessedKnowledge,
    query: str,
    per_candidate_limit: int,
) -> Tuple[List[_MarkdownUnit], int, str]:
    """生成按 Atom、相关度、原文顺序排列的全部不可分正文单元。"""
    content = str(candidate.content_md or "").strip()
    general_units = _split_markdown_units(content)
    matched_units = _matched_atom_units(candidate)
    all_units: List[_MarkdownUnit] = []
    seen: set[str] = set()
    for unit in [*matched_units, *general_units]:
        key = _unit_key(unit)
        if not key or key in seen:
            continue
        seen.add(key)
        unit.relevance = _relevance(query, unit)
        all_units.append(unit)

    # 内容本身能完整放入时保持原 Markdown 和原顺序，不做无意义重排。
    if content and _serialized_content_chars(content) <= per_candidate_limit:
        document = _MarkdownUnit(content, "", 0, "document")
        document.relevance = _relevance(query, document)
        return [document], 1, "full_content"

    ordered = sorted(
        all_units,
        key=lambda unit: (
            0 if unit.matched_atom_id is not None else 1,
            -unit.relevance,
            unit.order,
        ),
    )
    for priority, unit in enumerate(ordered):
        unit.priority = priority
    method = (
        "matched_atoms_then_keyword_blocks"
        if matched_units else "keyword_blocks_then_source_order"
    )
    return ordered, len(all_units), method


def _select_units_with_budget(
    ordered_units: Sequence[_MarkdownUnit],
    budget: int,
) -> Tuple[List[_MarkdownUnit], List[_MarkdownUnit]]:
    """按优先级选择能完整放入预算的单元，并返回未选择单元。"""
    selected: List[_MarkdownUnit] = []
    remaining: List[_MarkdownUnit] = []
    for unit in ordered_units:
        trial = [*selected, unit]
        if _units_serialized_chars(trial) <= budget:
            selected.append(unit)
        else:
            remaining.append(unit)
    return selected, remaining


def _add_one_fitting_unit(
    projection: _CandidateProjection,
    *,
    per_candidate_limit: int,
    total_remaining: int,
) -> int:
    """按原确定性优先级为一个候选增加一个完整单元，返回新增序列化字符数。"""
    current_chars = _units_serialized_chars(projection.units)
    for index, unit in enumerate(projection.remaining_units):
        trial = sorted([*projection.units, unit], key=lambda item: item.priority)
        trial_chars = _units_serialized_chars(trial)
        added_chars = trial_chars - current_chars
        if trial_chars <= per_candidate_limit and added_chars <= total_remaining:
            projection.units = trial
            projection.remaining_units.pop(index)
            return added_chars
    return 0


def _allocate_content_fairly(
    projections: Sequence[_CandidateProjection],
    *,
    body_budget: int,
    per_candidate_limit: int,
) -> Dict[str, int]:
    """先公平初配，再以候选顺序轮询复用未消耗的正文预算。"""
    candidate_count = len(projections)
    if not candidate_count or body_budget <= 0:
        return {
            "initial_per_candidate_chars": 0,
            "initial_remainder_chars": 0,
            "used_chars": 0,
            "remaining_chars": max(0, body_budget),
        }

    base_quota = min(per_candidate_limit, body_budget // candidate_count)
    distributable_remainder = (
        min(candidate_count, body_budget - base_quota * candidate_count)
        if base_quota < per_candidate_limit else 0
    )
    for index, projection in enumerate(projections):
        quota = min(
            per_candidate_limit,
            base_quota + (1 if index < distributable_remainder else 0),
        )
        projection.initial_body_budget_chars = quota
        projection.units, projection.remaining_units = _select_units_with_budget(
            projection.remaining_units, quota
        )
        projection.initial_unit_count = len(projection.units)

    used = sum(_units_serialized_chars(item.units) for item in projections)
    remaining = max(0, body_budget - used)
    while remaining > 0:
        progressed = False
        for projection in projections:
            added = _add_one_fitting_unit(
                projection,
                per_candidate_limit=per_candidate_limit,
                total_remaining=remaining,
            )
            if added:
                remaining -= added
                progressed = True
        if not progressed:
            break

    used = sum(_units_serialized_chars(item.units) for item in projections)
    return {
        "initial_per_candidate_chars": base_quota,
        "initial_remainder_chars": distributable_remainder,
        "used_chars": used,
        "remaining_chars": max(0, body_budget - used),
    }


def _render_user_prompt(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{_PROMPT_PREFIX}{serialized}{_PROMPT_SUFFIX}"


def _projection_payload(
    projections: Sequence[_CandidateProjection],
    *,
    query: str,
    top_k: int,
    include_content: bool,
    suppress_content: bool = False,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for projection in projections:
        row: Dict[str, Any] = {
            "evidence_id": projection.evidence_id,
            "title": projection.title,
        }
        if include_content:
            row["content_md"] = "" if suppress_content else projection.content()
        candidates.append(row)
    return {"query": query, "top_k": top_k, "candidates": candidates}


def _original_payload(
    evidence: Sequence[Tuple[str, ProcessedKnowledge]],
    *,
    query: str,
    top_k: int,
    include_content: bool,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for evidence_id, candidate in evidence:
        row: Dict[str, Any] = {
            "evidence_id": evidence_id,
            "title": str(candidate.name or ""),
        }
        if include_content:
            row["content_md"] = str(candidate.content_md or "")
        candidates.append(row)
    return {"query": query, "top_k": top_k, "candidates": candidates}


def _build_user_prompt(
    query: str,
    evidence: Sequence[Tuple[str, ProcessedKnowledge]],
    top_k: int,
    options: KnowledgeProcessingOptions,
    *,
    stage: str,
    system_prompt: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """构造有硬字符预算的 Prompt；返回 None 表示最小载荷仍超预算。"""
    is_batch = stage.startswith("batch_")
    mode = options.rerank_input_mode
    heading_intro_mode = mode == "headings_and_intro"
    include_content = heading_intro_mode or mode == "title_and_content" or (
        mode == "title_then_content" and not is_batch
    )
    content_limit = (
        options.prompt_max_chars_per_candidate
        if is_batch else options.global_prompt_max_chars_per_candidate
    )
    budget = options.batch_prompt_max_chars if is_batch else options.global_prompt_max_chars

    projections: List[_CandidateProjection] = []
    for evidence_id, candidate in evidence:
        units: List[_MarkdownUnit] = []
        remaining_units: List[_MarkdownUnit] = []
        source_unit_count = 0
        method = "title_only"
        if include_content:
            if heading_intro_mode:
                ordered_units, source_unit_count, method = _heading_and_intro_units(
                    candidate
                )
            else:
                ordered_units, source_unit_count, method = _ordered_content_units(
                    candidate, query, content_limit
                )
            if is_batch and not heading_intro_mode:
                units, remaining_units = _select_units_with_budget(
                    ordered_units, content_limit
                )
            else:
                remaining_units = ordered_units
        projections.append(_CandidateProjection(
            evidence_id=evidence_id,
            candidate=candidate,
            title=_project_title(candidate.name, options.prompt_max_chars_per_title),
            units=units,
            remaining_units=remaining_units,
            source_unit_count=source_unit_count,
            source_content_chars=len(str(candidate.content_md or "")),
            initial_unit_count=len(units),
            selection_method=method,
            initial_body_budget_chars=(
                content_limit if is_batch and include_content and not heading_intro_mode else 0
            ),
        ))

    original_user = _render_user_prompt(_original_payload(
        evidence,
        query=query,
        top_k=top_k,
        include_content=include_content,
    ))
    original_chars = len(system_prompt) + len(original_user)
    minimum_user_prompt = _render_user_prompt(_projection_payload(
        projections,
        query=query,
        top_k=top_k,
        include_content=include_content,
        suppress_content=True,
    ))
    minimum_required_chars = len(system_prompt) + len(minimum_user_prompt)
    safety_margin_chars = (
        min(
            _GLOBAL_PROMPT_SAFETY_MARGIN_CHARS,
            max(0, budget - minimum_required_chars),
        )
        if not is_batch and include_content else 0
    )
    fairly_allocate_content = include_content and (
        not is_batch or heading_intro_mode
    )
    body_budget_chars = (
        max(0, budget - minimum_required_chars - safety_margin_chars)
        if fairly_allocate_content else 0
    )
    allocation = {
        "initial_per_candidate_chars": 0,
        "initial_remainder_chars": 0,
        "used_chars": 0,
        "remaining_chars": body_budget_chars,
    }
    if fairly_allocate_content:
        allocation = _allocate_content_fairly(
            projections,
            body_budget=body_budget_chars,
            per_candidate_limit=content_limit,
        )

    payload = _projection_payload(
        projections,
        query=query,
        top_k=top_k,
        include_content=include_content,
    )
    user_prompt = _render_user_prompt(payload)
    initial_chars = len(system_prompt) + len(user_prompt)

    while len(system_prompt) + len(user_prompt) > budget:
        removable = [
            (_units_serialized_chars(projection.units), len(projection.units), -index, index)
            for index, projection in enumerate(projections)
            if projection.units
        ]
        if not removable:
            break
        _, _, _, projection_index = max(removable)
        projections[projection_index].units.pop()
        payload = _projection_payload(
            projections,
            query=query,
            top_k=top_k,
            include_content=include_content,
        )
        user_prompt = _render_user_prompt(payload)

    sent_chars = len(system_prompt) + len(user_prompt)
    fits = sent_chars <= budget
    candidate_details = []
    for projection in projections:
        matched_ids = _stable_unique([
            unit.matched_atom_id or ""
            for unit in projection.units
            if unit.matched_atom_id is not None
        ])
        candidate_details.append({
            "evidence_id": projection.evidence_id,
            "source_title_chars": len(str(projection.candidate.name or "")),
            "sent_title_chars": len(projection.title),
            "source_content_chars": projection.source_content_chars,
            "sent_content_chars": len(projection.content()) if include_content else 0,
            "sent_content_serialized_chars": (
                _units_serialized_chars(projection.units) if include_content else 0
            ),
            "initial_body_budget_chars": projection.initial_body_budget_chars,
            "source_unit_count": projection.source_unit_count,
            "initial_unit_count": projection.initial_unit_count,
            "sent_unit_count": len(projection.units),
            "remaining_unit_count": max(
                0, projection.source_unit_count - len(projection.units)
            ),
            "omitted_unit_count": max(0, projection.source_unit_count - len(projection.units)),
            "matched_atom_ids_used": matched_ids,
            "selection_method": projection.selection_method,
        })
    details = {
        "stage": stage,
        "input_mode": mode,
        "budget_chars": budget,
        "original_chars": original_chars,
        "initial_chars": initial_chars,
        "sent_chars": sent_chars if fits else 0,
        "minimum_required_chars": minimum_required_chars,
        "fixed_chars": minimum_required_chars,
        "safety_margin_chars": safety_margin_chars,
        "body_budget_chars": body_budget_chars,
        "initial_per_candidate_body_budget_chars": allocation[
            "initial_per_candidate_chars"
        ],
        "initial_body_budget_remainder_chars": allocation[
            "initial_remainder_chars"
        ],
        "sent_body_serialized_chars": sum(
            item["sent_content_serialized_chars"] for item in candidate_details
        ),
        "unused_body_budget_chars": max(
            0,
            body_budget_chars - sum(
                item["sent_content_serialized_chars"] for item in candidate_details
            ),
        ),
        "within_budget": fits,
        "content_included": include_content,
        "candidate_count": len(projections),
        "omitted_candidate_count": sum(
            1 for item in candidate_details
            if include_content and item["source_content_chars"] and not item["sent_content_chars"]
        ),
        "omitted_unit_count": sum(item["omitted_unit_count"] for item in candidate_details),
        "candidates": candidate_details,
    }
    return (user_prompt if fits else None), details


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
    """每批 Top5、全局最多 25 复排，最终返回 Top3。
    逻辑：先筛出能参与重排的候选 → 分批让模型选 Top5 → 把各批 Top5 汇总成池 
    → 再让模型全局选 Top3 → 任一阶段模型失败都能按原检索排名兜底
    processed_knowledge_candidates
          │
          ↓
   _stable_candidates()
   先稳定排序
          │
          ↓
检查是否有资格参与重排
  ├─ 缺 knowledge_id → 排除
  ├─ 没有有效正文 → 排除
  └─ 合格 → eligible
          │
          ↓
给每条知识分配临时 Evidence ID
E001、E002、E003……
          │
          ↓
eligible <= 3 ?
   │            │
  是            否
   │            ↓
直接返回      按 batch_size 分批
              │
              ↓
         每批 LLM 选 Top5
              │
              ↓
        各批 Top5 汇总
              │
              ↓
       最多取前 25 条
              │
              ↓
        LLM 全局选 Top3
              │
              ↓
         最终 Top3
    """
    # 保留现有函数签名和上下游契约；两阶段精简 Prompt 明确只发送原始 Query。
    del context, retrieval_query
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
    details: Dict[str, Any] = {
        "batches": [],
        "global": {},
        "eligible_count": len(eligible),
        "input_mode": options.rerank_input_mode,
    }
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
        prompt, prompt_details = _build_user_prompt(
            query,
            batch,
            expected,
            options,
            stage=stage,
            system_prompt=RERANK_BATCH_SYSTEM_PROMPT,
        )
        valid: List[str] = []
        complete = False
        batch_stage_warnings: List[ProcessingWarning] = []
        if prompt is None:
            warning = ProcessingWarning(
                code="rerank_prompt_budget_exceeded",
                message=f"{stage} 最小必要载荷超过 {options.batch_prompt_max_chars} 字符，已降级",
                field=stage,
                details=prompt_details,
            )
            warnings.append(warning)
            batch_stage_warnings.append(warning)
        else:
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
            "prompt": prompt_details,
        })

    pool_ids = pool_ids[:options.global_pool_size]
    pool = [(evidence_id, by_evidence[evidence_id]) for evidence_id in pool_ids]
    expected_global = min(options.final_top_k, len(pool))
    global_prompt, global_prompt_details = _build_user_prompt(
        query,
        pool,
        expected_global,
        options,
        stage="global",
        system_prompt=RERANK_GLOBAL_SYSTEM_PROMPT,
    )
    global_valid: List[str] = []
    global_complete = False
    global_stage_warnings: List[ProcessingWarning] = []
    if global_prompt is None:
        warning = ProcessingWarning(
            code="rerank_prompt_budget_exceeded",
            message=f"global 最小必要载荷超过 {options.global_prompt_max_chars} 字符，已降级",
            field="global",
            details=global_prompt_details,
        )
        warnings.append(warning)
        global_stage_warnings.append(warning)
    else:
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
        "prompt": global_prompt_details,
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
