"""标准候选的幂等 Markdown 构建。"""
from __future__ import annotations

import copy
import re
from typing import List, Sequence, Tuple

from .atoms import group_atoms, process_atoms, render_rule
from .annotations import filter_annotation_for_audience, strip_annotation_fields
from .eligibility import has_renderable_candidate_content
from .models import (
    KnowledgeCandidate,
    KnowledgeProcessingOptions,
    ProcessedKnowledge,
    ProcessingContext,
    ProcessingWarning,
)
from .richtext import (
    normalize_flattened_table_text,
    render_richtext,
    sanitize_title_text,
)


def _without_duplicate_title(text: str, title: str) -> str:
    lines = text.splitlines()
    if lines and re.sub(r"^#+\s*", "", lines[0]).strip() == title.strip():
        return "\n".join(lines[1:]).strip()
    return text


def _without_known_prefix(text: str, prefix: str) -> str:
    """仅删除正文开头完整匹配的已知标题前缀，不拆分任意冒号内容。"""
    prefix = prefix.strip()
    if not text or not prefix:
        return text
    matched = re.match(rf"^{re.escape(prefix)}\s*[：:]\s*", text)
    return text[matched.end():].lstrip() if matched else text


def _effective_period_section(candidate: KnowledgeCandidate) -> str:
    applicability = candidate.applicability
    start = applicability.effective_start or candidate.start_at
    end = applicability.effective_end or candidate.end_at
    lines = []
    if start:
        lines.append(f"- 生效时间：{start}")
    if end:
        lines.append(f"- 失效时间：{end}")
    return "## 有效期\n\n" + "\n".join(lines) if lines else ""


def build_candidate_markdown(
    candidate: KnowledgeCandidate,
    context: ProcessingContext | None = None,
    options: KnowledgeProcessingOptions | None = None,
) -> ProcessedKnowledge:
    """先处理 atoms → 拼主标题和正文
    → 按组拼 atom → 生成 content_md
    → 再包装成 ProcessedKnowledge → 最后检查是不是“空壳内容”"""
    context = context or ProcessingContext()
    options = options or KnowledgeProcessingOptions()
    warnings: List[ProcessingWarning] = []
    atoms, atom_warnings = process_atoms(candidate.atoms, context)
    warnings.extend(atom_warnings)
    display_name = sanitize_title_text(candidate.name) or (
        f"未命名知识-{candidate.source_index + 1:03d}"
    )
    raw_content_path = candidate.content_group_name is not None
    group_name = (
        sanitize_title_text(candidate.content_group_name)
        if raw_content_path else ""
    )
    sections = [f"# {display_name}"]
    main_text = render_richtext(candidate.content, warnings, "content")
    main_text = _without_duplicate_title(main_text, display_name)
    if raw_content_path and main_text:
        main_text = _without_known_prefix(main_text, display_name)
        main_text = _without_known_prefix(main_text, group_name)
        normalized_table, table_detected, table_converted = (
            normalize_flattened_table_text(main_text)
        )
        main_text = normalized_table
        if table_detected and not table_converted:
            warnings.append(ProcessingWarning(
                code="flattened_table_unparsed",
                message="检测到压平表格但无法可靠恢复，已保留原文",
                source_index=candidate.source_index,
                knowledge_id=candidate.knowledge_id,
                field="content",
            ))
    if main_text:
        sections.append(
            f"## {group_name}\n\n{main_text}" if group_name else main_text
        )
    for group, grouped_atoms in group_atoms(atoms):
        sections.append(f"## {group}")
        for atom in grouped_atoms:
            title = atom.param_name or atom.title or "详情"
            lines = [f"### {title}"]
            body = render_richtext(atom.content, warnings, f"atoms[{atom.source_index}].content")
            if body:
                unit = atom.wkuntt or atom.unit
                if unit and not body.rstrip().endswith(unit):
                    body = f"{body} {unit}"
                lines.append(body)
            if options.include_except_rules:
                rules = render_rule(atom.except_rules)
                if rules:
                    lines.append(f"- 例外规则：{rules}")
            if options.include_annotations:
                safe_annotation = filter_annotation_for_audience(
                    atom.annotation, context, warnings, atom_id=atom.atom_id
                )
                annotation = render_rule(safe_annotation)
                if annotation:
                    lines.append(f"- 备注：{annotation}")
            sections.append("\n\n".join(lines))
    # 有效期是业务正文的附加信息，不能单独让空候选变为可重排候选。
    if raw_content_path and main_text:
        period = _effective_period_section(candidate)
        if period:
            sections.append(period)
    content_md = "\n\n".join(section.strip() for section in sections if section.strip()).strip()
    base = copy.deepcopy(candidate)
    result = ProcessedKnowledge(
        knowledge_id=base.knowledge_id,
        chunk_id=base.chunk_id,
        name=display_name,
        content=base.content,
        atoms=atoms,
        retrieval_rank=base.retrieval_rank,
        retrieval_score=base.retrieval_score,
        matched_atom_ids=base.matched_atom_ids,
        source_routes=base.source_routes,
        knowledge_type=base.knowledge_type,
        template_id=base.template_id,
        status=base.status,
        start_at=base.start_at,
        end_at=base.end_at,
        regions=base.regions,
        channels=base.channels,
        region_ids=base.region_ids,
        channel_codes=base.channel_codes,
        applicability=base.applicability,
        source_index=base.source_index,
        metadata=strip_annotation_fields(base.metadata),
        raw=strip_annotation_fields(base.raw),
        content_group_name=group_name if raw_content_path else None,
        content_md=content_md,
        included_atom_count=len(atoms),
        processing_warnings=warnings,
    )
    if not has_renderable_candidate_content(result):
        warnings.append(ProcessingWarning(
            code="empty_rendered_content",
            message="候选没有可渲染的业务正文，不会进入重排",
            source_index=candidate.source_index,
            knowledge_id=candidate.knowledge_id,
            field="content_md",
        ))
    return result


def build_knowledge_markdown(
    candidates: Sequence[KnowledgeCandidate],
    context: ProcessingContext | None = None,
    options: KnowledgeProcessingOptions | None = None,
) -> Tuple[List[ProcessedKnowledge], List[ProcessingWarning]]:
    """批量构建且隔离单篇异常。
    candidates
    ↓
    逐条 candidate
    ↓
    build_candidate_markdown()
    ↓
    成功 → 收集 warning → 判断内容可不可用 → 加入 processed
    失败 → 记 markdown_build_error → 跳过这一条
    ↓
    返回 processed + warnings
    """
    processed: List[ProcessedKnowledge] = []
    warnings: List[ProcessingWarning] = []
    for candidate in candidates:
        try:
            item = build_candidate_markdown(candidate, context, options)
            warnings.extend(item.processing_warnings)
            if has_renderable_candidate_content(item):
                processed.append(item)
        except Exception as exc:  # noqa: BLE001
            warnings.append(ProcessingWarning(
                code="markdown_build_error",
                message=f"Markdown 构建失败，已跳过单篇: {exc}",
                source_index=candidate.source_index,
                knowledge_id=candidate.knowledge_id,
            ))
    return processed, warnings


build_markdown_candidates = build_knowledge_markdown
