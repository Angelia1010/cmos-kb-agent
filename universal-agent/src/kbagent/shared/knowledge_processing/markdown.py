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
from .richtext import render_richtext


def _without_duplicate_title(text: str, title: str) -> str:
    lines = text.splitlines()
    if lines and re.sub(r"^#+\s*", "", lines[0]).strip() == title.strip():
        return "\n".join(lines[1:]).strip()
    return text


def build_candidate_markdown(
    candidate: KnowledgeCandidate,
    context: ProcessingContext | None = None,
    options: KnowledgeProcessingOptions | None = None,
) -> ProcessedKnowledge:
    context = context or ProcessingContext()
    options = options or KnowledgeProcessingOptions()
    warnings: List[ProcessingWarning] = []
    atoms, atom_warnings = process_atoms(candidate.atoms, context)
    warnings.extend(atom_warnings)
    sections = [f"# {candidate.name}"]
    main_text = render_richtext(candidate.content, warnings, "content")
    main_text = _without_duplicate_title(main_text, candidate.name)
    if main_text:
        sections.append(main_text)
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
    content_md = "\n\n".join(section.strip() for section in sections if section.strip()).strip()
    base = copy.deepcopy(candidate)
    result = ProcessedKnowledge(
        knowledge_id=base.knowledge_id,
        name=base.name,
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
    """批量构建且隔离单篇异常。"""
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
