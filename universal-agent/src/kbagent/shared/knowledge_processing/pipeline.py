"""不依赖 Tool/Workspace 的知识处理流水线。"""
from __future__ import annotations

from typing import Any

from .adapter import normalize_knowledge_candidates, normalize_processing_context
from .analysis import analyze_candidates
from .applicability import filter_candidates
from .eligibility import is_rerank_eligible
from .markdown import build_knowledge_markdown
from .models import KnowledgeProcessingOptions, PipelineResult, ProcessingMeta


def process_knowledge_candidates(
    raw_candidates: Any,
    raw_context: Any = None,
    options: KnowledgeProcessingOptions | None = None,
) -> PipelineResult:
    options = options or KnowledgeProcessingOptions()
    context = normalize_processing_context(raw_context)
    normalized = normalize_knowledge_candidates(raw_candidates)
    analysis = analyze_candidates(normalized.candidates, options)
    filtered, decisions, filter_warnings = filter_candidates(normalized.candidates, context)
    processed, markdown_warnings = build_knowledge_markdown(filtered, context, options)
    warnings = [*normalized.warnings, *filter_warnings, *markdown_warnings]
    meta = ProcessingMeta(
        input_count=len(raw_candidates) if isinstance(raw_candidates, (list, tuple)) else analysis["candidate_count"],
        normalized_count=len(normalized.candidates),
        filtered_count=len(filtered),
        processed_count=len(processed),
        rerank_eligible_count=sum(is_rerank_eligible(item) for item in processed),
        warning_count=len(warnings),
        stage_order=["analyze", "filter", "build_markdown"],
    )
    return PipelineResult(
        normalized=normalized.candidates,
        filtered=filtered,
        processed=processed,
        decisions=decisions,
        warnings=warnings,
        analysis=analysis,
        meta=meta,
    )


run_processing_pipeline = process_knowledge_candidates
