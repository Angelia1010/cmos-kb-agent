"""知识处理的标准数据模型和纯函数能力。

该包不依赖 Workspace 或 LangChain Tool，可独立测试和复用。
"""

from .adapter import (
    normalize_candidate_applicability,
    normalize_knowledge_atom,
    normalize_knowledge_candidate,
    normalize_knowledge_candidates,
    normalize_processing_context,
)
from .analysis import analyze_candidates
from .annotations import filter_annotation_for_audience
from .applicability import evaluate_applicability, filter_candidates
from .atoms import apply_except_override, match_except_conditions, parse_except_rule
from .eligibility import has_renderable_candidate_content, is_rerank_eligible
from .markdown import build_candidate_markdown, build_knowledge_markdown
from .richtext import is_renderable_content, is_supported_content_type
from .models import (
    Applicability,
    FilterDecision,
    KnowledgeAtom,
    KnowledgeCandidate,
    KnowledgeProcessingOptions,
    NormalizationResult,
    PipelineResult,
    ProcessedKnowledge,
    ProcessingContext,
    ProcessingMeta,
    ProcessingWarning,
    RerankResult,
)

__all__ = [
    "Applicability",
    "FilterDecision",
    "KnowledgeAtom",
    "KnowledgeCandidate",
    "KnowledgeProcessingOptions",
    "NormalizationResult",
    "PipelineResult",
    "ProcessedKnowledge",
    "ProcessingContext",
    "ProcessingMeta",
    "ProcessingWarning",
    "RerankResult",
    "normalize_knowledge_atom",
    "normalize_candidate_applicability",
    "normalize_knowledge_candidate",
    "normalize_knowledge_candidates",
    "normalize_processing_context",
    "analyze_candidates",
    "filter_annotation_for_audience",
    "parse_except_rule",
    "match_except_conditions",
    "apply_except_override",
    "evaluate_applicability",
    "filter_candidates",
    "has_renderable_candidate_content",
    "is_rerank_eligible",
    "is_renderable_content",
    "is_supported_content_type",
    "build_candidate_markdown",
    "build_knowledge_markdown",
]
