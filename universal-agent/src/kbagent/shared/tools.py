"""无 Workspace 依赖的公共知识处理 Tool。

这里只负责 LangChain Tool 参数与返回值包装；规范化、过滤和 Markdown
业务规则仍由 ``shared.knowledge_processing`` 中的单一实现负责。
"""
from __future__ import annotations

import copy
import uuid
from collections import Counter
from typing import Any, Mapping, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from .knowledge_processing.adapter import (
    normalize_knowledge_candidates,
    normalize_processing_context,
)
from .knowledge_processing.analysis import analyze_candidates
from .knowledge_processing.applicability import filter_candidates
from .knowledge_processing.markdown import build_knowledge_markdown as build_markdown_core
from .knowledge_processing.models import (
    Applicability,
    KnowledgeAtom,
    KnowledgeCandidate,
    KnowledgeProcessingOptions,
    ProcessedKnowledge,
    ProcessingWarning,
)


def _options(value: Mapping[str, Any] | None) -> KnowledgeProcessingOptions:
    return KnowledgeProcessingOptions(**copy.deepcopy(dict(value or {})))


def _applicability_payload(value: Applicability) -> dict[str, Any]:
    return {
        "status": copy.deepcopy(value.status),
        "effective_start": copy.deepcopy(value.effective_start),
        "effective_end": copy.deepcopy(value.effective_end),
        "region_ids": copy.deepcopy(value.region_ids),
        "regions": copy.deepcopy(value.regions),
        "channel_codes": copy.deepcopy(value.channel_codes),
        "channels": copy.deepcopy(value.channels),
        "excluded_regions": copy.deepcopy(value.excluded_regions),
        "excluded_channels": copy.deepcopy(value.excluded_channels),
        "conditions": copy.deepcopy(value.conditions),
    }


def _warning_payload(value: ProcessingWarning) -> dict[str, Any]:
    return copy.deepcopy(value.to_dict())


def _atom_payload(value: KnowledgeAtom) -> dict[str, Any]:
    return {
        "atom_id": copy.deepcopy(value.atom_id),
        "group_id": copy.deepcopy(value.group_id),
        "param_name": copy.deepcopy(value.param_name),
        "param_type": copy.deepcopy(value.param_type),
        "content": copy.deepcopy(value.content),
        "except_rules": copy.deepcopy(value.except_rules),
        "annotation": copy.deepcopy(value.annotation),
        "arrange_seq_number": copy.deepcopy(value.arrange_seq_number),
        "wkuntt": copy.deepcopy(value.wkuntt),
        "applicability": _applicability_payload(value.applicability),
        # 额外字段只用于溯源；Adapter/Prompt 不会将其提升为业务字段。
        "metadata": copy.deepcopy(value.metadata),
        "raw": copy.deepcopy(value.raw),
    }


def _candidate_payload(value: KnowledgeCandidate) -> dict[str, Any]:
    payload = {
        "knowledge_id": copy.deepcopy(value.knowledge_id),
        "knowledge_name": copy.deepcopy(value.name),
        "content": copy.deepcopy(value.content),
        "retrieval_rank": copy.deepcopy(value.retrieval_rank),
        "retrieval_score": copy.deepcopy(value.retrieval_score),
        "matched_atom_ids": copy.deepcopy(value.matched_atom_ids),
        "source_routes": copy.deepcopy(value.source_routes),
        "knowledge_type": copy.deepcopy(value.knowledge_type),
        "template_id": copy.deepcopy(value.template_id),
        "applicability": _applicability_payload(value.applicability),
        "atoms": [_atom_payload(atom) for atom in value.atoms],
        "metadata": copy.deepcopy(value.metadata),
        "raw": copy.deepcopy(value.raw),
    }
    if isinstance(value, ProcessedKnowledge):
        payload.update({
            "content_md": value.content_md,
            "included_atom_count": value.included_atom_count,
            "processing_warnings": [
                _warning_payload(warning) for warning in value.processing_warnings
            ],
            "rerank_rank": value.rerank_rank,
        })
    return payload


def _candidate_payloads(values: Sequence[KnowledgeCandidate]) -> list[dict[str, Any]]:
    return [_candidate_payload(value) for value in values]


def _warning_counts(warnings: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get("code") or "unknown") for item in warnings).items()))


class _ArtifactStructuredTool(StructuredTool):
    """模型调用返回精简 content，工程结果保存在 artifact。

    LangChain 的 ``content_and_artifact`` 在直接 ``invoke(dict)`` 时只返回
    content。公共 Tool 已对 Python 调用者承诺返回完整字典，因此这里仅对
    非 ToolCall 调用构造内部 ToolCall，再解包 artifact；Agent 路径仍由
    LangChain 原生 ToolMessage 处理。
    """

    def invoke(
        self,
        input: str | dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(input, dict) and input.get("type") == "tool_call":
            return super().invoke(input, config, **kwargs)
        message = super().invoke(
            {
                "name": self.name,
                "args": input,
                "id": f"direct_{uuid.uuid4().hex}",
                "type": "tool_call",
            },
            config,
            **kwargs,
        )
        if not isinstance(message, ToolMessage):
            raise RuntimeError(f"{self.name} 未产生带 artifact 的 ToolMessage")
        return message.artifact


def _analyze_knowledge_candidates(
    candidates: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """独立规范化并分析标准 snake_case 知识候选，不读写 RunWorkspace。"""
    normalized = normalize_knowledge_candidates(candidates)
    analysis = analyze_candidates(normalized.candidates, _options(options))
    artifact = {
        "candidates": _candidate_payloads(normalized.candidates),
        "analysis": copy.deepcopy(analysis),
        "warnings": [_warning_payload(warning) for warning in normalized.warnings],
    }
    content = {
        "stage": "analyze",
        "candidate_count": analysis["candidate_count"],
        "atom_count": analysis["atom_count"],
        "warning_codes": _warning_counts(artifact["warnings"]),
    }
    return content, artifact


def _filter_knowledge_candidates(
    candidates: list[dict[str, Any]],
    processing_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """独立按状态、有效期、地区、渠道和条件过滤标准候选及原子。"""
    normalized = normalize_knowledge_candidates(candidates)
    context = normalize_processing_context(processing_context)
    accepted, decisions, warnings = filter_candidates(normalized.candidates, context)
    artifact = {
        "candidates": _candidate_payloads(accepted),
        "decisions": [copy.deepcopy(decision.to_dict()) for decision in decisions],
        "warnings": [
            _warning_payload(warning)
            for warning in (*normalized.warnings, *warnings)
        ],
    }
    content = {
        "stage": "filter",
        "input_count": len(normalized.candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(normalized.candidates) - len(accepted),
        "warning_codes": _warning_counts(artifact["warnings"]),
    }
    return content, artifact


def _build_knowledge_markdown(
    candidates: list[dict[str, Any]],
    processing_context: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """独立将标准知识候选构建为 Markdown，不要求先执行 Processing 流水线。"""
    normalized = normalize_knowledge_candidates(candidates)
    context = normalize_processing_context(processing_context)
    processed, warnings = build_markdown_core(
        normalized.candidates,
        context,
        _options(options),
    )
    artifact = {
        "candidates": _candidate_payloads(processed),
        "warnings": [
            _warning_payload(warning)
            for warning in (*normalized.warnings, *warnings)
        ],
    }
    content = {
        "stage": "build_markdown",
        "processed_count": len(processed),
        "warning_codes": _warning_counts(artifact["warnings"]),
    }
    return content, artifact


analyze_knowledge_candidates = _ArtifactStructuredTool.from_function(
    func=_analyze_knowledge_candidates,
    name="shared_analyze_knowledge_candidates",
    response_format="content_and_artifact",
)
filter_knowledge_candidates = _ArtifactStructuredTool.from_function(
    func=_filter_knowledge_candidates,
    name="shared_filter_knowledge_candidates",
    response_format="content_and_artifact",
)
build_knowledge_markdown = _ArtifactStructuredTool.from_function(
    func=_build_knowledge_markdown,
    name="shared_build_knowledge_markdown",
    response_format="content_and_artifact",
)


SHARED_KNOWLEDGE_TOOLS: list[BaseTool] = [
    analyze_knowledge_candidates,
    filter_knowledge_candidates,
    build_knowledge_markdown,
]

__all__ = [
    "SHARED_KNOWLEDGE_TOOLS",
    "analyze_knowledge_candidates",
    "filter_knowledge_candidates",
    "build_knowledge_markdown",
]
