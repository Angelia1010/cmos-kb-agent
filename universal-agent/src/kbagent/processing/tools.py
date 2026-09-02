# -*- coding: utf-8 -*-
"""数据处理子智能体工具集。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.tools import BaseTool, tool

from ..shared.models import Chunk
from ..shared.knowledge_processing.adapter import (
    normalize_knowledge_candidates,
    normalize_processing_context,
)
from ..shared.knowledge_processing.analysis import analyze_candidates
from ..shared.knowledge_processing.applicability import filter_candidates
from ..shared.knowledge_processing.markdown import build_knowledge_markdown as build_markdown_core
from ..shared.knowledge_processing.eligibility import is_rerank_eligible
from ..shared.knowledge_processing.models import (
    KnowledgeProcessingOptions,
    ProcessingMeta,
    ProcessingWarning,
)
from ..shared.workspace import get_workspace
from .rerank import rerank_candidates


def _obs(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False, default=str)


def _chunks() -> List[Chunk]:
    return get_workspace().data.get("chunks", [])


@tool
def analyze_data() -> str:
    """分析候选集的数量/类目分布/在售状态,为处理决策提供依据。建议首先调用。"""
    ws = get_workspace()
    chunks = _chunks()
    cats: Dict[str, int] = {}
    for c in chunks:
        cats[c.category] = cats.get(c.category, 0) + 1
    report = dict(count=len(chunks), categories=cats,
                  statuses=sorted({c.extra.get("status", "?") for c in chunks}))
    ws.data["analysis"] = report
    return _obs(**report)


@tool
def clean_data() -> str:
    """清洗:去除候选片段内容中的多余空白与控制字符。"""
    for c in _chunks():
        c.content = re.sub(r"\s+", " ", c.content).strip()
    return _obs(cleaned=len(_chunks()))


@tool
def denoise_data() -> str:
    """去噪:剔除下架/失效的知识片段。"""
    ws = get_workspace()
    before = len(_chunks())
    ws.data["chunks"] = [c for c in _chunks() if c.extra.get("status") != "下架"]
    return _obs(before=before, after=len(_chunks()))


@tool
def dedupe_data() -> str:
    """去重:同 chunk_id 只保留得分最高的一条。"""
    ws = get_workspace()
    before = len(_chunks())
    seen: Dict[str, Chunk] = {}
    for c in sorted(_chunks(), key=lambda x: x.score, reverse=True):
        seen.setdefault(c.chunk_id, c)
    ws.data["chunks"] = list(seen.values())
    return _obs(before=before, after=len(_chunks()))


@tool
def structure_data() -> str:
    """结构化:抽取金额/时限等业务字段挂到片段元数据(只增不删)。"""
    n = 0
    for c in _chunks():
        fees = re.findall(r"(\d+(?:\.\d+)?)元", c.content)
        if fees:
            c.extra["fees_yuan"] = fees
            n += 1
        hours = re.findall(r"(\d+)小时", c.content)
        if hours:
            c.extra["deadlines_hours"] = hours
    return _obs(structured=n)


@tool
def sort_data() -> str:
    """排序:得分优先,同分知识按更新时间新者优先。"""
    ws = get_workspace()
    ws.data["chunks"] = sorted(_chunks(),
                               key=lambda c: (c.score, c.updated_at), reverse=True)
    return _obs(sorted=len(_chunks()))


@tool
def apply_business_skill(category: str) -> str:
    """按业务类目做字段归一与业务噪音剔除(不裁剪内容)。category 取值: 套餐|宽带|账单|投诉。技能提示中会给出该类目的具体归一规则。"""
    if category not in ("套餐", "宽带", "账单", "投诉"):
        return _obs(error="category 必须是 套餐|宽带|账单|投诉 之一")
    for c in _chunks():
        if category == "套餐":
            c.content = c.content.replace("月费", "月费(每月)")
        c.extra["skill_applied"] = category
    return _obs(applied=category, count=len(_chunks()))


PROCESSING_TOOLS = [analyze_data, clean_data, denoise_data, dedupe_data,
                    structure_data, sort_data, apply_business_skill]


def run_fallback_pipeline() -> None:
    """确定性保底流水线:子智能体无有效产出时调用。"""
    clean_data.func()
    denoise_data.func()
    dedupe_data.func()
    structure_data.func()
    sort_data.func()


# ---------------------------------------------------------------------------
# 独立知识级 Processing 链路。下列 Tool 不加入旧 PROCESSING_TOOLS，
# 避免与旧 Chunk 工具混合注册。

_DERIVED_KEYS = (
    "normalized_knowledge_candidates",
    "adapter_warnings",
    "knowledge_candidate_analysis",
    "filtered_knowledge_candidates",
    "knowledge_filter_reasons",
    "processed_knowledge_candidates",
    "processing_warnings",
    "rerank_evidence_map",
    "rerank_details",
    "top3_candidates",
    "processed_chunks",
    "processing_meta",
)


def _warning_key(warning: ProcessingWarning) -> tuple[Any, ...]:
    return (
        warning.code, warning.message, warning.source_index,
        warning.knowledge_id, warning.field,
        json.dumps(warning.details, ensure_ascii=False, sort_keys=True, default=str),
    )


def _dedupe_warnings(warnings: List[ProcessingWarning]) -> List[ProcessingWarning]:
    result: List[ProcessingWarning] = []
    seen = set()
    for warning in warnings:
        key = _warning_key(warning)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result


def _trace(event: str, **payload: Any) -> None:
    ws = get_workspace()
    ws.tracer.log("processing.knowledge", event, **payload)


def _analyze_knowledge_candidates(options: KnowledgeProcessingOptions) -> str:
    ws = get_workspace()
    raw = ws.data.get("knowledge_candidates", [])
    normalization = normalize_knowledge_candidates(raw)
    analysis = analyze_candidates(normalization.candidates, options)
    # analyze 是新一轮固定编排的起点，先覆盖所有派生产物。
    ws.data.update({
        "normalized_knowledge_candidates": normalization.candidates,
        "adapter_warnings": normalization.warnings,
        "knowledge_candidate_analysis": analysis,
        "filtered_knowledge_candidates": [],
        "knowledge_filter_reasons": [],
        "processed_knowledge_candidates": [],
        "processing_warnings": _dedupe_warnings(normalization.warnings),
        "rerank_evidence_map": {},
        "rerank_details": {},
        "top3_candidates": [],
        "processed_chunks": [],
        "processing_meta": ProcessingMeta(
            input_count=len(raw) if isinstance(raw, (list, tuple)) else analysis["candidate_count"],
            normalized_count=len(normalization.candidates),
            warning_count=len(normalization.warnings),
            stage_order=["analyze"],
        ),
    })
    _trace(
        "analyze",
        candidate_count=len(normalization.candidates),
        warning_count=len(normalization.warnings),
    )
    return _obs(
        candidates=len(normalization.candidates),
        atoms=analysis["atom_count"],
        warnings=len(normalization.warnings),
    )


def _filter_knowledge_candidates(options: KnowledgeProcessingOptions) -> str:
    del options  # 保留统一的工厂调用契约
    ws = get_workspace()
    normalized = ws.data.get("normalized_knowledge_candidates")
    if not isinstance(normalized, list):
        raise RuntimeError("缺少 normalized_knowledge_candidates，请先执行 analyze")
    context = normalize_processing_context(ws.data.get("processing_context"))
    filtered, decisions, warnings = filter_candidates(normalized, context)
    ws.data["filtered_knowledge_candidates"] = filtered
    ws.data["knowledge_filter_reasons"] = decisions
    combined = _dedupe_warnings([
        *ws.data.get("processing_warnings", []), *warnings,
    ])
    ws.data["processing_warnings"] = combined
    meta = ws.data.get("processing_meta")
    if not isinstance(meta, ProcessingMeta):
        meta = ProcessingMeta(normalized_count=len(normalized))
    meta.filtered_count = len(filtered)
    meta.warning_count = len(combined)
    meta.stage_order = ["analyze", "filter"]
    ws.data["processing_meta"] = meta
    _trace("filter", before=len(normalized), after=len(filtered), warnings=len(warnings))
    return _obs(before=len(normalized), after=len(filtered), filtered=len(normalized) - len(filtered))


def _build_knowledge_markdown(options: KnowledgeProcessingOptions) -> str:
    ws = get_workspace()
    filtered = ws.data.get("filtered_knowledge_candidates")
    if not isinstance(filtered, list):
        raise RuntimeError("缺少 filtered_knowledge_candidates，请先执行 filter")
    context = normalize_processing_context(ws.data.get("processing_context"))
    processed, warnings = build_markdown_core(filtered, context, options)
    combined = _dedupe_warnings([
        *ws.data.get("processing_warnings", []), *warnings,
    ])
    ws.data["processed_knowledge_candidates"] = processed
    ws.data["processing_warnings"] = combined
    meta = ws.data.get("processing_meta")
    if not isinstance(meta, ProcessingMeta):
        meta = ProcessingMeta(filtered_count=len(filtered))
    meta.processed_count = len(processed)
    meta.rerank_eligible_count = sum(is_rerank_eligible(item) for item in processed)
    meta.warning_count = len(combined)
    meta.stage_order = ["analyze", "filter", "build_markdown"]
    ws.data["processing_meta"] = meta
    _trace("build_markdown", before=len(filtered), after=len(processed), warnings=len(warnings))
    return _obs(processed=len(processed), warnings=len(warnings))


async def _rerank_knowledge_candidates(
    model: Any,
    options: KnowledgeProcessingOptions,
) -> str:
    ws = get_workspace()
    processed = ws.data.get("processed_knowledge_candidates")
    if not isinstance(processed, list):
        raise RuntimeError("缺少 processed_knowledge_candidates，请先执行 build_markdown")
    context = normalize_processing_context(ws.data.get("processing_context"))
    result = await rerank_candidates(
        model=model,
        query=ws.query,
        context=context,
        retrieval_query=ws.data.get("retrieval_query"),
        candidates=processed,
        options=options,
    )
    ws.data["rerank_evidence_map"] = result.evidence_map
    ws.data["rerank_details"] = result.details
    ws.data["top3_candidates"] = result.candidates
    ws.data["processing_warnings"] = _dedupe_warnings([
        *ws.data.get("processing_warnings", []), *result.warnings,
    ])
    meta = ws.data.get("processing_meta")
    if not isinstance(meta, ProcessingMeta):
        meta = ProcessingMeta(processed_count=len(processed))
    meta.top_count = len(result.candidates)
    meta.degraded = result.degraded
    # rerank.py 是降级判定和原因的唯一权威来源；Tool 只稳定写回。
    meta.degradation_reasons = list(dict.fromkeys(
        result.details.get("fallback_reasons", [])
    ))
    meta.warning_count = len(ws.data.get("processing_warnings", []))
    meta.stage_order = ["analyze", "filter", "build_markdown", "rerank"]
    ws.data["processing_meta"] = meta
    _trace("rerank", input_count=len(processed), top_count=len(result.candidates), degraded=result.degraded)
    return _obs(top=len(result.candidates), degraded=result.degraded, warnings=len(result.warnings))


@tool
def analyze_knowledge_candidates() -> str:
    """规范化并分析 Workspace 中的知识候选。"""
    return _analyze_knowledge_candidates(KnowledgeProcessingOptions())


@tool
def filter_knowledge_candidates() -> str:
    """按状态、有效期、地区和渠道过滤标准候选。"""
    return _filter_knowledge_candidates(KnowledgeProcessingOptions())


@tool
def build_knowledge_markdown() -> str:
    """将过滤后的知识候选构建为幂等 Markdown。"""
    return _build_knowledge_markdown(KnowledgeProcessingOptions())


@tool
async def rerank_knowledge_candidates() -> str:
    """重排知识候选；正常使用时应通过 build_knowledge_processing_tools 绑定模型。"""
    return _obs(error="rerank_knowledge_candidates 需要通过工厂绑定模型")


def build_knowledge_processing_tools(
    model: Any,
    options: KnowledgeProcessingOptions,
) -> List[BaseTool]:
    """构造独立工具表，通过闭包捕获当前模型和选项。"""
    options = options or KnowledgeProcessingOptions()

    @tool("analyze_knowledge_candidates")
    def captured_analyze() -> str:
        """规范化并分析 Workspace 中的知识候选。"""
        return _analyze_knowledge_candidates(options)

    @tool("filter_knowledge_candidates")
    def captured_filter() -> str:
        """按上下文过滤标准候选及其知识原子。"""
        return _filter_knowledge_candidates(options)

    @tool("build_knowledge_markdown")
    def captured_markdown() -> str:
        """将过滤后候选构建为幂等 Markdown。"""
        return _build_knowledge_markdown(options)

    @tool("rerank_knowledge_candidates")
    async def captured_rerank() -> str:
        """对 Markdown 候选做两阶段重排并写入 Top3。"""
        return await _rerank_knowledge_candidates(model, options)

    return [captured_analyze, captured_filter, captured_markdown, captured_rerank]
