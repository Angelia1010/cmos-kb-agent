# -*- coding: utf-8 -*-
"""核心数据结构:候选知识片段、检索参数、最终答案。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    category: str
    position: Dict[str, Any] = field(default_factory=dict)
    version: str = "v1.0"
    updated_at: str = ""
    score: float = 0.0
    source_chunk_ids: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalParams:
    keywords: List[str]
    expanded_terms: List[str] = field(default_factory=list)
    filters: Dict[str, str] = field(default_factory=dict)
    boost_fields: Dict[str, float] = field(default_factory=dict)
    retrieval_mode: str = "hybrid"

    @staticmethod
    def from_llm_output(data: Dict[str, Any]) -> "RetrievalParams":
        def _str_list(v: Any) -> List[str]:
            if not isinstance(v, list):
                return []
            return [str(x)[:64] for x in v if isinstance(x, (str, int, float))][:10]

        keywords = _str_list(data.get("keywords"))
        expanded = _str_list(data.get("expanded_terms"))
        filters = {
            str(k): str(v)[:64]
            for k, v in (data.get("filters") or {}).items()
            if isinstance(k, str)
        }
        boosts: Dict[str, float] = {}
        for k, v in (data.get("boost_fields") or {}).items():
            try:
                boosts[str(k)] = max(0.1, min(float(v), 10.0))
            except (TypeError, ValueError):
                continue
        mode = data.get("retrieval_mode", "hybrid")
        if mode not in ("keyword", "vector", "hybrid"):
            mode = "hybrid"
        return RetrievalParams(keywords, expanded, filters, boosts, mode)


@dataclass
class SufficiencyResult:
    sufficient: bool
    rule_top3_score: bool
    rule_min_count: bool
    reason: str = ""


@dataclass
class RetrievalRound:
    round_no: int
    params: RetrievalParams
    dsl: Dict[str, Any]
    recalled_titles: List[str]
    chunk_count: int
    sufficiency: Optional[SufficiencyResult] = None


@dataclass
class AnswerSentence:
    text: str
    citations: List[str]
    hard_fact: bool = False
    anchored: bool = True
    dropped: bool = False
    note: str = ""


@dataclass
class SourceRef:
    chunk_id: str
    doc_title: str
    snippet: str
    updated_at: str
    stale: bool = False


@dataclass
class FinalAnswer:
    trace_id: str
    query: str
    business_explanation: str
    handling_suggestion: str
    sentences: List[AnswerSentence] = field(default_factory=list)
    sources: List[SourceRef] = field(default_factory=list)
    degraded: bool = False
    elapsed_ms: int = 0

    def render(self) -> str:
        lines = []
        if self.degraded:
            lines.append("[降级结果,未经加工,请核实原文]")
        lines.append("【业务说明】")
        lines.append(self.business_explanation or "(无)")
        lines.append("")
        lines.append("【办理建议】")
        lines.append(self.handling_suggestion or "(无)")
        if self.sources:
            lines.append("")
            lines.append("【知识来源】")
            for i, s in enumerate(self.sources, 1):
                stale = " (知识可能过旧,请核实)" if s.stale else ""
                lines.append(f"  {i}. {s.doc_title} [{s.chunk_id}] 更新于 {s.updated_at}{stale}")
                lines.append(f"     摘录: {s.snippet[:80]}")
        return "\n".join(lines)
