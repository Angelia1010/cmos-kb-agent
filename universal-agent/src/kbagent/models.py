# -*- coding: utf-8 -*-
"""核心数据结构:候选知识片段、检索参数、充分性判定、最终答案。

设计原则(对应方案 3.5 / 4.3):
- Chunk 元数据从召回时刻起只增不删,全链路透传,支撑溯源;
- RetrievalParams 是 LLM 的唯一输出契约,LLM 不接触 ES DSL。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# 候选知识片段(带完整溯源元数据)
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str                    # 例: kb_20260301_0042#p3
    doc_id: str
    doc_title: str
    content: str
    category: str                    # 宽带 / 套餐 / 账单 / 投诉 / ...
    position: Dict[str, Any] = field(default_factory=dict)   # {"para": 3, "offset": [120, 480]}
    version: str = "v1.0"
    updated_at: str = ""
    score: float = 0.0               # 召回相关性得分(融合后)
    source_chunk_ids: List[str] = field(default_factory=list)  # 聚合场景记录来源
    extra: Dict[str, Any] = field(default_factory=dict)        # 处理链路附加信息(只增不删)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 结构化检索参数(方案 3.2:LLM 输出契约,代码侧校验+拼装 DSL)
# ---------------------------------------------------------------------------
@dataclass
class RetrievalParams:
    keywords: List[str]
    expanded_terms: List[str] = field(default_factory=list)
    filters: Dict[str, str] = field(default_factory=dict)
    boost_fields: Dict[str, float] = field(default_factory=dict)
    retrieval_mode: str = "hybrid"   # keyword / vector / hybrid

    @staticmethod
    def from_llm_output(data: Dict[str, Any]) -> "RetrievalParams":
        """从 LLM 的 JSON 输出构造,缺失字段用安全默认值,类型不符直接丢弃。"""
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
                boosts[str(k)] = max(0.1, min(float(v), 10.0))  # 值域夹紧
            except (TypeError, ValueError):
                continue
        mode = data.get("retrieval_mode", "hybrid")
        if mode not in ("keyword", "vector", "hybrid"):
            mode = "hybrid"
        return RetrievalParams(keywords, expanded, filters, boosts, mode)


# ---------------------------------------------------------------------------
# 充分性检验结果(方案 3.4)
# ---------------------------------------------------------------------------
@dataclass
class SufficiencyResult:
    sufficient: bool
    rule_top3_score: bool            # 规则:top3 得分均高于阈值
    rule_min_count: bool             # 规则:候选数量下限
    llm_intent_coverage: Optional[bool]   # LLM:意图覆盖(规则不通过时为 None,未调用)
    uncovered_intents: List[str] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# 检索轮次记录(写入 trace,同时作为下一轮的负例反馈)
# ---------------------------------------------------------------------------
@dataclass
class RetrievalRound:
    round_no: int
    params: RetrievalParams
    dsl: Dict[str, Any]
    recalled_titles: List[str]
    chunk_count: int
    sufficiency: Optional[SufficiencyResult] = None


# ---------------------------------------------------------------------------
# 答案生成产物
# ---------------------------------------------------------------------------
@dataclass
class AnswerSentence:
    text: str
    citations: List[str]             # 引用的 chunk_id 列表
    hard_fact: bool = False          # 是否硬事实(资费/办理条件/生效规则)
    anchored: bool = True            # 锚定校验是否通过
    dropped: bool = False            # 锚定失败被删除
    note: str = ""                   # 例如 "建议核实"


@dataclass
class SourceRef:
    chunk_id: str
    doc_title: str
    snippet: str
    updated_at: str
    stale: bool = False              # 知识版本过旧提示


@dataclass
class FinalAnswer:
    trace_id: str
    query: str
    business_explanation: str        # 业务说明
    handling_suggestion: str         # 办理建议
    sentences: List[AnswerSentence] = field(default_factory=list)
    sources: List[SourceRef] = field(default_factory=list)
    degraded: bool = False           # 是否降级结果
    from_cache: bool = False
    elapsed_ms: int = 0

    def render(self) -> str:
        """渲染为坐席可读文本。"""
        lines = []
        if self.degraded:
            lines.append("[降级结果,未经加工,请核实原文]")
        if self.from_cache:
            lines.append("[缓存命中]")
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
