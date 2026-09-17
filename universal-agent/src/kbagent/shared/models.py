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
class SourceRef:
    """引用文档 — 相关度 + 最能回答问题的原文片段 + 整篇原文。"""
    chunk_id: str
    doc_id: str
    doc_title: str
    relevance: int = 0               # 相关度 0-100,最相关一篇 = 100
    key_fragment: str = ""           # 该文档最能回答用户问题的原文逐字片段
    content: str = ""                # 整篇文档原文
    updated_at: str = ""
    stale: bool = False              # 知识版本过旧提示


# 坐席视角的话术可用性三态(前端据此渲染绿/黄/红横幅)
USABILITY_DIRECT = "directly_usable"    # 可直接使用
USABILITY_VERIFY = "verify_first"       # 核实后使用
USABILITY_NOT = "not_usable"            # 不可用,转人工

_USABILITY_SEVERITY = {USABILITY_DIRECT: 0, USABILITY_VERIFY: 1, USABILITY_NOT: 2}


@dataclass
class Usability:
    """话术可用性判定 — LLM 生成时自评 + 确定性规则纠偏(规则只收紧、不放宽)。"""
    level: str = USABILITY_VERIFY
    reasons: List[str] = field(default_factory=list)      # 判定依据(给坐席看)
    uncovered: List[str] = field(default_factory=list)    # 问题中知识未覆盖的方面

    def tighten(self, level: str, reason: str = "") -> None:
        """规则纠偏:仅当 level 比当前更严格时收紧,并追加依据。"""
        if _USABILITY_SEVERITY.get(level, 2) > _USABILITY_SEVERITY.get(self.level, 1):
            self.level = level
        if reason and reason not in self.reasons:
            self.reasons.append(reason)


# ---------------------------------------------------------------------------
# 文档内证据片段定位(对每篇输入文档,摘出能回答问题的原文逐字片段)
# ---------------------------------------------------------------------------
@dataclass
class LocatedFragment:
    """文档内一段能回答用户问题的原文片段(逐字、可溯源)。"""
    text: str                        # 原文逐字片段(== 所属文档 content[start:end])
    start: int = -1                  # 在文档 content 中的起始偏移;-1 表示未精确定位
    end: int = -1                    # 结束偏移(不含)
    reason: str = ""                 # 该片段为何能回答问题(模型简述)


@dataclass
class DocFragments:
    """单篇文档的片段定位结果。"""
    chunk_id: str
    doc_id: str
    doc_title: str
    answerable: bool                                       # 该文档能否回答用户问题
    relevance: int = 0                                     # 模型自评相关度 0-100(未归一化)
    fragments: List[LocatedFragment] = field(default_factory=list)


@dataclass
class FinalAnswer:
    trace_id: str
    query: str
    script: str = ""                 # 可直接念给用户的口语化话术(注意事项已并入)
    handling_suggestion: str = ""    # 办理建议
    sources: List[SourceRef] = field(default_factory=list)
    degraded: bool = False           # 是否降级结果
    from_cache: bool = False
    elapsed_ms: int = 0
    usability: Usability = field(default_factory=Usability)      # 可用性判定

    def render(self) -> str:
        """渲染为坐席可读文本。"""
        lines = []
        if self.degraded:
            lines.append("[降级结果,未经加工,请核实原文]")
        if self.from_cache:
            lines.append("[缓存命中]")
        lines.append("【坐席话术】")
        lines.append(self.script or "(无)")
        lines.append("")
        lines.append("【办理建议】")
        lines.append(self.handling_suggestion or "(无)")
        if self.sources:
            lines.append("")
            lines.append("【引用文档】")
            for i, s in enumerate(self.sources, 1):
                stale = " (知识可能过旧,请核实)" if s.stale else ""
                lines.append(f"  {i}. {s.doc_title} [{s.chunk_id}] "
                             f"相关度 {s.relevance}% 更新于 {s.updated_at}{stale}")
                if s.key_fragment:
                    lines.append(f"     关键片段: {s.key_fragment[:80]}")
        return "\n".join(lines)
