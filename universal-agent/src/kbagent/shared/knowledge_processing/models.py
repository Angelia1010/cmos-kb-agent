"""知识处理链路的标准对象。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Dict, List, Optional


class Serializable:
    """为 Workspace、Tool 摘要和测试提供稳定的序列化接口。"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingContext(Serializable):
    query: str = ""
    region: Optional[str] = None
    channel: Optional[str] = None
    as_of: Optional[str] = None
    customer_type: Optional[str] = None
    attributes: Dict[str, Any] = dc_field(default_factory=dict)
    raw: Any = None
    # 接口约定字段；放在旧字段之后以保持原有位置参数兼容。
    region_id: Optional[str] = None
    region_name: Optional[str] = None
    channel_code: Optional[str] = None
    request_time: Optional[str] = None
    audience: str = "agent"
    audience_defaulted: bool = False


@dataclass
class Applicability(Serializable):
    status: Optional[str] = None
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    regions: List[str] = dc_field(default_factory=list)
    channels: List[str] = dc_field(default_factory=list)
    excluded_regions: List[str] = dc_field(default_factory=list)
    excluded_channels: List[str] = dc_field(default_factory=list)
    conditions: Dict[str, Any] = dc_field(default_factory=dict)
    raw: Any = None
    effective_start: Optional[str] = None
    effective_end: Optional[str] = None
    region_ids: List[str] = dc_field(default_factory=list)
    channel_codes: List[str] = dc_field(default_factory=list)


@dataclass
class KnowledgeAtom(Serializable):
    atom_id: Optional[str] = None
    title: str = ""
    content: Any = ""
    group: str = ""
    order: int = 0
    unit: Optional[str] = None
    except_rules: Any = None
    annotation: Any = None
    applicability: Applicability = dc_field(default_factory=Applicability)
    source_index: int = 0
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    raw: Any = None
    param_name: str = ""
    param_type: Optional[str] = None
    group_id: str = ""
    arrange_seq_number: Optional[int] = None
    wkuntt: Optional[str] = None


@dataclass
class KnowledgeCandidate(Serializable):
    knowledge_id: Optional[str] = None
    name: str = ""
    content: Any = ""
    atoms: List[KnowledgeAtom] = dc_field(default_factory=list)
    retrieval_rank: int = 0
    status: Optional[str] = None
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    regions: List[str] = dc_field(default_factory=list)
    channels: List[str] = dc_field(default_factory=list)
    applicability: Applicability = dc_field(default_factory=Applicability)
    source_index: int = 0
    metadata: Dict[str, Any] = dc_field(default_factory=dict)
    raw: Any = None
    retrieval_score: Optional[float] = None
    matched_atom_ids: List[str] = dc_field(default_factory=list)
    source_routes: List[str] = dc_field(default_factory=list)
    knowledge_type: Optional[str] = None
    template_id: Optional[str] = None
    # 候选顶层公开适用性字段；Adapter 会与 applicability 双向同步。
    region_ids: List[str] = dc_field(default_factory=list)
    channel_codes: List[str] = dc_field(default_factory=list)


@dataclass
class ProcessedKnowledge(KnowledgeCandidate):
    content_md: str = ""
    included_atom_count: int = 0
    processing_warnings: List["ProcessingWarning"] = dc_field(default_factory=list)
    rerank_rank: Optional[int] = None


@dataclass
class ProcessingWarning(Serializable):
    code: str
    message: str
    source_index: Optional[int] = None
    knowledge_id: Optional[str] = None
    field: Optional[str] = None
    details: Dict[str, Any] = dc_field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class FilterDecision(Serializable):
    accepted: bool
    reasons: List[str] = dc_field(default_factory=list)
    knowledge_id: Optional[str] = None
    source_index: Optional[int] = None
    kept_atom_count: int = 0
    filtered_atom_count: int = 0


@dataclass
class ProcessingMeta(Serializable):
    input_count: int = 0
    normalized_count: int = 0
    filtered_count: int = 0
    processed_count: int = 0
    rerank_eligible_count: int = 0
    top_count: int = 0
    warning_count: int = 0
    degraded: bool = False
    degradation_reasons: List[str] = dc_field(default_factory=list)
    stage_order: List[str] = dc_field(default_factory=list)


@dataclass
class KnowledgeProcessingOptions(Serializable):
    batch_size: int = 20
    batch_top_k: int = 5
    global_pool_size: int = 25
    final_top_k: int = 3
    prompt_max_chars_per_candidate: int = 6000
    long_content_threshold: int = 12000
    include_annotations: bool = True
    include_except_rules: bool = True
    rerank_timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        # 第一版契约的上限是安全边界；测试可用更小值，但不可放大。
        self.batch_size = min(20, max(1, int(self.batch_size)))
        self.batch_top_k = min(5, max(1, int(self.batch_top_k)))
        self.global_pool_size = min(25, max(1, int(self.global_pool_size)))
        self.final_top_k = min(3, max(1, int(self.final_top_k)))
        self.prompt_max_chars_per_candidate = max(1, int(self.prompt_max_chars_per_candidate))
        self.long_content_threshold = max(1, int(self.long_content_threshold))
        self.rerank_timeout_seconds = max(0.001, float(self.rerank_timeout_seconds))


@dataclass
class NormalizationResult(Serializable):
    candidates: List[KnowledgeCandidate] = dc_field(default_factory=list)
    warnings: List[ProcessingWarning] = dc_field(default_factory=list)

    def __iter__(self):
        # 便于使用 candidates, warnings = result。
        yield self.candidates
        yield self.warnings


@dataclass
class PipelineResult(Serializable):
    normalized: List[KnowledgeCandidate] = dc_field(default_factory=list)
    filtered: List[KnowledgeCandidate] = dc_field(default_factory=list)
    processed: List[ProcessedKnowledge] = dc_field(default_factory=list)
    decisions: List[FilterDecision] = dc_field(default_factory=list)
    warnings: List[ProcessingWarning] = dc_field(default_factory=list)
    analysis: Dict[str, Any] = dc_field(default_factory=dict)
    meta: ProcessingMeta = dc_field(default_factory=ProcessingMeta)


@dataclass
class RerankResult(Serializable):
    candidates: List[ProcessedKnowledge] = dc_field(default_factory=list)
    evidence_map: Dict[str, Optional[str]] = dc_field(default_factory=dict)
    details: Dict[str, Any] = dc_field(default_factory=dict)
    warnings: List[ProcessingWarning] = dc_field(default_factory=list)
    degraded: bool = False

    @property
    def top_candidates(self) -> List[ProcessedKnowledge]:
        return self.candidates
