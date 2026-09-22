"""知识处理链路的标准对象。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Dict, List, Literal, Optional, get_args


VerificationStatus = Literal[
    "passed",
    "failed",
    "unknown",
]

VerificationReasonCode = Literal[
    "no_valid_candidates",
    "off_topic",
    "partial_intent_coverage",
    "missing_key_fact",
    "conflicting_evidence",
    "verifier_timeout",
    "verifier_model_error",
    "verifier_invalid_output",
]

RetryStrategy = Literal[
    "supplement_missing_aspects",
    "replace_off_topic_results",
    "broaden_semantic_recall",
    "narrow_to_business_dimension",
]


_BUSINESS_REASON_CODES = frozenset({
    "no_valid_candidates",
    "off_topic",
    "partial_intent_coverage",
    "missing_key_fact",
    "conflicting_evidence",
})
_TECHNICAL_REASON_CODES = frozenset({
    "verifier_timeout",
    "verifier_model_error",
    "verifier_invalid_output",
})
_GENERIC_MISSING_ASPECTS = frozenset({
    "信息不足",
    "缺少信息",
    "知识不足",
    "资料不足",
    "内容不足",
    "无法回答",
})
_GENERIC_KEYWORDS = frozenset({"信息", "内容", "知识", "问题", "相关", "业务"})


def _clean_string_list(values: list[str], field_name: str) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{field_name} 必须是字符串列表")
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _validate_reason_strategy(
    reason_codes: list[VerificationReasonCode],
    retry_strategy: RetryStrategy,
) -> None:
    reasons = set(reason_codes)
    if "no_valid_candidates" in reasons:
        if reasons != {"no_valid_candidates"} or retry_strategy != "broaden_semantic_recall":
            raise ValueError("no_valid_candidates 只能使用 broaden_semantic_recall")
        return
    if "off_topic" in reasons:
        if reasons != {"off_topic"} or retry_strategy != "replace_off_topic_results":
            raise ValueError("off_topic 只能使用 replace_off_topic_results")
        return
    if "conflicting_evidence" in reasons:
        if retry_strategy != "narrow_to_business_dimension":
            raise ValueError("conflicting_evidence 应使用 narrow_to_business_dimension")
        return
    if retry_strategy != "supplement_missing_aspects":
        raise ValueError("意图覆盖或关键事实缺失应使用 supplement_missing_aspects")


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
    # Retrieval Chunk 的稳定关联键；放在旧字段后保持位置参数兼容。
    chunk_id: str = ""
    # 非结构化正文的展示分组；None 表示非 raw/vector 正文路径。
    content_group_name: Optional[str] = None


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
    rerank_input_mode: Literal[
        "title_only", "headings_and_intro", "title_then_content", "title_and_content"
    ] = "title_then_content"
    prompt_max_chars_per_title: int = 200
    batch_prompt_max_chars: int = 10000
    prompt_max_chars_per_candidate: int = 6000
    global_prompt_max_chars_per_candidate: int = 1000
    global_prompt_max_chars: int = 30000
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
        if self.rerank_input_mode not in {
            "title_only", "headings_and_intro", "title_then_content", "title_and_content",
        }:
            raise ValueError(f"非法 rerank_input_mode: {self.rerank_input_mode}")
        self.prompt_max_chars_per_title = max(1, int(self.prompt_max_chars_per_title))
        self.batch_prompt_max_chars = max(1, int(self.batch_prompt_max_chars))
        self.prompt_max_chars_per_candidate = max(1, int(self.prompt_max_chars_per_candidate))
        self.global_prompt_max_chars_per_candidate = max(
            1, int(self.global_prompt_max_chars_per_candidate)
        )
        self.global_prompt_max_chars = max(1, int(self.global_prompt_max_chars))
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


@dataclass
class RetrievalFeedback(Serializable):
    suggested_query: str
    missing_aspects: list[str]
    suggested_keywords: list[str]
    retry_strategy: RetryStrategy

    def __post_init__(self) -> None:
        if not isinstance(self.suggested_query, str) or not self.suggested_query.strip():
            raise ValueError("suggested_query 必须是非空字符串")
        self.suggested_query = self.suggested_query.strip()
        self.missing_aspects = _clean_string_list(self.missing_aspects, "missing_aspects")
        self.suggested_keywords = _clean_string_list(
            self.suggested_keywords, "suggested_keywords"
        )
        if not self.missing_aspects:
            raise ValueError("missing_aspects 不能为空")
        if any(value in _GENERIC_MISSING_ASPECTS for value in self.missing_aspects):
            raise ValueError("missing_aspects 必须描述具体缺失内容")
        if not self.suggested_keywords:
            raise ValueError("suggested_keywords 不能为空")
        if any(value in _GENERIC_KEYWORDS for value in self.suggested_keywords):
            raise ValueError("suggested_keywords 不能包含无意义泛词")
        if self.retry_strategy not in get_args(RetryStrategy):
            raise ValueError(f"非法 retry_strategy: {self.retry_strategy}")


@dataclass
class Top3VerificationResult(Serializable):
    status: VerificationStatus
    reason_codes: list[VerificationReasonCode]
    summary: str
    evidence_chunk_ids: list[str]
    retrieval_feedback: RetrievalFeedback | None

    def __post_init__(self) -> None:
        if self.status not in get_args(VerificationStatus):
            raise ValueError(f"非法 status: {self.status}")
        self.reason_codes = _clean_string_list(self.reason_codes, "reason_codes")
        if any(reason not in get_args(VerificationReasonCode) for reason in self.reason_codes):
            raise ValueError("reason_codes 包含非法值")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("summary 必须是非空字符串")
        self.summary = self.summary.strip()
        self.evidence_chunk_ids = _clean_string_list(
            self.evidence_chunk_ids, "evidence_chunk_ids"
        )
        if self.retrieval_feedback is not None and not isinstance(
            self.retrieval_feedback, RetrievalFeedback
        ):
            raise ValueError("retrieval_feedback 类型非法")

        reasons = set(self.reason_codes)
        if self.status == "passed":
            if reasons or self.retrieval_feedback is not None:
                raise ValueError("passed 时 reason_codes 必须为空且不能携带检索反馈")
            if not self.evidence_chunk_ids:
                raise ValueError("passed 时 evidence_chunk_ids 不能为空")
            return
        if self.status == "failed":
            if not reasons or not reasons <= _BUSINESS_REASON_CODES:
                raise ValueError("failed 时必须且只能包含业务原因")
            if self.retrieval_feedback is None:
                raise ValueError("failed 时必须提供 retrieval_feedback")
            _validate_reason_strategy(
                self.reason_codes, self.retrieval_feedback.retry_strategy
            )
            return
        if not reasons or not reasons <= _TECHNICAL_REASON_CODES:
            raise ValueError("unknown 时必须且只能包含技术异常原因")
        if self.retrieval_feedback is not None:
            raise ValueError("unknown 时不能生成 retrieval_feedback")
