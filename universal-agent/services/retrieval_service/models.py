# -*- coding: utf-8 -*-
"""retrieval 服务契约 — 请求/响应 Pydantic 模型。

请求:轻量检索契约(对照 kbagent_service 的灵犀对话格式,此处无对话上下文)
  query / region_code

响应:灵犀返回信封
  rtnCode / rtnMsg / object
  object 内容面向检索场景(召回明细 + 降级标记 + Agent Loop 验证结论),
  不含答案生成字段。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── 请求 ────────────────────────────────────────────────────────────────────

class RetrievalRequest(BaseModel):
    """独立 Retrieval 服务的标准输入。"""
    query: str = Field(min_length=1, description="检索query")
    region_code: str = Field(
        default="000",
        description="区域编码,支持省份名或区号(如 福建/591/000),缺省 000 表示全国")
    mode: str = Field(
        default="integrate",
        description="召回路径:当前固定 integrate(GoalLoop 三轮形态);保留参数兼容旧调用方")
    vector_mode: str = Field(
        default="new",
        description="向量模板选择:new(缺省,新模板)/old(旧模板)/both(双模板按权重混合)")
    # 多余字段直接拒绝,避免调用方拼错字段名被静默忽略
    model_config = ConfigDict(extra="forbid")


class KeywordRequest(BaseModel):
    """关键词单路召回(/keyword)的标准输入。

    关键词路径只走 keyword_search 流水线(LLM 关键词提取 → 知识主索引 → 原子表拼接),
    不走向量召回,故不需要 vector_mode 参数;多余字段直接拒绝。
    """
    query: str = Field(min_length=1, description="检索query")
    region_code: str = Field(
        default="000",
        description="区域编码,支持省份名或区号(如 福建/591/000),缺省 000 表示全国")
    # mode 对 /keyword 端点无实际意义(固定关键词路径),保留默认值兼容旧调用方 body
    mode: str = Field(
        default="integrate",
        description="保留参数,/keyword 固定走关键词召回,不参与分发")
    # 多余字段直接拒绝,避免调用方拼错字段名被静默忽略
    model_config = ConfigDict(extra="forbid")


class VectorRequest(BaseModel):
    """向量单路召回(/vector)的标准输入。

    向量路径只走在线 embedding 向量检索,不经过关键词/槽位提取,
    也不存在多 mode 分发,故不需要 mode 参数;多余字段直接拒绝。
    """
    query: str = Field(min_length=1, description="检索query")
    region_code: str = Field(
        default="000",
        description="区域编码,支持省份名或区号(如 福建/591/000),缺省 000 表示全国")
    vector_mode: str = Field(
        default="new",
        description="向量模板选择:new(缺省,新模板)/old(旧模板)/both(双模板按权重混合)")
    # 多余字段直接拒绝,避免调用方拼错字段名被静默忽略
    model_config = ConfigDict(extra="forbid")


# ── 响应 ────────────────────────────────────────────────────────────────────

class RetrievalChunk(BaseModel):
    """召回片段 — 共享 Chunk 契约的 HTTP 白名单表示。

    与 Processing 服务的 ProcessedChunk 同源,仅暴露检索阶段产出的字段,
    不包含 Processing 后才有的 rerank_rank 等字段。
    """
    chunk_id: str = Field(description="知识片段ID")
    doc_id: str = Field(description="所属文档ID")
    doc_title: str = Field(description="文档标题")
    content: str = Field(description="原文内容")
    category: str = Field(description="知识分类")
    position: dict[str, Any] = Field(description="在文档中的位置")
    version: str = Field(description="知识版本")
    updated_at: str = Field(description="知识更新日期")
    score: float = Field(description="召回得分")
    source_chunk_ids: list[str] = Field(description="溯源片段ID列表")
    extra: dict[str, Any] = Field(description="扩展字段")


class RetrievalResponseObject(BaseModel):
    """object 层 — 检索业务载荷(召回明细 + Agent Loop 验证结论)。

    keywords:关键词召回为 LLM 提取结果。
    rewritten_keywords:query_rewrite 工具产出的改写关键词列表。
    degraded:true 时表示未经主路径召回(走兜底降级),请人工核实。
    """
    request_id: str = Field(description="回传请求ID(优先取 X-Request-ID 头,缺省服务端生成)")
    trace_id: str = Field(description="检索智能体内部trace ID")
    outcome: Literal["success", "no_results", "degraded"] = Field(
        description="结果:success 正常召回;no_results 零召回;degraded 走兜底降级路径")
    degraded: bool = Field(description="是否降级兜底结果;true 时未经主路径召回,请人工核实")
    recalled_count: int = Field(description="召回片段数")
    elapsed_ms: int = Field(description="端到端耗时(毫秒)")
    region_code: str = Field(description="本次检索使用的区域编码")
    keywords: list[str] = Field(default_factory=list, description="检索关键词(LLM 提取结果)")
    rewritten_keywords: list[str] = Field(default_factory=list, description="query_rewrite 产出的改写关键词列表")
    kids: list[str] = Field(default_factory=list, description="召回的知识ID(kid)列表,按得分降序排列")
    chunks: list[RetrievalChunk] = Field(description="召回片段列表")
    example: dict[str, Any] = Field(default_factory=dict, description="检索各阶段示例数据(info_resp/atom_resp原始响应、parsed解析结果、infos/atoms条目列表)")
    # ── Agent Loop 验证结论(检索→处理→Top3 充分性验证) ──
    verified: bool = Field(default=False, description="Top3 充分性验证是否通过")
    verification_status: Literal["passed", "failed", "unknown", "not_run"] = Field(
        default="not_run",
        description="验证器结论:passed 通过;failed 未通过(已按反馈重试到轮次上限);"
                    "unknown 验证器技术异常;not_run 未走 Agent Loop(零 LLM 直调)")
    verification_reason_codes: list[str] = Field(
        default_factory=list,
        description="未通过原因码(off_topic/partial_intent_coverage/missing_key_fact/conflicting_evidence)")
    loop_iterations: int = Field(default=0, description="GoalLoop 实际迭代轮数")


class RetrievalResponse(BaseModel):
    """灵犀返回信封。"""
    rtnCode: str = Field(default="0", description="返回码:0成功,非0见错误码表")
    rtnMsg: str = Field(default="success", description="返回消息")
    object: RetrievalResponseObject = Field(description="业务载荷(检索召回明细 + 降级标记)")


# ── 批量检索契约(基于测试集选取条目,逐条执行检索并统计命中) ────────────────

class BatchRetrievalRequest(BaseModel):
    """批量检索请求:从测试集选取前 count 条逐条执行检索。

    count=0 或留空表示选取测试集全部条目。
    """
    count: int = Field(
        default=10,
        ge=0,
        description="从测试集选取的条目数;0 表示全部")
    model_config = ConfigDict(extra="forbid")


class BatchRetrievalItem(BaseModel):
    """批量检索结果中的单条记录。"""
    index: int = Field(description="测试集索引(0-based)")
    user_query: str = Field(description="用户问题")
    region_code: str = Field(description="区域编码(测试集 province)")
    expected_kids: list[str] = Field(
        default_factory=list,
        description="测试集标注的期望知识ID列表(knowledge_ids)")
    recalled_kids: list[str] = Field(
        default_factory=list,
        description="后端实际召回的kid列表(ranked_kids,按得分降序)")
    hit: bool = Field(
        description="是否命中(recalled_kids 至少包含一个 expected_kid)")
    hit_kids: list[str] = Field(
        default_factory=list,
        description="命中的 expected_kid 子集(便于前端高亮)")
    outcome: str = Field(description="单条检索结果状态(success/no_results/degraded/error)")
    elapsed_ms: int = Field(description="单条检索耗时(毫秒)")
    recalled_count: int = Field(description="召回片段数")


class BatchRetrievalResponseObject(BaseModel):
    """批量检索响应载荷。"""
    total: int = Field(description="实际处理的测试集条目数")
    hit_count: int = Field(description="命中条目数(至少一个 expected_kid 在 recalled_kids 中)")
    total_elapsed_ms: int = Field(description="批量处理端到端总耗时(毫秒)")
    results: list[BatchRetrievalItem] = Field(
        default_factory=list,
        description="逐条结果,顺序与测试集选取一致")


class BatchRetrievalResponse(BaseModel):
    """批量检索返回信封。"""
    rtnCode: str = Field(default="0", description="返回码:0成功,非0见错误码表")
    rtnMsg: str = Field(default="success", description="返回消息")
    object: BatchRetrievalResponseObject = Field(description="批量检索业务载荷")


# ── 错误码 ──────────────────────────────────────────────────────────────────

RTN_OK = "0"                # 成功(含降级结果,降级通过 object.outcome/degraded 表达)
RTN_BAD_REQUEST = "40001"   # 参数校验失败(缺 query / 字段类型错误 / 多余字段)
RTN_INTERNAL = "50001"      # 服务内部未预期异常
RTN_TIMEOUT = "50002"       # 端到端处理超时


def error_body(code: str, msg: str) -> dict:
    """错误响应体:object 为空对象(契约要求 object 必含)。"""
    return {"rtnCode": code, "rtnMsg": msg, "object": {}}
