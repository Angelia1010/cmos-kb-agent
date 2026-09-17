# -*- coding: utf-8 -*-
"""Retrieval 服务边界:绑定 Workspace、调用检索子智能体并构造安全响应。

设计要点:
- 每次请求新建独立 RunWorkspace,不跨请求共享状态;
- 注入 ESClient(生产 ProduceESClient / 离线 MockESClient),不在此处硬编码;
- RetrievalSubAgent 为 GoalLoop 三轮形态(intergrate_all→query_rewrite→intergrate_all),
  需要注入 model 供 LLM 推理与 query_rewrite/关键词提取使用;
- 对外只暴露 HTTP 白名单字段,不回显内部 merged_results / DSL。
"""
from __future__ import annotations

import copy
from typing import Any

from kbagent.retrieval.agent import RetrievalSubAgent
from kbagent.retrieval.tools import keyword_recall, vector_recall
from kbagent.shared.config import DEFAULT_CONFIG, Config
from kbagent.shared.models import Chunk
from kbagent.shared.search import ESClient
from kbagent.shared.tracing import Tracer
from kbagent.shared.workspace import RunWorkspace, workspace_scope

# from .models import (
#     RetrievalChunk,
#     RetrievalRequest,
#     RetrievalResponseObject,
# )
from .models import (
    KeywordRequest,
    RetrievalChunk,
    RetrievalRequest,
    RetrievalResponseObject,
    VectorRequest,
)


def _chunk_row(item: Chunk) -> RetrievalChunk:
    """内部 Chunk → HTTP 白名单视图;mutable 字段深拷避免回写工作区。"""
    return RetrievalChunk(
        chunk_id=item.chunk_id,
        doc_id=item.doc_id,
        doc_title=item.doc_title,
        content=item.content,
        category=item.category,
        position=copy.deepcopy(item.position),
        version=item.version,
        updated_at=item.updated_at,
        score=item.score,
        source_chunk_ids=copy.deepcopy(item.source_chunk_ids),
        extra=copy.deepcopy(item.extra),
    )


async def run_retrieval_request(
    request: RetrievalRequest,
    *,
    es: ESClient,
    model: Any = None,
    request_id: str,
    mode: str = "integrate",
    vector_mode: str = "both",
    cfg: Config | None = None,
) -> RetrievalResponseObject:
    """执行一次检索请求;不共享 Workspace,也不返回 merged_results 等内部字段。

    当前仅支持 ``integrate`` 模式:RetrievalSubAgent(GoalLoop 三轮形态)
    - intergrate_all(首轮召回) → query_rewrite(重写) → intergrate_all(二次召回)
    需要注入 model 供 LLM 推理。

    Args:
        request: HTTP 请求体(query / region_code)。
        es: 检索后端客户端(生产 ProduceESClient / 离线 MockESClient)。
        model: BaseChatModel,供 RetrievalSubAgent 的 GoalLoop 与 query_rewrite/关键词提取使用。
        request_id: HTTP 边界下发的请求标识(用于日志关联)。
        mode: 召回路径,当前固定 integrate(保留参数兼容旧调用方)。
        vector_mode: 向量模板(new/old/both),缺省 both(双模板混合)。
        cfg: 领域配置;缺省使用 DEFAULT_CONFIG。
    """
    cfg = cfg or DEFAULT_CONFIG
    tracer = Tracer()
    ws = RunWorkspace(
        query=request.query,
        cfg=cfg,
        es=es,
        tracer=tracer,
        model=model,
        stage="retrieval",
    )

    degraded = False
    with workspace_scope(ws):
        agent = RetrievalSubAgent(model=model, cfg=cfg, tracer=tracer)
        try:
            chunks: list[Chunk] = await agent.run(
                query=request.query,
                region_code=request.region_code,
            )
        except RuntimeError as exc:
            tracer.log("retrieval", "fatal", reason=str(exc))
            raise

        keywords: list[str] = list(ws.data.get("keywords") or [])
        # 原 rewritten_queries(查询重写,已弃用,改为关键词重写)
        # rewritten_queries: list[str] = list(ws.data.get("rewritten_queries") or [])
        rewritten_keywords: list[str] = list(ws.data.get("rewritten_keywords") or [])
        kids: list[str] = list(ws.data.get("ranked_kids") or [])

    chunk_rows = [_chunk_row(c) for c in chunks]
    if not chunk_rows:
        outcome = "no_results"
    elif degraded:
        outcome = "degraded"
    else:
        outcome = "success"

    return RetrievalResponseObject(
        request_id=request_id,
        trace_id=tracer.trace_id,
        outcome=outcome,
        degraded=degraded,
        recalled_count=len(chunk_rows),
        elapsed_ms=tracer.elapsed_ms(),
        region_code=request.region_code,
        keywords=keywords,
        # rewritten_queries=rewritten_queries,
        rewritten_keywords=rewritten_keywords,
        kids=kids,
        chunks=chunk_rows,
    )


def _build_response(
    *,
    request_id: str,
    tracer: Tracer,
    region_code: str,
    chunks: list[Chunk],
    keywords: list[str] | None = None,
    # 原 rewritten_queries(查询重写,已弃用,改为关键词重写)
    # rewritten_queries: list[str] | None = None,
    rewritten_keywords: list[str] | None = None,
    kids: list[str] | None = None,
    degraded: bool = False,
) -> RetrievalResponseObject:
    """构造单路(keyword/vector)响应体,复用 _chunk_row 与 outcome 判定。"""
    chunk_rows = [_chunk_row(c) for c in chunks]
    if not chunk_rows:
        outcome = "no_results"
    elif degraded:
        outcome = "degraded"
    else:
        outcome = "success"
    return RetrievalResponseObject(
        request_id=request_id,
        trace_id=tracer.trace_id,
        outcome=outcome,
        degraded=degraded,
        recalled_count=len(chunk_rows),
        elapsed_ms=tracer.elapsed_ms(),
        region_code=region_code,
        keywords=list(keywords or []),
        # rewritten_queries=list(rewritten_queries or []),
        rewritten_keywords=list(rewritten_keywords or []),
        kids=list(kids or []),
        chunks=chunk_rows,
    )


async def run_keyword_request(
    # 关键词路径不使用 vector_mode,入参类型由 RetrievalRequest 收窄为 KeywordRequest
    # request: RetrievalRequest,
    request: KeywordRequest,
    *,
    es: ESClient,
    model: Any = None,
    request_id: str,
    cfg: Config | None = None,
) -> RetrievalResponseObject:
    """关键词单路召回:直调 keyword_recall 工具(不走 GoalLoop)。

    走 keyword_search 流水线(LLM 关键词提取 → 知识主索引 → 原子表拼接)。
    """
    cfg = cfg or DEFAULT_CONFIG
    tracer = Tracer()
    ws = RunWorkspace(
        query=request.query,
        cfg=cfg,
        es=es,
        tracer=tracer,
        model=model,
        stage="retrieval",
    )
    with workspace_scope(ws):
        keyword_recall.func(
            query=request.query,
            region_code=request.region_code,
        )
        chunks: list[Chunk] = ws.data.get("chunks", [])
        keywords: list[str] = list(ws.data.get("keywords") or [])
        kids: list[str] = list(ws.data.get("ranked_kids") or [])

    return _build_response(
        request_id=request_id,
        tracer=tracer,
        region_code=request.region_code,
        chunks=chunks,
        keywords=keywords,
        kids=kids,
    )


async def run_vector_request(
    # 向量路径不做 mode 分发,入参类型由 RetrievalRequest 收窄为 VectorRequest
    # request: RetrievalRequest,
    request: VectorRequest,
    *,
    es: ESClient,
    model: Any = None,
    request_id: str,
    vector_mode: str = "both",
    cfg: Config | None = None,
) -> RetrievalResponseObject:
    """向量单路召回:直调 vector_recall 工具(不走 GoalLoop)。

    直接走在线知识 embedding 向量检索服务,按语义相似度返回候选片段,
    不经过关键词/槽位提取。
    """
    cfg = cfg or DEFAULT_CONFIG
    tracer = Tracer()
    ws = RunWorkspace(
        query=request.query,
        cfg=cfg,
        es=es,
        tracer=tracer,
        model=model,
        stage="retrieval",
    )
    with workspace_scope(ws):
        vector_recall.func(
            query=request.query,
            region_code=request.region_code,
            vector_mode=vector_mode,
        )
        chunks: list[Chunk] = ws.data.get("chunks", [])
        kids: list[str] = list(ws.data.get("ranked_kids") or [])

    return _build_response(
        request_id=request_id,
        tracer=tracer,
        region_code=request.region_code,
        chunks=chunks,
        kids=kids,
    )
