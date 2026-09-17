# -*- coding: utf-8 -*-
"""Retrieval 服务边界:绑定 Workspace、调用固定检索流水线并构造安全响应。

设计要点(对照 processing_service/runner.py):
- 每次请求新建独立 RunWorkspace,不跨请求共享状态;
- 注入 ESClient(生产 ProduceESClient / 离线 MockESClient),不在此处硬编码;
- 复用 RetrievalSubAgent.run 已封装的降级护栏
  (intergrate_all 失败 → keyword_extraction + coarse_recall 兜底);
- 对外只暴露 HTTP 白名单字段,不回显内部 merged_results / DSL。
"""
from __future__ import annotations

import copy
from typing import Any

from kbagent.retrieval.agent import RetrievalSubAgent
from kbagent.shared.config import DEFAULT_CONFIG, Config
from kbagent.shared.models import Chunk
from kbagent.shared.search import ESClient
from kbagent.shared.tracing import Tracer
from kbagent.shared.workspace import RunWorkspace, workspace_scope

from .models import (
    RetrievalChunk,
    RetrievalRequest,
    RetrievalResponseObject,
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
    request_id: str,
    cfg: Config | None = None,
) -> RetrievalResponseObject:
    """执行一次检索请求;不共享 Workspace,也不返回 merged_results 等内部字段。

    Args:
        request: HTTP 请求体(query / region_code)。
        es: 检索后端客户端(生产 ProduceESClient / 离线 MockESClient)。
        request_id: HTTP 边界下发的请求标识(用于日志关联)。
        cfg: 领域配置;缺省使用 DEFAULT_CONFIG。
    """
    cfg = cfg or DEFAULT_CONFIG
    tracer = Tracer()
    ws = RunWorkspace(
        query=request.query,
        cfg=cfg,
        es=es,
        tracer=tracer,
        stage="retrieval",
    )

    degraded = False
    with workspace_scope(ws):
        # RetrievalSubAgent 的 model 参数在直调形态下不参与推理
        # (全程零 LLM),传 None 仅满足构造签名。
        agent = RetrievalSubAgent(model=None, cfg=cfg, tracer=tracer)
        try:
            chunks: list[Chunk] = await agent.run(
                query=request.query,
                region_code=request.region_code,
            )
        except RuntimeError as exc:
            # intergrate_all + coarse_recall 兜底均失败 → 显式失败,
            # 交由 app 层转为 500 错误响应。此处不再降级返回空列表,
            # 避免掩盖检索后端故障。
            tracer.log("retrieval", "fatal", reason=str(exc))
            raise

        # 若主路径(intergrate_all)报错走了兜底,RetrievalSubAgent 已在
        # tracer 中记录 intergrate_all_fallback 事件,据此标记降级。
        events = [e.event for e in tracer.events]
        degraded = "intergrate_all_fallback" in events
        keywords: list[str] = list(ws.data.get("keywords") or [])

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
        chunks=chunk_rows,
    )
