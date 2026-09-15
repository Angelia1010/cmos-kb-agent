# -*- coding: utf-8 -*-
"""Retrieval 服务边界:绑定 Workspace、调用检索子智能体并构造安全响应。

设计要点(对照 processing_service/runner.py):
- 每次请求新建独立 RunWorkspace,不跨请求共享状态;
- 注入 ESClient(生产 ProduceESClient / 离线 MockESClient),不在此处硬编码;
- 复用各 RetrievalSubAgent.run 已封装的降级护栏
  (主路径报错 → keyword_extraction + coarse_recall 兜底);
- 对外只暴露 HTTP 白名单字段,不回显内部 merged_results / DSL。
"""
from __future__ import annotations

import copy
from typing import Any

from kbagent.retrieval.agent import (
    RetrievalKeywordSubAgent,
    RetrievalSubAgent,
    RetrievalVectorSubAgent,
)
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
    mode: str = "keyword",
    vector_mode: str = "new",
    cfg: Config | None = None,
) -> RetrievalResponseObject:
    """执行一次检索请求;不共享 Workspace,也不返回 merged_results 等内部字段。

    通过 ``mode`` 选择召回路径(对应不同的 RetrievalSubAgent):
    - ``keyword``(缺省):RetrievalKeywordSubAgent → keyword_recall 工具
      (槽位提取 → 知识主索引 → 原子表拼接);keywords 为槽位提取结果。
    - ``vector``:RetrievalVectorSubAgent → vector_recall 工具
      (在线 embedding 向量召回);不经槽位提取,keywords 固定为空列表。
    - ``integrate``:RetrievalSubAgent → intergrate_all 工具
      (keyword + vector 双路召回,跨路去重)。

    Args:
        request: HTTP 请求体(query / region_code)。
        es: 检索后端客户端(生产 ProduceESClient / 离线 MockESClient)。
        request_id: HTTP 边界下发的请求标识(用于日志关联)。
        mode: 召回路径(keyword/vector/integrate),缺省 keyword。
        vector_mode: 向量模板(new/old/both),缺省 new。
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

    # 按 mode 选择 agent 与对应降级事件名
    agent_map = {
        "keyword": (RetrievalKeywordSubAgent, "keyword_recall_fallback"),
        "vector": (RetrievalVectorSubAgent, "vector_recall_fallback"),
        "integrate": (RetrievalSubAgent, "intergrate_all_fallback"),
    }
    if mode not in agent_map:
        raise ValueError(f"不支持的 mode: {mode!r},可选 keyword/vector/integrate")
    agent_cls, fallback_event = agent_map[mode]

    degraded = False
    with workspace_scope(ws):
        # 各 SubAgent 的 model 参数在直调形态下不参与推理
        # (全程零 LLM),传 None 仅满足构造签名。
        agent = agent_cls(model=None, cfg=cfg, tracer=tracer)
        try:
            run_kwargs: dict = {"query": request.query,
                                "region_code": request.region_code}
            if mode in ("vector", "integrate"):
                run_kwargs["vector_mode"] = vector_mode
            await agent.run(**run_kwargs)
        except RuntimeError as exc:
            # 主路径 + coarse_recall 兜底均失败 → 显式失败,
            # 交由 app 层转为 500 错误响应。此处不再降级返回空列表,
            # 避免掩盖检索后端故障。
            tracer.log("retrieval", "fatal", reason=str(exc))
            raise

        example: dict = dict(ws.data.get("example") or {})

    return RetrievalResponseObject(
        example=example
    )
# async def run_retrieval_request(
#     request: RetrievalRequest,
#     *,
#     es: ESClient,
#     request_id: str,
#     mode: str = "keyword",
#     cfg: Config | None = None,
# ) -> RetrievalResponseObject:
#     """执行一次检索请求;不共享 Workspace,也不返回 merged_results 等内部字段。

#     通过 ``mode`` 选择召回路径(对应不同的 RetrievalSubAgent):
#     - ``keyword``(缺省):RetrievalKeywordSubAgent → keyword_recall 工具
#       (槽位提取 → 知识主索引 → 原子表拼接);keywords 为槽位提取结果。
#     - ``vector``:RetrievalVectorSubAgent → vector_recall 工具
#       (在线 embedding 向量召回);不经槽位提取,keywords 固定为空列表。
#     - ``integrate``:RetrievalSubAgent → intergrate_all 工具
#       (keyword + vector 双路召回,跨路去重)。

#     Args:
#         request: HTTP 请求体(query / region_code)。
#         es: 检索后端客户端(生产 ProduceESClient / 离线 MockESClient)。
#         request_id: HTTP 边界下发的请求标识(用于日志关联)。
#         mode: 召回路径(keyword/vector/integrate),缺省 keyword。
#         cfg: 领域配置;缺省使用 DEFAULT_CONFIG。
#     """
#     cfg = cfg or DEFAULT_CONFIG
#     tracer = Tracer()
#     ws = RunWorkspace(
#         query=request.query,
#         cfg=cfg,
#         es=es,
#         tracer=tracer,
#         stage="retrieval",
#     )

#     # 按 mode 选择 agent 与对应降级事件名
#     agent_map = {
#         "keyword": (RetrievalKeywordSubAgent, "keyword_recall_fallback"),
#         "vector": (RetrievalVectorSubAgent, "vector_recall_fallback"),
#         "integrate": (RetrievalSubAgent, "intergrate_all_fallback"),
#     }
#     if mode not in agent_map:
#         raise ValueError(f"不支持的 mode: {mode!r},可选 keyword/vector/integrate")
#     agent_cls, fallback_event = agent_map[mode]

#     degraded = False
#     with workspace_scope(ws):
#         # 各 SubAgent 的 model 参数在直调形态下不参与推理
#         # (全程零 LLM),传 None 仅满足构造签名。
#         agent = agent_cls(model=None, cfg=cfg, tracer=tracer)
#         try:
#             chunks: list[Chunk] = await agent.run(
#                 query=request.query,
#                 region_code=request.region_code,
#             )
#         except RuntimeError as exc:
#             # 主路径 + coarse_recall 兜底均失败 → 显式失败,
#             # 交由 app 层转为 500 错误响应。此处不再降级返回空列表,
#             # 避免掩盖检索后端故障。
#             tracer.log("retrieval", "fatal", reason=str(exc))
#             raise

#         # 若主路径报错走了兜底,SubAgent 已在 tracer 中记录
#         # 对应的 *_fallback 事件,据此标记降级。
#         events = [e.event for e in tracer.events]
#         degraded = fallback_event in events
#         keywords: list[str] = list(ws.data.get("keywords") or [])

#     chunk_rows = [_chunk_row(c) for c in chunks]
#     if not chunk_rows:
#         outcome = "no_results"
#     elif degraded:
#         outcome = "degraded"
#     else:
#         outcome = "success"

#     return RetrievalResponseObject(
#         request_id=request_id,
#         trace_id=tracer.trace_id,
#         outcome=outcome,
#         degraded=degraded,
#         recalled_count=len(chunk_rows),
#         elapsed_ms=tracer.elapsed_ms(),
#         region_code=request.region_code,
#         keywords=keywords,
#         chunks=chunk_rows,
#     )
