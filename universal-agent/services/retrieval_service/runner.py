# -*- coding: utf-8 -*-
"""Retrieval 服务边界:绑定 Workspace、调用检索流水线并构造安全响应。

设计要点(对照 processing_service/runner.py):
- 每次请求新建独立 RunWorkspace,不跨请求共享状态;
- 注入 ESClient(生产 ProduceESClient / 离线 MockESClient),不在此处硬编码;
- 注入 model 时(服务缺省路径)使用 RetrievalSubAgent:
  create_agent() 组装 GoalLoop —— ReAct 检索 → ProcessingSubAgent 处理
  → Top3AnswerabilityVerifier 验证 → failed 注入反馈重召(最多 N 轮);
  model=None 时回退 DirectRetrievalSubAgent(零 LLM 直调,仅兼容调试);
- 验证器结论(verified / verification_status / reason_codes / loop_iterations)
  随响应透出,调用方无需读 trace 即可知道结果是否通过充分性验证;
- 对外只暴露 HTTP 白名单字段,不回显内部 merged_results / DSL。
"""
from __future__ import annotations

import copy
from typing import Any

from kbagent.retrieval.agent import DirectRetrievalSubAgent, RetrievalSubAgent
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
    model: Any = None,
) -> RetrievalResponseObject:
    """执行一次检索请求;不共享 Workspace,也不返回 merged_results 等内部字段。

    Args:
        request: HTTP 请求体(query / region_code)。
        es: 检索后端客户端(生产 ProduceESClient / 离线 MockESClient)。
        request_id: HTTP 边界下发的请求标识(用于日志关联)。
        cfg: 领域配置;缺省使用 DEFAULT_CONFIG。
        model: LLM 模型;注入时(服务缺省)走 create_agent GoalLoop:
               检索→处理→验证,failed 自动注入反馈重召;
               None 时回退零 LLM 直调(DirectRetrievalSubAgent,兼容调试)。
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
        if model is not None:
            # Agent Loop: 检索 → 处理 → 验证
            agent = RetrievalSubAgent(model=model, cfg=cfg, tracer=tracer)
        else:
            # 零 LLM 直调: 纯检索无 processing/verification
            agent = DirectRetrievalSubAgent(
                model=None, cfg=cfg, tracer=tracer,
            )
        try:
            chunks: list[Chunk] = await agent.run(
                query=request.query,
                region_code=request.region_code,
            )
        except RuntimeError as exc:
            tracer.log("retrieval", "fatal", reason=str(exc))
            raise

        events = [e.event for e in tracer.events]
        degraded = "intergrate_all_fallback" in events
        keywords: list[str] = list(ws.data.get("keywords") or [])
        verification = _collect_verification(tracer)

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
        **verification,
    )


def _collect_verification(tracer: Tracer) -> dict:
    """从 trace 事件汇总 Agent Loop 的验证结论(无 LLM 直调时为 not_run)。

    事件来源(RetrievalSubAgent / ProcessingVerifier):
      - ("retrieval", "loop_result")            → success / iterations
      - ("retrieval.verify", "done")            → status(passed/failed) + reason_codes
      - ("retrieval.verify", "unknown_degrade") → 验证器技术异常
    """
    status = "not_run"
    reason_codes: list[str] = []
    iterations = 0
    for e in tracer.events:
        if e.stage == "retrieval" and e.event == "loop_result":
            iterations = int(e.payload.get("iterations") or 0)
        elif e.stage == "retrieval.verify" and e.event == "done":
            status = str(e.payload.get("status") or status)
            reason_codes = [str(c) for c in
                            (e.payload.get("reason_codes") or [])]
        elif e.stage == "retrieval.verify" and e.event == "unknown_degrade":
            status = "unknown"
            reason_codes = [str(c) for c in
                            (e.payload.get("reason_codes") or [])]
    return {
        "verified": status == "passed",
        "verification_status": status,          # type: ignore[dict-item]
        "verification_reason_codes": reason_codes,
        "loop_iterations": iterations,
    }