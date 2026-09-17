# -*- coding: utf-8 -*-
"""检索子智能体 — 检索→处理→验证 Agent Loop。

架构(参考 docs/0917/top3_verifier_integration_guide_20260917.md):
    RetrievalSubAgent (检索模块内部 loop)
      └─ 每轮迭代:
           ① retrieve: intergrate_all (主路径) / keyword_extraction+coarse_recall (降级)
           ② process: retrieval_to_candidates → ProcessingSubAgent.run
                       (analyze → filter → build_markdown → rerank)
           ③ verify: Top3AnswerabilityVerifier.verify(query, top3_candidates)
              ├─ passed  → 结束 loop,返回 verified chunks
              ├─ failed  → 提取 retrieval_feedback,调整策略进入下一轮检索
              └─ unknown → 内部重试 Verifier (最多1次),仍 unknown 则降级处理

向后兼容:
    - ``DirectRetrievalSubAgent`` 保留原有直调 intergrate_all 形态(零 LLM),
      供 processing_service 等不需要 agent loop 的场景使用。
    - ``RetrievalSubAgent`` 现在统一为新 Agent Loop 形态。
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..processing import Top3AnswerabilityVerifier
from ..processing.agent import ProcessingSubAgent
from ..shared.config import Config
from ..shared.knowledge_processing.adapter import normalize_processing_context
from ..shared.knowledge_processing.bridge import retrieval_to_candidates
from ..shared.knowledge_processing.models import (
    ProcessedKnowledge,
    RetrievalFeedback,
    Top3VerificationResult,
)
from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .tools import (
    coarse_recall,
    intergrate_all,
    keyword_extraction,
)

logger = logging.getLogger("kbagent.retrieval")

# ---------------------------------------------------------------------------
# Agent Loop 配置
# ---------------------------------------------------------------------------

@dataclass
class RetrievalLoopConfig:
    """检索 Agent Loop 的可调参数。

    参考 docs/0917/top3_verifier_integration_guide_20260917.md:
    - max_rounds:         最大检索轮次,2轮对齐现有 max_retrieval_rounds
    - verifier_retry:     unknown 时 Verifier 内部重试次数
    - verifier_timeout:   Verifier 模型调用超时(秒)
    - dup_query_threshold: 连续相同 query 的检测阈值,达到后强制退出
    """
    max_rounds: int = 2
    verifier_retry: int = 1
    verifier_timeout: float = 15.0
    dup_query_threshold: int = 2


# ---------------------------------------------------------------------------
# Agent Loop 实现
# ---------------------------------------------------------------------------

class RetrievalSubAgent:
    """检索→处理→验证 闭环 Agent。

    由 MainAgent 创建(每请求一实例),内部循环:
    retrieve → process → verify → (passed: exit / failed: retry / unknown: degrade)。

    与 MainAgent 的契约:
    - 输入: 原始 query + region_code
    - 输出: List[Chunk] (经过检索、处理重排、验证的 top chunks)
    - 异常: 全部路径失败时 raise RuntimeError,触发 MainAgent 降级兜底
    """

    def __init__(
        self,
        model: Any,
        cfg: Config,
        tracer: Tracer,
        *,
        judge_model: Any = None,
        loop_cfg: RetrievalLoopConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.tracer = tracer
        self.loop_cfg = loop_cfg or RetrievalLoopConfig()

        # Verifier: 独立模型调用,使用注入的 model
        self._verifier = Top3AnswerabilityVerifier(
            model,
            timeout_seconds=self.loop_cfg.verifier_timeout,
        )
        # Processing: 可复用(固定流水线,无请求级可变状态)
        self._processing = ProcessingSubAgent(model)

    # ------------------------------------------------------------------
    async def run(self, query: str, region_code: str = "000") -> List[Chunk]:
        """执行检索→处理→验证闭环。

        返回经过 processing+rerank 且通过 verifier 的 top chunks;
        若全部轮次耗尽仍验证失败,返回当前最优 top3 并标记 degraded。
        若零召回且全部路径失败,raise RuntimeError 触发 MainAgent 降级。
        """
        ws = get_workspace()
        ws.stage = "retrieval"

        # 记录初始 query 到 workspace,供本轮与后续组件使用
        ws.data["original_query"] = query
        ws.data["region_code"] = region_code

        current_query = query
        current_keywords: List[str] = []
        history_queries: List[str] = []
        best_chunks: List[Chunk] = []

        for round_num in range(1, self.loop_cfg.max_rounds + 1):
            self.tracer.log(
                "retrieval.loop", f"round_{round_num}_start",
                query=current_query, region_code=region_code,
                keywords=current_keywords,
            )

            # ── ① Retrieve ──────────────────────────────────────────
            chunks = await self._do_retrieve(
                query=current_query,
                region_code=region_code,
                round_num=round_num,
                is_retry=(round_num > 1),
            )
            if not chunks:
                self.tracer.log("retrieval.loop", f"round_{round_num}_zero_recall")
                if best_chunks:
                    # 本轮零召回但有历史最优 → 退出
                    break
                continue

            # 更新历史最优
            best_chunks = list(chunks)

            # ── ② Process ──────────────────────────────────────────
            top3 = await self._do_process(chunks)
            if not top3:
                self.tracer.log("retrieval.loop", f"round_{round_num}_empty_top3")
                continue

            # ── ③ Verify ───────────────────────────────────────────
            context = normalize_processing_context(
                ws.data.get("processing_context")
            )
            result = await self._do_verify(
                query=query,              # 始终用原始 query
                candidates=top3,
                retrieval_query=current_query,
                context=context,
            )

            self.tracer.log(
                "retrieval.loop", f"round_{round_num}_verify",
                status=result.status,
                reason_codes=result.reason_codes,
            )

            if result.status == "passed":
                self.tracer.log("retrieval.loop", "passed",
                                round=round_num,
                                evidence=result.evidence_chunk_ids)
                return self._final_chunks(chunks, result)

            if result.status == "failed":
                feedback = result.retrieval_feedback
                if feedback is None:
                    # 防御: no_valid_candidates 等场景 feedback 非 None
                    break

                # 重复检测
                dup_count = sum(
                    1 for hq in history_queries
                    if hq.strip() == (feedback.suggested_query or "").strip()
                )
                if dup_count >= self.loop_cfg.dup_query_threshold:
                    self.tracer.log(
                        "retrieval.loop", "dup_query_break",
                        query=feedback.suggested_query,
                    )
                    break

                # 准备下一轮
                history_queries.append(current_query)
                current_query = feedback.suggested_query or current_query
                current_keywords = list(feedback.suggested_keywords)
                self.tracer.log(
                    "retrieval.loop", f"round_{round_num}_feedback",
                    next_query=current_query,
                    keywords=current_keywords,
                    strategy=feedback.retry_strategy,
                    missing=feedback.missing_aspects,
                )
                continue

            # status == "unknown": 技术异常,内部重试后仍异常则降级
            if result.status == "unknown":
                retry_ok = await self._retry_verifier(
                    query=query,
                    candidates=top3,
                    retrieval_query=current_query,
                    context=context,
                )
                if retry_ok:
                    # 重试通过
                    return self._final_chunks(chunks, retry_ok)
                # 仍 unknown → 记录告警,携当前最优退出
                self.tracer.log(
                    "retrieval.loop", f"round_{round_num}_unknown_degrade",
                    reason_codes=result.reason_codes,
                )
                break

        # ── Loop 结束: 轮次耗尽 / 异常退出 ─────────────────────────
        if best_chunks:
            self.tracer.log(
                "retrieval.loop", "exhausted_best_effort",
                rounds=round_num,
                chunk_count=len(best_chunks),
            )
            return best_chunks

        raise RuntimeError(
            f"检索 Agent Loop 全部 {self.loop_cfg.max_rounds} 轮未产生有效候选"
        )

    # ------------------------------------------------------------------
    # 内部步骤
    # ------------------------------------------------------------------

    async def _do_retrieve(
        self,
        query: str,
        region_code: str,
        round_num: int,
        is_retry: bool,
    ) -> List[Chunk]:
        """执行一轮检索: intergrate_all 主路径,失败则 keyword_extraction + coarse_recall 兜底。

        重试轮(is_retry=True)会在 coarse_recall 中开启 relax_filters。
        """
        ws = get_workspace()

        # 主路径: 生产一体化流水线
        obs = json.loads(
            intergrate_all.func(query=query, region_code=region_code)
        )
        if "error" not in obs:
            chunks: List[Chunk] = ws.data.get("chunks", [])
            if chunks:
                self.tracer.log(
                    "retrieval.loop", f"round_{round_num}_intergrate_all",
                    count=len(chunks),
                )
                return chunks

        # 降级路径: keyword_extraction + coarse_recall
        self.tracer.log(
            "retrieval.loop", f"round_{round_num}_fallback",
            reason=obs.get("error", "zero_recall"),
        )
        keyword_extraction.func()
        fallback_obs = json.loads(
            coarse_recall.func(
                relax_filters=is_retry,
                retrieval_mode="hybrid",
            )
        )
        if "error" in fallback_obs:
            ws_chunks: List[Chunk] = ws.data.get("chunks", [])
            if not ws_chunks:
                self.tracer.log(
                    "retrieval.loop", f"round_{round_num}_fallback_error",
                    error=fallback_obs["error"],
                )
                return []
        return list(ws.data.get("chunks", []))

    async def _do_process(
        self, chunks: List[Chunk]
    ) -> List[ProcessedKnowledge]:
        """执行固定处理流水线: retrieval_to_candidates → ProcessingSubAgent.run。

        ProcessingSubAgent 内部: analyze → filter → build_markdown → rerank,
        产出 ws.data["top3_candidates"] (List[ProcessedKnowledge])。
        """
        ws = get_workspace()

        # 将 Chunk 映射为 KnowledgeCandidate
        ws.data["knowledge_candidates"] = retrieval_to_candidates(chunks=chunks)

        # 执行固定流水线
        await self._processing.run()

        top3: List[ProcessedKnowledge] = ws.data.get("top3_candidates", [])
        if not top3:
            logger.warning("Processing 后 top3_candidates 为空")
        return top3

    async def _do_verify(
        self,
        query: str,
        candidates: Sequence[ProcessedKnowledge],
        retrieval_query: str | None = None,
        context: Any = None,
    ) -> Top3VerificationResult:
        """调用 Verifier,将可能的异常收敛为 unknown 结果。"""
        try:
            return await self._verifier.verify(
                query=query,
                candidates=candidates,
                retrieval_query=retrieval_query,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Verifier 调用异常: %s", exc)
            from ..shared.knowledge_processing.models import (
                RetrievalFeedback,
            )
            # 防御: 构造 unknown 结果
            return Top3VerificationResult(
                status="unknown",
                reason_codes=["verifier_model_error"],
                summary=f"Verifier 调用异常: {exc}",
                evidence_chunk_ids=[],
                retrieval_feedback=None,
            )

    async def _retry_verifier(
        self,
        query: str,
        candidates: Sequence[ProcessedKnowledge],
        retrieval_query: str | None = None,
        context: Any = None,
    ) -> Top3VerificationResult | None:
        """unknown 时重试 Verifier 最多 ``verifier_retry`` 次。

        Returns:
            重试通过时返回 passed 结果;全部失败返回 None。
        """
        for attempt in range(1, self.loop_cfg.verifier_retry + 1):
            self.tracer.log(
                "retrieval.loop", f"verifier_retry_{attempt}",
            )
            try:
                result = await self._verifier.verify(
                    query=query,
                    candidates=candidates,
                    retrieval_query=retrieval_query,
                    context=context,
                )
            except Exception:
                continue

            if result.status == "passed":
                self.tracer.log(
                    "retrieval.loop", "verifier_retry_passed",
                    attempt=attempt,
                )
                return result
            if result.status == "failed":
                # 重试中模型明确判定 failed,不再重试
                break

        return None

    @staticmethod
    def _final_chunks(
        chunks: List[Chunk],
        verification: Top3VerificationResult,
    ) -> List[Chunk]:
        """根据验证结果过滤 chunks,只返回 evidence_chunk_ids 中的 chunk。"""
        ws = get_workspace()
        processed = ws.data.get("processed_chunks") or ws.data.get("chunks") or chunks
        if not verification.evidence_chunk_ids:
            return list(processed)
        evidence_set = set(verification.evidence_chunk_ids)
        filtered = [c for c in processed if c.chunk_id in evidence_set]
        return filtered if filtered else list(processed)


# ---------------------------------------------------------------------------
# 向后兼容: 直调形态(零 LLM,供 processing_service 等场景)
# ---------------------------------------------------------------------------

class DirectRetrievalSubAgent:
    """检索候选知识子智能体:直调 intergrate_all,不走 agent loop。

    供不需要 processing+verification 闭环的场景(如 processing_service
    独立调试)使用;主链路请使用 ``RetrievalSubAgent``。
    """

    def __init__(self, model: Any, cfg: Config, tracer: Tracer,
                 judge_model: Any = None):
        self.model, self.cfg, self.tracer = model, cfg, tracer

    async def run(self, query: str, region_code: str = "000") -> List[Chunk]:
        """固定流水线:intergrate_all 一次产出候选片段。

        region_code 传省份名或区号(如 福建/591),缺省 "000" 全国。
        """
        ws = get_workspace()
        ws.stage = "retrieval"
        obs = json.loads(intergrate_all.func(query=query,
                                              region_code=region_code))
        if "error" in obs:
            self.tracer.log("retrieval", "intergrate_all_fallback",
                            reason=obs["error"])
            keyword_extraction.func()
            fallback = json.loads(coarse_recall.func())
            if "error" in fallback and not ws.data.get("chunks"):
                raise RuntimeError(
                    f"检索子智能体失败: {fallback['error']}")
        chunks: List[Chunk] = ws.data.get("chunks", [])
        self.tracer.log("retrieval", "done", region_code=region_code,
                        count=len(chunks))
        return chunks