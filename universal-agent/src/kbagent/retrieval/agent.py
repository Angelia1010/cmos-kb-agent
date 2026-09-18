# -*- coding: utf-8 -*-
"""检索子智能体 — GoalLoop 驱动的检索→处理→验证闭环。

架构(参考 docs/0917/top3_verifier_integration_guide_20260917.md):
    RetrievalSubAgent 使用 uniagent GoalLoop:
      └─ 每轮 GoalLoop 迭代:
           ① ReAct Agent 自主决定调用检索工具(intergrate_all / query_rewrite /
              keyword_recall / vector_recall)
           ② 迭代结束后,ProcessingVerifier 自动运行:
              - ProcessingSubAgent.run (analyze → filter → markdown → rerank)
              - Top3AnswerabilityVerifier.verify
              ├─ passed  → GoalLoop 返回成功,结束
              └─ failed  → GoalLoop 注入反馈 HumanMessage → 下一轮 ReAct

向后兼容:
    - ``DirectRetrievalSubAgent`` 保留原有直调 intergrate_all 形态(零 LLM),
      供 processing_service 等场景使用。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from uniagent import AgentFeatures, Budget, BudgetConfig, create_agent
from uniagent.verification.verifier import VerificationResult

from ..processing import Top3AnswerabilityVerifier
from ..processing.agent import ProcessingSubAgent
from ..shared.config import Config
from ..shared.knowledge_processing.adapter import normalize_processing_context
from ..shared.knowledge_processing.bridge import retrieval_to_candidates
from ..shared.knowledge_processing.models import (
    ProcessedKnowledge,
    Top3VerificationResult,
)
from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .prompt import RETRIEVAL_GOAL, RETRIEVAL_SYSTEM_PROMPT
from .tools import (
    RETRIEVAL_TOOLS,
    intergrate_all,
    keyword_recall,
)

logger = logging.getLogger("kbagent.retrieval")


# ---------------------------------------------------------------------------
# ProcessingVerifier — 整合 Processing + Top3 验证,挂载到 GoalLoop
# ---------------------------------------------------------------------------

class ProcessingVerifier:
    """GoalLoop 验证器:每轮迭代后自动执行 Processing + Top3 验证。

    实现 uniagent 的 Verifier 协议,在 GoalLoop 的 verify 步骤被调用。
    passed 时 GoalLoop 返回成功;failed 时 GoalLoop 自动注入反馈消息,
    驱动下一轮 ReAct 迭代。

    生成器/评估器分离:验证器拿 goal+state 独立判断,不依赖检索 Agent 的自述。
    """

    def __init__(
        self,
        model: Any,
        tracer: Tracer,
        *,
        verifier_timeout: float = 15.0,
    ) -> None:
        self._processing = ProcessingSubAgent(model)
        self._verifier = Top3AnswerabilityVerifier(
            model, timeout_seconds=verifier_timeout,
        )
        self._tracer = tracer

    async def verify(
        self, goal: str, state: Dict[str, Any]
    ) -> VerificationResult:
        """GoalLoop 每轮迭代结束后调用。

        1. 从 workspace 取 chunks
        2. 跑 Processing 流水线
        3. 跑 Top3 验证
        4. 返回 uniagent VerificationResult
        """
        ws = get_workspace()
        chunks: List[Chunk] = ws.data.get("chunks", [])

        # ── 零召回 → 直接 failed ──
        if not chunks:
            self._tracer.log("retrieval.verify", "zero_chunks")
            return VerificationResult(
                passed=False,
                evidence="本轮未召回任何候选知识。请尝试改写问题、使用不同关键词、或放宽过滤条件(relax_filters=true)。",
                layer="processing",
                confidence=1.0,
            )

        # ── ① Processing ──
        try:
            ws.data["knowledge_candidates"] = retrieval_to_candidates(
                chunks=chunks)
            await self._processing.run()
            top3: List[ProcessedKnowledge] = ws.data.get(
                "top3_candidates", [])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Processing 流水线异常")
            self._tracer.log("retrieval.verify", "processing_error",
                             error=str(exc))
            return VerificationResult(
                passed=False,
                evidence=f"知识处理流水线异常: {exc}。请尝试不同检索策略。",
                layer="processing",
                confidence=1.0,
            )

        if not top3:
            self._tracer.log("retrieval.verify", "empty_top3")
            return VerificationResult(
                passed=False,
                evidence="处理后无有效 Top3 候选。请扩大检索范围或使用不同检索词。",
                layer="processing",
                confidence=1.0,
            )

        # ── ② Verification ──
        retrieval_query = ws.data.get(
            "rewritten_query") or ws.data.get("original_query") or ws.query
        context = normalize_processing_context(
            ws.data.get("processing_context")
        )

        try:
            result: Top3VerificationResult = await self._verifier.verify(
                query=ws.query,
                candidates=top3,
                retrieval_query=retrieval_query,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Top3 Verifier 调用异常")
            self._tracer.log("retrieval.verify", "verifier_exception",
                             error=str(exc))
            return VerificationResult(
                passed=False,
                evidence=f"验证器调用异常: {exc}。请尝试不同检索策略。",
                layer="verifier",
                confidence=1.0,
            )

        self._tracer.log(
            "retrieval.verify", "done",
            status=result.status,
            reason_codes=result.reason_codes,
            evidence_count=len(result.evidence_chunk_ids),
        )

        # ── ③ 映射到 VerificationResult ──
        if result.status == "passed":
            return VerificationResult(
                passed=True,
                evidence=result.summary,
                layer="top3_answerability",
                confidence=1.0,
                details={"evidence_chunk_ids": result.evidence_chunk_ids},
            )

        # failed: 把 structured feedback 写入 workspace,
        #         把 human-readable 建议写入 evidence 供 GoalLoop 注入
        if result.status == "failed" and result.retrieval_feedback is not None:
            fb = result.retrieval_feedback
            ws.data["retrieval_feedback"] = {
                "suggested_query": fb.suggested_query,
                "suggested_keywords": fb.suggested_keywords,
                "missing_aspects": fb.missing_aspects,
                "retry_strategy": fb.retry_strategy,
            }
            evidence_lines = [
                f"验证未通过: {result.summary}",
                f"建议检索语句: {fb.suggested_query}",
                f"缺失方面: {', '.join(fb.missing_aspects)}",
                f"建议关键词: {', '.join(fb.suggested_keywords)}",
                f"调整策略: {fb.retry_strategy}",
                "请根据以上建议调整检索策略后重新召回。",
            ]
            return VerificationResult(
                passed=False,
                evidence="\n".join(evidence_lines),
                layer="top3_answerability",
                confidence=1.0,
                details={"reason_codes": result.reason_codes},
            )

        # unknown: 记录告警,不消耗普通检索重试,携现有结果结束
        self._tracer.log(
            "retrieval.verify", "unknown_degrade",
            reason_codes=result.reason_codes,
        )
        return VerificationResult(
            passed=False,
            evidence=(
                f"验证器技术异常({result.reason_codes}): {result.summary}。"
                "不再重试验证,使用当前检索结果。"
            ),
            layer="top3_answerability",
            confidence=0.0,
            details={"reason_codes": result.reason_codes},
        )


# ---------------------------------------------------------------------------
# RetrievalSubAgent — GoalLoop 驱动的检索→处理→验证
# ---------------------------------------------------------------------------

class RetrievalSubAgent:
    """检索候选知识子智能体:GoalLoop 驱动 ReAct 自主规划 + ProcessingVerifier。

    每轮 GoalLoop 迭代:
      1. ReAct Agent 自主决定调用哪些检索工具
      2. ProcessingVerifier 自动运行 Processing + Top3 验证
      3. passed → 结束; failed → GoalLoop 注入反馈 → 下一轮
    """

    def __init__(
        self,
        model: Any,
        cfg: Config,
        tracer: Tracer,
        *,
        judge_model: Any = None,
        verifier_timeout: float = 15.0,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.tracer = tracer
        self._verifier_timeout = verifier_timeout

    async def run(
        self, query: str, region_code: str = "000"
    ) -> List[Chunk]:
        """GoalLoop 驱动的检索→处理→验证。

        region_code 传省份名或区号(如 福建/591),缺省 "000" 全国。
        """
        ws = get_workspace()
        ws.stage = "retrieval"
        ws.data["original_query"] = query
        ws.data["region_code"] = region_code

        verifier = ProcessingVerifier(
            model=self.model,
            tracer=self.tracer,
            verifier_timeout=self._verifier_timeout,
        )

        loop = create_agent(
            model=self.model,
            tools=RETRIEVAL_TOOLS,
            features=AgentFeatures(skill=False),
            system_prompt=RETRIEVAL_SYSTEM_PROMPT,
            goal=RETRIEVAL_GOAL,
            verifier=verifier,
            budget=Budget(config=BudgetConfig(
                max_iterations=self.cfg.max_retrieval_rounds,
                max_time_seconds=(
                    self.cfg.budget["retrieval_total"] / 1000.0
                ),
            )),
            name="retrieval_subagent",
        )

        result = await loop.run(
            input_messages=[{"role": "user",
                             "content": f"用户问题:{query}\n省份:{region_code}"}],
            thread_id=self.tracer.trace_id,
        )

        self.tracer.log(
            "retrieval", "loop_result",
            success=result.success,
            iterations=result.iterations,
            reason=result.reason,
        )

        chunks: List[Chunk] = ws.data.get("processed_chunks") or ws.data.get(
            "chunks") or []

        if not result.success:
            if not chunks and str(result.reason).startswith("错误"):
                raise RuntimeError(
                    f"检索子智能体失败: {result.reason}")
            self.tracer.log(
                "retrieval", "exit_with_best",
                reason=result.reason, count=len(chunks),
            )

        return chunks


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
        """固定流水线:intergrate_all 一次产出候选片段。"""
        ws = get_workspace()
        ws.stage = "retrieval"
        obs = json.loads(intergrate_all.func(query=query,
                                              region_code=region_code))
        if "error" in obs and not ws.data.get("chunks"):
            # 双路零召回/报错时,单走 keyword 通道再兜底一次
            self.tracer.log("retrieval", "intergrate_all_fallback",
                            reason=obs["error"])
            fallback = json.loads(keyword_recall.func(query=query,
                                                       region_code=region_code))
            if "error" in fallback and not ws.data.get("chunks"):
                raise RuntimeError(
                    f"检索子智能体失败: {fallback['error']}")
        chunks: List[Chunk] = ws.data.get("chunks", [])
        self.tracer.log("retrieval", "done", region_code=region_code,
                        count=len(chunks))
        return chunks