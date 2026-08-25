# -*- coding: utf-8 -*-
"""检索子智能体 — ReAct 自主规划 + GoalLoop 循环护栏。

自主:理解/改写/关键词/召回的编排与参数由 LLM 决定。
护栏:轮次与时间上限(Budget)、充分性判定(SufficiencyVerifier)、
     DSL 字段白名单(coarse_recall 工具内部)。
"""
from __future__ import annotations

from typing import Any, List

from uniagent import AgentFeatures, Budget, BudgetConfig, create_agent

from ..shared.config import Config
from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .sufficiency import SufficiencyVerifier
from .tools import RETRIEVAL_TOOLS

_RETRIEVAL_GOAL = (
    "为用户问题召回足量、高相关的候选知识片段。"
    "可用工具:query_understanding / question_rewrite / keyword_extraction / coarse_recall,"
    "由你自主决定调用顺序;若收到验证失败反馈,请换策略(改写问题/放宽过滤)重新召回。"
)


class RetrievalSubAgent:
    """检索候选知识子智能体。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer):
        self.model = model
        self.cfg = cfg
        self.tracer = tracer
        self._verifier = SufficiencyVerifier()

    async def run(self, query: str) -> List[Chunk]:
        ws = get_workspace()
        ws.stage = "retrieval"
        loop = create_agent(
            model=self.model,
            tools=RETRIEVAL_TOOLS,
            features=AgentFeatures(skill=False),
            system_prompt="你是候选知识检索子智能体,自主规划检索步骤。",
            goal=_RETRIEVAL_GOAL,
            verifier=self._verifier,
            budget=Budget(config=BudgetConfig(
                max_iterations=self.cfg.max_retrieval_rounds,
                max_time_seconds=self.cfg.budget["retrieval_total"] / 1000.0,
            )),
            name="retrieval_subagent",
        )
        result = await loop.run(
            input_messages=[{"role": "user", "content": f"用户问题:{query}"}],
            thread_id=self.tracer.trace_id,
        )
        self.tracer.log("retrieval", "loop_result",
                        success=result.success, iterations=result.iterations,
                        reason=result.reason)
        chunks: List[Chunk] = ws.data.get("chunks", [])
        if not result.success:
            if not chunks and str(result.reason).startswith("错误"):
                raise RuntimeError(f"检索子智能体失败: {result.reason}")
            self.tracer.log("retrieval", "exit_with_best",
                            reason=result.reason, count=len(chunks))
        return chunks
