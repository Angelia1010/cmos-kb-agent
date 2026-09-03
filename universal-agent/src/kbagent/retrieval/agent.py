# -*- coding: utf-8 -*-
"""检索子智能体 — ReAct 自主规划 + GoalLoop 循环护栏。

与主智能体的分工:
  主智能体(main_agent.MainAgent) = **编排好的**:阶段顺序、缓存、降级、trace 固定;
  检索子智能体 = **自主规划的**:LLM 看着检索工具集(Tool List)自主决定
  "调哪个工具、传什么参数、何时结束"(uniagent create_agent → ReAct)。

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

RETRIEVAL_GOAL = (
    "为用户问题召回足量、高相关的候选知识片段。"
    "可用工具:query_understanding / question_rewrite / keyword_extraction / "
    "coarse_recall / intergrate_all,由你自主决定调用顺序;"
    "若收到验证失败反馈,请换策略(改写问题/放宽过滤)重新召回。"
    "提示:intergrate_all 是生产一体化流水线(槽位提取→知识检索→原子表拼接),"
    "仅接入生产 ngkm 检索时可用;离线环境请用 coarse_recall。"
)


class RetrievalSubAgent:
    """检索候选知识子智能体:ReAct 自主规划 + GoalLoop 循环护栏。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer,
                 judge_model: Any = None):
        self.model, self.cfg, self.tracer = model, cfg, tracer
        judge = judge_model if judge_model is not None else model
        self._verifier = SufficiencyVerifier(
            llm_judge=getattr(judge, "judge", None))

    async def run(self, query: str) -> List[Chunk]:
        ws = get_workspace()
        ws.stage = "retrieval"
        loop = create_agent(
            model=self.model, tools=RETRIEVAL_TOOLS,
            features=AgentFeatures(skill=False),        # 业务skill属于处理阶段
            system_prompt="你是候选知识检索子智能体,自主规划检索步骤。",
            goal=RETRIEVAL_GOAL,
            verifier=self._verifier,                    # 充分性检验(规则先行+LLM)
            budget=Budget(config=BudgetConfig(          # 轮次/延迟硬上限
                max_iterations=self.cfg.max_retrieval_rounds,
                max_time_seconds=self.cfg.budget["retrieval_total"] / 1000.0)),
            name="retrieval_subagent",
        )
        result = await loop.run(
            input_messages=[{"role": "user", "content": f"用户问题:{query}"}],
            thread_id=self.tracer.trace_id)
        self.tracer.log("retrieval", "loop_result",
                        success=result.success, iterations=result.iterations,
                        reason=result.reason)
        chunks: List[Chunk] = ws.data.get("chunks", [])
        if not result.success:
            # 错误驱动的失败(LLM/工具异常)且零召回 → 显式失败,触发主智能体降级。
            # 注:此语义此前"碰巧"由框架的 logger 缺失 bug 实现,审查后改为显式表达。
            if not chunks and str(result.reason).startswith("错误"):
                raise RuntimeError(f"检索子智能体失败: {result.reason}")
            # 轮次/预算耗尽仍不充分 → 携带当前最优候选继续(V2 方案 3.1)
            self.tracer.log("retrieval", "exit_with_best",
                            reason=result.reason, count=len(chunks))
        return chunks
