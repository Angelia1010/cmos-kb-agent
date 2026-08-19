# -*- coding: utf-8 -*-
"""三个子智能体 —— 均为"自主规划"形态(LLM 面对工具集自主决定调用序列)。

与主智能体的分工:
  主智能体(main_agent.MainAgent) = **编排好的**:阶段顺序、缓存、降级、trace 固定;
  子智能体 = **自主规划的**:每个阶段内部,LLM 看着自己的工具集(Tool List)
  自主决定"调哪个工具、传什么参数、何时结束"(uniagent create_agent → ReAct)。

自主性与护栏的边界(刻意设计,不交给 LLM):
  检索子智能体   自主: 理解/改写/关键词/召回的编排与参数
                护栏: 轮次与时间上限(Budget)、充分性判定(SufficiencyVerifier)、
                      DSL 白名单(coarse_recall 工具内部)
  处理子智能体   自主: 清洗工具的取舍与顺序、业务skill的套用
                护栏: 无产出时的确定性保底流水线、"不裁剪片段"写入工具约束
  答案子智能体   自主: 素材取舍与答案组织(内联引用由 LLM 生成)
                护栏: 逐句锚定校验为确定性代码,硬事实锚定失败直接删句
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.messages import HumanMessage
from uniagent import AgentFeatures, Budget, BudgetConfig, create_agent

from .answer import generate, select_fragments
from .config import Config
from .models import Chunk, FinalAnswer
from .sufficiency import SufficiencyVerifier
from .tools import PROCESSING_TOOLS, RETRIEVAL_TOOLS, run_fallback_pipeline
from .tracing import Tracer
from .workspace import get_workspace

RETRIEVAL_GOAL = (
    "为用户问题召回足量、高相关的候选知识片段。"
    "可用工具:query_understanding / question_rewrite / keyword_extraction / coarse_recall,"
    "由你自主决定调用顺序;若收到验证失败反馈,请换策略(改写问题/放宽过滤)重新召回。"
)

_PROCESSING_PROMPT = (
    "你是数据处理子智能体,自主规划清洗流程,把候选知识处理为可用于答案生成的素材。"
    "可用工具:analyze_data/clean_data/denoise_data/dedupe_data/structure_data/"
    "sort_data/apply_business_skill,由你决定取舍与顺序。"
    "若技能提示给出了业务类目的归一规则,套用 apply_business_skill。"
    "不要裁剪片段内容 —— 片段取舍由答案生成子智能体负责。"
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


class ProcessingSubAgent:
    """数据处理子智能体:ReAct 自主规划 + SkillMiddleware 业务技能包注入。"""

    def __init__(self, model: Any, tracer: Tracer, enable_skills: bool = True):
        self.tracer = tracer
        self._agent = create_agent(
            model=model, tools=PROCESSING_TOOLS,
            features=AgentFeatures(skill=enable_skills, goal_loop=False),
            system_prompt=_PROCESSING_PROMPT,
            name="processing_subagent",
        )

    async def run(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        ws = get_workspace()
        ws.stage = "processing"
        ws.data["chunks"] = list(chunks)
        before = [c.chunk_id for c in chunks]
        cats = sorted({c.category for c in chunks})
        cat_hint = f"业务类目:{cats[0]}" if len(cats) == 1 else f"类目分布:{cats}"
        try:
            # SkillMiddleware 由循环引擎执行;裸 agent 场景手动跑一次 before_agent
            state = {"messages": [HumanMessage(
                content=f"清洗候选知识,共 {len(chunks)} 条。{cat_hint}。用户问题:{query}")]}
            for mw in getattr(self._agent, "_uniagent_middleware", []):
                patch = await mw.before_agent(state)
                if patch:
                    state.update(patch)
            await self._agent.ainvoke(state)
        except Exception as exc:                        # noqa: BLE001
            self.tracer.log("processing", "agent_error", error=repr(exc))
        # ---- 保底护栏:子智能体无有效产出 → 确定性流水线 ----
        out = ws.data.get("chunks", [])
        if not out:
            self.tracer.log("processing", "fallback_pipeline",
                            reason="子智能体产出为空")
            ws.data["chunks"] = list(chunks)
            run_fallback_pipeline()
            out = ws.data["chunks"]
        self.tracer.log("processing", "snapshot",
                        before=before, after=[c.chunk_id for c in out])
        return out


class AnswerSubAgent:
    """答案生成子智能体:LLM 自主组织答案(素材取舍/结构/内联引用),
    逐句锚定校验为确定性护栏(硬事实锚定失败直接删句,不交给 LLM 裁量)。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer):
        self.model, self.cfg, self.tracer = model, cfg, tracer

    def run(self, query: str, chunks: List[Chunk], trace_id: str) -> FinalAnswer:
        materials = select_fragments(query, chunks)
        return generate(self.model, query, materials,
                        self.cfg, self.tracer, trace_id)
