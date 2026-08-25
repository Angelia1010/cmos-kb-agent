# -*- coding: utf-8 -*-
"""数据处理子智能体 — ReAct 自主规划 + SkillMiddleware 业务技能包注入。

自主:清洗工具的取舍与顺序、业务 skill 的套用。
护栏:无产出时的确定性保底流水线;"不裁剪片段"写入工具约束。
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.messages import HumanMessage
from uniagent import AgentFeatures, create_agent

from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .tools import PROCESSING_TOOLS, run_fallback_pipeline

_PROCESSING_PROMPT = (
    "你是数据处理子智能体,自主规划清洗流程,把候选知识处理为可用于答案生成的素材。"
    "可用工具:analyze_data/clean_data/denoise_data/dedupe_data/structure_data/"
    "sort_data/apply_business_skill,由你决定取舍与顺序。"
    "若技能提示给出了业务类目的归一规则,套用 apply_business_skill。"
    "不要裁剪片段内容 —— 片段取舍由答案生成子智能体负责。"
)


class ProcessingSubAgent:
    """数据处理子智能体。"""

    def __init__(self, model: Any, tracer: Tracer, enable_skills: bool = True):
        self.tracer = tracer
        self._agent = create_agent(
            model=model,
            tools=PROCESSING_TOOLS,
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
            state = {"messages": [HumanMessage(
                content=f"清洗候选知识,共 {len(chunks)} 条。{cat_hint}。用户问题:{query}")]}
            for mw in getattr(self._agent, "_uniagent_middleware", []):
                patch = await mw.before_agent(state)
                if patch:
                    state.update(patch)
            await self._agent.ainvoke(state)
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("processing", "agent_error", error=repr(exc))
        out = ws.data.get("chunks", [])
        if not out:
            self.tracer.log("processing", "fallback_pipeline", reason="子智能体产出为空")
            ws.data["chunks"] = list(chunks)
            run_fallback_pipeline()
            out = ws.data["chunks"]
        self.tracer.log("processing", "snapshot",
                        before=before, after=[c.chunk_id for c in out])
        return out
