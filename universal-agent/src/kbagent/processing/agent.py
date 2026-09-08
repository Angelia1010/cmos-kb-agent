# -*- coding: utf-8 -*-
"""数据处理子智能体 — 知识级固定流水线,零 LLM 编排。

``ProcessingSubAgent`` 固定执行 analyze → filter → build_markdown → rerank,
面向知识级候选(工作区 ``knowledge_candidates``),主链路与
processing_service 共用;``KnowledgeProcessingOrchestrator`` 为其向后兼容别名。
早期"ReAct 自主规划 + SkillMiddleware"形态已注释归档于下方。
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.messages import HumanMessage
from uniagent import AgentFeatures, create_agent

from ..shared.knowledge_processing.models import (
    KnowledgeProcessingOptions,
    ProcessedKnowledge,
)
from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .output import top3_to_processed_chunks
from .tools import (
    PROCESSING_TOOLS,
    build_knowledge_processing_tools,
    run_fallback_pipeline,
)

_PROCESSING_PROMPT = (
    "你是数据处理子智能体,自主规划清洗流程,把候选知识处理为可用于答案生成的素材。"
    "可用工具:analyze_data/clean_data/denoise_data/dedupe_data/structure_data/"
    "sort_data/apply_business_skill,由你决定取舍与顺序。"
    "若技能提示给出了业务类目的归一规则,套用 apply_business_skill。"
    "不要裁剪片段内容 —— 片段取舍由答案生成子智能体负责。"
)


# class ProcessingSubAgent:
#     """数据处理子智能体:ReAct 自主规划 + SkillMiddleware 业务技能包注入。"""

#     def __init__(self, model: Any, tracer: Tracer, enable_skills: bool = True):
#         self.tracer = tracer
#         self._agent = create_agent(
#             model=model, tools=PROCESSING_TOOLS,
#             features=AgentFeatures(skill=enable_skills, goal_loop=False),
#             system_prompt=_PROCESSING_PROMPT,
#             name="processing_subagent",
#         )

#     async def run(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
#         ws = get_workspace()
#         ws.stage = "processing"
#         ws.data["chunks"] = list(chunks)
#         before = [c.chunk_id for c in chunks]
#         cats = sorted({c.category for c in chunks})
#         cat_hint = f"业务类目:{cats[0]}" if len(cats) == 1 else f"类目分布:{cats}"
#         try:
#             # SkillMiddleware 由循环引擎执行;裸 agent 场景手动跑一次 before_agent
#             state = {"messages": [HumanMessage(
#                 content=f"清洗候选知识,共 {len(chunks)} 条。{cat_hint}。用户问题:{query}")]}
#             for mw in getattr(self._agent, "_uniagent_middleware", []):
#                 patch = await mw.before_agent(state)
#                 if patch:
#                     state.update(patch)
#             await self._agent.ainvoke(state)
#         except Exception as exc:                        # noqa: BLE001
#             self.tracer.log("processing", "agent_error", error=repr(exc))
#         # ---- 保底护栏:子智能体无有效产出 → 确定性流水线 ----
#         out = ws.data.get("chunks", [])
#         if not out:
#             self.tracer.log("processing", "fallback_pipeline",
#                             reason="子智能体产出为空")
#             ws.data["chunks"] = list(chunks)
#             run_fallback_pipeline()
#             out = ws.data["chunks"]
#         self.tracer.log("processing", "snapshot",
#                         before=before, after=[c.chunk_id for c in out])
#         return out


class ProcessingSubAgent:
    """固定执行 analyze → filter → build_markdown → rerank。"""

    _STEPS = (
        ("analyze_knowledge_candidates", "normalized_knowledge_candidates"),
        ("filter_knowledge_candidates", "filtered_knowledge_candidates"),
        ("build_knowledge_markdown", "processed_knowledge_candidates"),
        ("rerank_knowledge_candidates", "top3_candidates"),
    )

    def __init__(
        self,
        model: Any,
        options: KnowledgeProcessingOptions | None = None,
    ) -> None:
        self.model = model
        self.options = options or KnowledgeProcessingOptions()
        tools = build_knowledge_processing_tools(model, self.options)
        self.tools = {tool.name: tool for tool in tools}

    async def run(self) -> List[ProcessedKnowledge]:
        ws = get_workspace()
        ws.stage = "processing"
        # 防止同一个 Workspace 重跑或本轮异常时误用上一次的输出。
        ws.data.pop("processed_chunks", None)
        for tool_name, artifact_key in self._STEPS:
            tool = self.tools[tool_name]
            await tool.ainvoke({})
            artifact = ws.data.get(artifact_key)
            if not isinstance(artifact, list):
                raise RuntimeError(f"{tool_name} 未产生有效工作区产物 {artifact_key}")
            ws.tracer.log(
                "processing.knowledge",
                "orchestrator_step",
                step=tool_name,
                artifact=artifact_key,
                count=len(artifact),
            )
        top3 = ws.data["top3_candidates"]
        ws.data["processed_chunks"] = top3_to_processed_chunks(top3)
        ws.tracer.log(
            "processing.knowledge",
            "processed_chunks_adapted",
            count=len(ws.data["processed_chunks"]),
        )
        return list(top3)


# 向后兼容别名:原 KnowledgeProcessingOrchestrator(固定流水线)已更名为主链路
# 的 ProcessingSubAgent;processing_service / 演示脚本 / 测试仍按旧名引用。
KnowledgeProcessingOrchestrator = ProcessingSubAgent
