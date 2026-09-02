"""知识级 Processing 固定流水线入口。"""
from __future__ import annotations

from typing import Any, List

from ..shared.knowledge_processing.models import KnowledgeProcessingOptions, ProcessedKnowledge
from ..shared.workspace import get_workspace
from .output import top3_to_processed_chunks
from .tools import build_knowledge_processing_tools


class KnowledgeProcessingOrchestrator:
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
