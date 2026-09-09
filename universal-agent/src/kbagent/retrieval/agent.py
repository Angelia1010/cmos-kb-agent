# -*- coding: utf-8 -*-
"""检索子智能体(三个直调形态,均不走 agent loop):

- ``RetrievalSubAgent`` — 直调 ``intergrate_all`` 工具:同时调 keyword 与
  vector 两路召回,跨路去重后产出候选片段。region_code 传省份名或区号
  (福建/591/000,由 shared.search 内部归一为区号)。
- ``RetrievalKeywordSubAgent`` — 直调 ``keyword_recall`` 工具:只走 keyword
  一体化流水线(槽位提取 → 知识主索引 → 原子表拼接)。
- ``RetrievalVectorSubAgent`` — 直调 ``vector_recall`` 工具:只走 vector
  在线 embedding 向量召回。
三个形态全程零 LLM 参与;降级护栏统一为:主路径报错 → 兜底 keyword_extraction
+ coarse_recall;兜底仍零召回且报错 → 显式失败,交由主智能体降级。
"""
from __future__ import annotations

from typing import Any, List

from uniagent import AgentFeatures, Budget, BudgetConfig, create_agent

from ..shared.config import Config
from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .sufficiency import SufficiencyVerifier
from .tools import (
    RETRIEVAL_TOOLS,
    intergrate_all,
    keyword_recall,
    vector_recall,
)

RETRIEVAL_GOAL = (
    "为用户问题召回足量、高相关的候选知识片段。"
    "可用工具: intergrate_all, keyword_recall, vector_recall,由你自主决定调用顺序。"
)

# class RetrievalSubAgent:
#     """检索候选知识子智能体。"""

#     def __init__(self, model: Any, cfg: Config, tracer: Tracer):
#         self.model = model
#         self.cfg = cfg
#         self.tracer = tracer
#         self._verifier = SufficiencyVerifier()

#     async def run(self, query: str) -> List[Chunk]:
#         ws = get_workspace()
#         ws.stage = "retrieval"
#         loop = create_agent(
#             model=self.model,
#             tools=RETRIEVAL_TOOLS,
#             features=AgentFeatures(skill=False),
#             system_prompt="你是候选知识检索子智能体,直接调用intergrate_all工具返回检索结果。",
#             goal=RETRIEVAL_GOAL,
#             verifier=self._verifier,
#             budget=Budget(config=BudgetConfig(
#                 max_iterations=self.cfg.max_retrieval_rounds,
#                 max_time_seconds=self.cfg.budget["retrieval_total"] / 1000.0,
#             )),
#             name="retrieval_subagent",
#         )
#         result = await loop.run(
#             input_messages=[{"role": "user", "content": f"用户问题:{query}"}],
#             thread_id=self.tracer.trace_id,
#         )
#         self.tracer.log("retrieval", "loop_result",
#                         success=result.success, iterations=result.iterations,
#                         reason=result.reason)
#         chunks: List[Chunk] = ws.data.get("chunks", [])
#         if not result.success:
#             if not chunks and str(result.reason).startswith("错误"):
#                 raise RuntimeError(f"检索子智能体失败: {result.reason}")
#             self.tracer.log("retrieval", "exit_with_best",
#                             reason=result.reason, count=len(chunks))
#         return chunks

class RetrievalSubAgent:
    """检索候选知识子智能体:直调 intergrate_all(keyword+vector 双路去重)。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer,
                 judge_model: Any = None):
        # judge_model 仅为调用方兼容保留:直调形态下不参与推理
        self.model, self.cfg, self.tracer = model, cfg, tracer

    async def run(self, query: str, region_code: str = "000") -> List[Chunk]:
        """固定流水线:intergrate_all 一次产出候选片段。

        region_code 传省份名或区号(如 福建/591),缺省 "000" 全国。
        """
        ws = get_workspace()
        ws.stage = "retrieval"
        intergrate_all.func(query=query,region_code=region_code)
        chunks: List[Chunk] = ws.data.get("chunks", [])
        self.tracer.log("retrieval", "done", region_code=region_code,
                        count=len(chunks))
        return chunks


class RetrievalKeywordSubAgent:
    """关键词召回子智能体:直调 keyword_recall(仅 keyword 一体化流水线)。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer,
                 judge_model: Any = None):
        # judge_model 仅为调用方兼容保留:直调形态下不参与推理
        self.model, self.cfg, self.tracer = model, cfg, tracer

    async def run(self, query: str, region_code: str = "000") -> List[Chunk]:
        """固定流水线:keyword_recall 一次产出候选片段。

        region_code 传省份名或区号(如 福建/591),缺省 "000" 全国。
        """
        ws = get_workspace()
        ws.stage = "retrieval"
        keyword_recall.func(query=query,region_code=region_code)
        chunks: List[Chunk] = ws.data.get("chunks", [])
        self.tracer.log("retrieval", "done", region_code=region_code,
                        count=len(chunks))
        return chunks


class RetrievalVectorSubAgent:
    """向量召回子智能体:直调 vector_recall(传省份信息),不走 agent loop。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer,
                 judge_model: Any = None):
        # judge_model 仅为调用方兼容保留:去除循环后不再有充分性判定
        self.model, self.cfg, self.tracer = model, cfg, tracer

    async def run(self, query: str, region_code: str = "000") -> List[Chunk]:
        """固定流水线:vector_recall 一次产出候选片段。

        region_code 传省份名或区号(如 福建/591),缺省 "000" 全国。
        """
        ws = get_workspace()
        ws.stage = "retrieval"
        vector_recall.func(query=query,region_code=region_code)
        chunks: List[Chunk] = ws.data.get("chunks", [])
        self.tracer.log("retrieval", "done", region_code=region_code,
                        count=len(chunks))
        return chunks
