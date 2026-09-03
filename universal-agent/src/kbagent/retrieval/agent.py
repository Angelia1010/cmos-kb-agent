# -*- coding: utf-8 -*-
"""检索子智能体 — 直调生产一体化检索的固定流水线。

取代早期"ReAct 自主规划 + GoalLoop 循环护栏"形态:一次性直接调用
tools.py 的 intergrate_all 工具(槽位提取 → 知识主索引召回 → 原子表拼接),
并传入省份信息 region_code(省份名或区号均可,如 福建/591/000,
由 shared.search 内部归一为区号);全程零 LLM 参与,不走 agent loop。

降级护栏:检索后端不支持一体化流水线(如离线 MockESClient)或
流水线报错时,确定性退化为 keyword_extraction → coarse_recall;
兜底仍零召回且报错 → 显式失败,交由主智能体降级。
"""
from __future__ import annotations

import json
from typing import Any, List

from ..shared.config import Config
from ..shared.models import Chunk
from ..shared.tracing import Tracer
from ..shared.workspace import get_workspace
from .tools import coarse_recall, intergrate_all, keyword_extraction


class RetrievalSubAgent:
    """检索候选知识子智能体:直调 intergrate_all(传省份信息),不走 agent loop。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer,
                 judge_model: Any = None):
        # judge_model 仅为调用方兼容保留:去除循环后不再有充分性判定
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
