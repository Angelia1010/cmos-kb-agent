# -*- coding: utf-8 -*-
"""答案生成子智能体。

自主:素材取舍与答案组织(内联引用由 LLM 生成)。
护栏:逐句锚定校验为确定性代码,硬事实锚定失败直接删句,不交给 LLM 裁量。
"""
from __future__ import annotations

from typing import Any, List

from ..shared.config import Config
from ..shared.models import Chunk, FinalAnswer
from ..shared.tracing import Tracer
from .generate import generate, select_fragments


class AnswerSubAgent:
    """答案生成子智能体。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer):
        self.model = model
        self.cfg = cfg
        self.tracer = tracer

    def run(self, query: str, chunks: List[Chunk], trace_id: str) -> FinalAnswer:
        materials = select_fragments(query, chunks)
        return generate(self.model, query, materials,
                        self.cfg, self.tracer, trace_id)
