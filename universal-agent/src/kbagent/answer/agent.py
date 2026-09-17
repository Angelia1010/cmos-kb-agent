# -*- coding: utf-8 -*-
"""答案生成子智能体。

自主:素材取舍与答案组织(内联引用由 LLM 生成)。
护栏:逐句锚定校验为确定性代码,硬事实锚定失败直接删句,不交给 LLM 裁量。
新增:对全部输入文档定位「能回答问题的原文逐字片段」(matched_fragments),
     与答案生成相互独立,仅供坐席溯源(详见 .locate)。
"""
from __future__ import annotations

from typing import Any, List

from ..shared.config import Config
from ..shared.models import Chunk, DocFragments, FinalAnswer
from ..shared.tracing import Tracer
from .generate import generate, select_fragments
from .locate import locate_fragments


class AnswerSubAgent:
    """答案生成子智能体。"""

    def __init__(self, model: Any, cfg: Config, tracer: Tracer):
        self.model = model
        self.cfg = cfg
        self.tracer = tracer

    def run(self, query: str, chunks: List[Chunk], trace_id: str) -> FinalAnswer:
        # 文档内证据片段定位:对**全部**输入文档逐篇定位能回答问题的原文片段。
        # 与答案生成相互独立(additive),仅产出 matched_fragments 供坐席溯源。
        matched: List[DocFragments] = []
        for c in chunks:
            df = locate_fragments(self.model, query, c)
            self.tracer.log("answer", "locate_fragments", chunk_id=c.chunk_id,
                            answerable=df.answerable,
                            fragment_count=len(df.fragments))
            matched.append(df)

        materials = select_fragments(query, chunks)
        ans = generate(self.model, query, materials,
                       self.cfg, self.tracer, trace_id)
        ans.matched_fragments = matched
        return ans
