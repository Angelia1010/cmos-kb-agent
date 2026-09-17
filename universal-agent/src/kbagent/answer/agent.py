# -*- coding: utf-8 -*-
"""答案生成子智能体。

自主:素材取舍与话术组织(LLM 输出 script + handling_suggestion)。
护栏:批量话术一致性校验为确定性代码,校验不过收紧 usability,不交给 LLM 裁量。
证据定位:对全部输入文档定位「能回答问题的原文逐字片段」+ 模型自评相关度,
         直接喂给 generate 组装 sources(keyFragment / relevance),零额外调用。

LLM 调用次数 = N(逐篇定位)+ 1(答案组织)+ 1(批量一致性校验)。
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
        # 文档内证据片段定位:对**全部**输入文档逐篇定位能回答问题的原文片段,
        # 并取模型自评相关度;结果供 generate 组装 sources(关键片段/相关度)。
        matched: List[DocFragments] = []
        for c in chunks:
            df = locate_fragments(self.model, query, c)
            self.tracer.log("answer", "locate_fragments", chunk_id=c.chunk_id,
                            answerable=df.answerable, relevance=df.relevance,
                            fragment_count=len(df.fragments))
            matched.append(df)

        materials = select_fragments(query, chunks)
        return generate(self.model, query, materials,
                        self.cfg, self.tracer, trace_id, matched=matched)
