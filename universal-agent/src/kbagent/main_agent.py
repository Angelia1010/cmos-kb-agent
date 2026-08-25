# -*- coding: utf-8 -*-
"""主智能体 — **编排好的**智能体:阶段顺序固定,不做自主规划。

并发说明:MainAgent 实例不支持同一实例并发 run;
并发服务请为每个请求创建 MainAgent 实例。
已在事件循环中时请调用 arun(),run() 内部使用 asyncio.run 会与现有循环冲突。

职责:
  ① 检索子智能体 → ② 处理子智能体 → ③ 答案子智能体 → 降级兜底
自主性全部下放到三个子智能体内部;主智能体只持有编排、降级与全链路 trace。
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from uniagent.agents.config_factory import register_skill_directory

from .answer.agent import AnswerSubAgent
from .processing.agent import ProcessingSubAgent
from .retrieval.agent import RetrievalSubAgent
from .shared import lexicon
from .shared.config import Config, DEFAULT_CONFIG
from .shared.models import FinalAnswer, RetrievalParams, SourceRef
from .shared.search import ESClient, build_dsl
from .shared.tracing import Tracer
from .shared.workspace import RunWorkspace, set_workspace


class MainAgent:
    def __init__(self, model: Any, es: ESClient,
                 cfg: Config = DEFAULT_CONFIG,
                 enable_skills: bool = True,
                 skill_dirs: Optional[List[str]] = None):
        self.model = model
        self.es = es
        self.cfg = cfg
        self.tracer = Tracer()
        self._enable_skills = enable_skills
        if enable_skills:
            for d in (skill_dirs or ["skills"]):
                register_skill_directory(d)
        # 处理子智能体可复用(create_agent 成本高);检索/答案子智能体每次新建
        self._processing = ProcessingSubAgent(
            model, self.tracer, enable_skills=enable_skills)

    def run(self, query: str) -> FinalAnswer:
        return asyncio.run(self.arun(query))

    async def arun(self, query: str) -> FinalAnswer:
        self.tracer = Tracer()
        self._processing.tracer = self.tracer
        self.tracer.log("run", "start", query=query)
        ws = RunWorkspace(query=query, cfg=self.cfg, es=self.es,
                          tracer=self.tracer)
        set_workspace(ws)
        try:
            # ① 检索子智能体(自主规划,GoalLoop 护栏)
            chunks = await RetrievalSubAgent(
                self.model, self.cfg, self.tracer).run(query)

            # ② 数据处理子智能体(自主规划,技能包注入)
            processed = await self._processing.run(query, chunks)

            # ③ 答案生成子智能体(自主组织 + 确定性锚定)
            ans = AnswerSubAgent(self.model, self.cfg, self.tracer).run(
                query, processed, self.tracer.trace_id)
            ans.elapsed_ms = self.tracer.elapsed_ms()
            self.tracer.log("finalize", "done", elapsed_ms=ans.elapsed_ms)
            return ans
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("degrade", "triggered", error=repr(exc))
            return self._degrade(query, repr(exc))

    def _degrade(self, query: str, reason: str) -> FinalAnswer:
        """降级:原始 query → 保守单轮关键词检索 → 返回 topN 原始片段。"""
        try:
            params = RetrievalParams(
                keywords=lexicon.extract_keywords(query),
                retrieval_mode="keyword",
            )
            hits = self.es.keyword_search(build_dsl(params, size=5))
        except Exception:  # noqa: BLE001
            hits = []
        ans = FinalAnswer(
            trace_id=self.tracer.trace_id, query=query,
            business_explanation="(系统降级,以下为原始知识片段,请人工核实)",
            handling_suggestion="",
            sources=[SourceRef(c.chunk_id, c.doc_title, c.content, c.updated_at)
                     for c in hits],
            degraded=True, elapsed_ms=self.tracer.elapsed_ms())
        self.tracer.log("degrade", "done", reason=reason, hit_count=len(hits))
        return ans
