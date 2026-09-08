# -*- coding: utf-8 -*-
"""主智能体 —— **编排好的**智能体:阶段顺序固定,不做自主规划。

并发说明:MainAgent 实例持有每次运行的 tracer,**不支持同一实例并发 run**;
并发服务请为每个请求创建 MainAgent(轻量),或在外层加锁。
已在事件循环中时请调用 arun(),run() 内部使用 asyncio.run 会与现有循环冲突。

职责(与原方案图一致):
  快速通道(缓存) → ① 检索子智能体 → ② 处理子智能体 → ③ 答案子智能体 → 降级兜底
自主性全部下放到三个子智能体内部;主智能体只持有编排、缓存、降级与全链路 trace。
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from .answer.agent import AnswerSubAgent
from .processing.agent import ProcessingSubAgent
from .retrieval.agent import RetrievalSubAgent
from .shared.cache import AnswerCache, normalize_query
from .shared.config import Config, DEFAULT_CONFIG
from .shared.knowledge_processing.bridge import retrieval_to_candidates
from .shared.models import FinalAnswer, RetrievalParams, SourceRef
from .shared.search import ESClient, build_dsl
from .shared.tracing import Tracer
from .shared.workspace import RunWorkspace, set_workspace


class MainAgent:
    def __init__(self, model: Any, es: ESClient,
                 cfg: Config = DEFAULT_CONFIG,
                 cache: Optional[AnswerCache] = None,
                 enable_skills: bool = True,
                 skill_dirs: Optional[List[str]] = None):
        from .shared.llm_bridge import ensure_judge_interface
        self.model, self.es, self.cfg = model, es, cfg
        # 判据/答案接口适配:普通 BaseChatModel 自动包 LLMBridge(审查修复)
        self.judge_model = ensure_judge_interface(model)
        self.cache = cache or AnswerCache(sim_threshold=cfg.cache_sim_threshold)
        self.tracer = Tracer()
        self._enable_skills = enable_skills
        if enable_skills:
            self._init_skills(skill_dirs or ["skills"])
        # 处理子智能体可复用(固定流水线,无请求级状态)
        self._processing = ProcessingSubAgent(model)

    @staticmethod
    def _init_skills(dirs: List[str]) -> None:
        """初始化 uniagent 全局技能注册表(业务技能包按关键词触发)。"""
        from uniagent.agents import config_factory
        from uniagent.skills.registry import SkillRegistry
        if config_factory._skill_registry is None:
            config_factory._skill_registry = SkillRegistry()
        # 幂等扫描:同一目录只扫一次,避免多实例场景重复注册告警
        scanned = getattr(config_factory, "_kb_scanned_dirs", set())
        new_dirs = [d for d in dirs if d not in scanned]
        if new_dirs:
            config_factory._skill_registry.scan(*new_dirs)
            config_factory._kb_scanned_dirs = scanned | set(new_dirs)

    # ------------------------------------------------------------------
    def run(self, query: str, region_code: str = "000") -> FinalAnswer:
        return asyncio.run(self.arun(query, region_code))

    async def arun(self, query: str, region_code: str = "000") -> FinalAnswer:
        """region_code 传省份名或区号(如 福建/591),缺省 "000" 全国。"""
        self.tracer = Tracer()
        self.tracer.log("run", "start", query=query, region_code=region_code)
        ws = RunWorkspace(query=query, cfg=self.cfg, es=self.es,
                          tracer=self.tracer)
        set_workspace(ws)
        try:
            # ---- 快速通道 ----
            nq = normalize_query(query)
            hit = self.cache.lookup(nq)
            self.tracer.log("cache", "hit" if hit else "miss")
            if hit:
                hit.from_cache = True
                hit.trace_id = self.tracer.trace_id
                hit.elapsed_ms = self.tracer.elapsed_ms()   # 命中耗时,而非原次耗时
                return hit

            # ---- ① 检索子智能体(直调一体化流水线,传省份信息) ----
            chunks = await RetrievalSubAgent(
                self.model, self.cfg, self.tracer,
                judge_model=self.judge_model).run(query, region_code)

            # ---- 阶段衔接:检索产物 → 处理阶段标准候选 ----
            ws.data["knowledge_candidates"] = retrieval_to_candidates(
                merged=ws.data.get("merged_results"), chunks=chunks)

            # ---- ② 数据处理子智能体(知识级固定流水线) ----
            await self._processing.run()
            processed = ws.data.get("processed_chunks") or []

            # ---- ③ 答案生成子智能体(自主组织 + 确定性锚定) ----
            # generate 使用标准 model.invoke,直接传原始模型即可
            ans = AnswerSubAgent(self.model, self.cfg, self.tracer).run(
                query, processed, self.tracer.trace_id)
            ans.elapsed_ms = self.tracer.elapsed_ms()
            self.cache.put(nq, ans)
            self.tracer.log("finalize", "done", elapsed_ms=ans.elapsed_ms)
            return ans
        except Exception as exc:                        # noqa: BLE001
            self.tracer.log("degrade", "triggered", error=repr(exc))
            return self._degrade(query, repr(exc))

    # ------------------------------------------------------------------
    def _degrade(self, query: str, reason: str) -> FinalAnswer:
        """降级:原始 query → 保守单轮关键词检索 → 返回 topN 原始片段。"""
        try:
            from .shared import lexicon
            params = RetrievalParams(keywords=lexicon.extract_keywords(query),
                                     retrieval_mode="keyword")
            hits = self.es.keyword_search(build_dsl(params, size=5))
        except Exception:                               # noqa: BLE001
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
