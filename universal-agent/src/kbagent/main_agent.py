# -*- coding: utf-8 -*-
"""主智能体 —— **编排好的**智能体:阶段顺序固定,不做自主规划。

并发说明:MainAgent 实例持有每次运行的 tracer,**不支持同一实例并发 run**;
并发服务请为每个请求创建 MainAgent(轻量),或在外层加锁。
已在事件循环中时请调用 arun(),run() 内部使用 asyncio.run 会与现有循环冲突。

职责(更新后):
  快速通道(缓存) → ① 检索子智能体(内含 检索→处理→验证 Agent Loop)
                 → ② 答案子智能体 → 降级兜底
检索模块内部自主完成 Processing + Top3AnswerabilityVerifier;
MainAgent 不再单独调用 ProcessingSubAgent。
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from .answer.agent import AnswerSubAgent
from .retrieval.agent import RetrievalSubAgent
from .shared.cache import AnswerCache, normalize_query
from .shared.config import Config, DEFAULT_CONFIG
from .shared.models import (
    USABILITY_NOT,
    FinalAnswer,
    SourceRef,
    Usability,
)
from .shared.search import ESClient, kresult_to_chunks
from .shared.tracing import Tracer
from .shared.workspace import RunWorkspace, set_workspace


class MainAgent:
    """主智能体:快速通道 → 检索(含处理+验证) → 答案生成 → 降级兜底。

    检索子智能体内部已包含 Processing + Top3AnswerabilityVerifier,
    MainAgent 不再直接调度 ProcessingSubAgent。
    """

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
        """region_code 传省份名或区号(如 福建/591),缺省 "000" 全国。

        流程:
          快速通道(缓存) → 检索(内含 Processing+Verifier Agent Loop)
                        → 答案生成 → 降级兜底
        """
        self.tracer = Tracer()
        self.tracer.log("run", "start", query=query, region_code=region_code)
        ws = RunWorkspace(query=query, cfg=self.cfg, es=self.es,
                          tracer=self.tracer, model=self.model)
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

            # ---- ① 检索子智能体(内含 Agent Loop:检索→处理→验证) ----
            # RetrievalSubAgent 内部已完成:
            #   retrieval_to_candidates → ProcessingSubAgent.run
            #   → Top3AnswerabilityVerifier.verify
            # 产物写入 workspace: processed_chunks / top3_candidates
            chunks = await RetrievalSubAgent(
                self.model, self.cfg, self.tracer,
                judge_model=self.judge_model,
            ).run(query, region_code)

            # 取 processing 后的 chunks(已包含 Markdown 内容和 rerank 排名)
            processed = ws.data.get("processed_chunks") or chunks

            # ---- ② 答案生成子智能体(自主组织 + 确定性锚定) ----
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
            result = (self.es.keyword_search(query=query)
                      if hasattr(self.es, "keyword_search") else {})
            merged = result.get("merged", []) if isinstance(result, dict) else []
            hits = kresult_to_chunks(merged)[:5]
        except Exception:                               # noqa: BLE001
            hits = []
        # 降级来源:相关度按召回名次归一化(最相关一篇 = 100),无关键片段
        # (未经定位),content 直接给整篇原文供坐席人工核实。
        n = len(hits)
        sources = [
            SourceRef(chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title,
                      relevance=int(round((n - i) / n * 100)) if n else 0,
                      key_fragment="", content=c.content,
                      updated_at=c.updated_at, stale=False)
            for i, c in enumerate(hits)]
        ans = FinalAnswer(
            trace_id=self.tracer.trace_id, query=query,
            script="", handling_suggestion="",
            sources=sources,
            degraded=True, elapsed_ms=self.tracer.elapsed_ms(),
            usability=Usability(
                level=USABILITY_NOT,
                reasons=["系统降级兜底结果,未经答案生成与一致性校验,不可直接答复用户"]))
        self.tracer.log("degrade", "done", reason=reason, hit_count=len(hits))
        return ans