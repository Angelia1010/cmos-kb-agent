# -*- coding: utf-8 -*-
"""检索子智能体工具集。

安全边界:LLM 只传结构化参数,coarse_recall 内部走
RetrievalParams 清洗 + build_dsl 字段白名单,LLM 永远不接触 ES DSL。
"""
from __future__ import annotations

import json
from typing import Dict, List

from langchain_core.tools import tool

from ..shared import lexicon
from ..shared.models import Chunk, RetrievalParams
from ..shared.search import build_dsl, rrf_fuse
from ..shared.workspace import get_workspace


def _obs(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False, default=str)


@tool
def query_understanding(query: str = "") -> str:
    """理解用户问题,识别子意图与业务类目。建议在检索开始时首先调用。参数 query 缺省时读取当前用户问题。"""
    ws = get_workspace()
    query = query or ws.query
    intents = lexicon.detect_categories(query) or [query[:4]]
    ws.data["intents"] = intents
    return _obs(intents=intents)


@tool
def question_rewrite(feedback: str = "") -> str:
    """根据上一轮检索失败的反馈改写检索问题(口语归一/补充意图词)。仅在收到验证失败反馈的重试轮调用。"""
    ws = get_workspace()
    rewritten = ws.query.replace("怎么", "如何").replace("办不了", "无法办理")
    for intent in ws.data.get("intents", []):
        if intent not in rewritten:
            rewritten += f" {intent}"
    ws.data["rewritten_query"] = rewritten
    ws.data["is_retry"] = True
    return _obs(rewritten_query=rewritten)


@tool
def keyword_extraction(expand: bool = True) -> str:
    """从(改写后的)问题中提取检索关键词,expand=True 时附带同义扩展词。"""
    ws = get_workspace()
    q = ws.data.get("rewritten_query") or ws.query
    keywords = lexicon.extract_keywords(q)
    expanded = lexicon.expand_terms(keywords) if expand else []
    ws.data["keywords"] = keywords
    ws.data["expanded_terms"] = expanded
    return _obs(keywords=keywords, expanded_terms=expanded)


@tool
def coarse_recall(relax_filters: bool = False, retrieval_mode: str = "hybrid") -> str:
    """执行混合召回(BM25+向量+RRF)。relax_filters=True 时放宽类目/状态过滤(重试轮建议开启)。retrieval_mode 可选 keyword/vector/hybrid。"""
    ws = get_workspace()
    keywords = ws.data.get("keywords") or []
    if not keywords:
        return _obs(error="缺少关键词,请先调用 keyword_extraction")

    filters: Dict[str, str] = {}
    if not relax_filters and not ws.data.get("is_retry"):
        cats = [i for i in ws.data.get("intents", [])
                if i in ("套餐", "宽带", "账单", "投诉")]
        if len(cats) == 1:
            filters = {"category": cats[0], "status": "在售"}

    params = RetrievalParams.from_llm_output({
        "keywords": keywords,
        "expanded_terms": ws.data.get("expanded_terms", []),
        "filters": filters,
        "boost_fields": {"title": 2.0, "content": 1.0},
        "retrieval_mode": retrieval_mode,
    })
    dsl = build_dsl(params, size=ws.cfg.recall_size)
    qtext = ws.data.get("rewritten_query") or ws.query
    khits = ws.es.keyword_search(dsl) if params.retrieval_mode in ("keyword", "hybrid") else []
    vhits = ws.es.vector_search(qtext, params.filters, ws.cfg.recall_size) \
        if params.retrieval_mode in ("vector", "hybrid") else []
    fused: List[Chunk] = rrf_fuse(khits, vhits, top_n=ws.cfg.fuse_top_n)

    ws.data["chunks"] = fused
    ws.data["last_params"] = params
    ws.data["last_dsl"] = dsl
    rnd = ws.data.get("recall_round", 0) + 1
    ws.data["recall_round"] = rnd
    ws.tracer.log(f"{ws.stage}.round{rnd}", "recall", dsl=dsl,
                  titles=[c.doc_title for c in fused],
                  scores=[c.score for c in fused])
    return _obs(recalled=len(fused), titles=[c.doc_title for c in fused],
                scores=[c.score for c in fused])


RETRIEVAL_TOOLS = [query_understanding, question_rewrite,
                   keyword_extraction, coarse_recall]
