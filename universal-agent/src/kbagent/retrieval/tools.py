# -*- coding: utf-8 -*-
"""检索子智能体工具集。

安全边界:LLM 只传结构化参数,coarse_recall 内部走
RetrievalParams 清洗 + build_dsl 字段白名单,LLM 永远不接触 ES DSL。
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List

from langchain_core.tools import tool

from ..shared import lexicon
from ..shared.models import Chunk, RetrievalParams
from ..shared.search import build_dsl, merged_to_chunks, rrf_fuse
from ..shared.workspace import get_workspace

logger = logging.getLogger("kbagent.retrieval")


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

#使用es完成
@tool
def intergrate_all(query: str = "", region_code: str = "000",
                   timeout: int = 30) -> str:
    """生产一体化检索流水线:槽位提取→知识主索引召回→原子表拼接,一次调用直接产出候选片段。仅在接入生产 ngkm 检索(ProduceESClient)时可用;离线环境请改用 coarse_recall。region_code 支持区号或省份名(如 000/福建)。"""
    ws = get_workspace()
    full_recall = getattr(ws.es, "full_recall", None)
    if full_recall is None:
        logger.warning("intergrate_all: 当前检索后端 %s 无 full_recall,"
                       "回退关键词召回", type(ws.es).__name__)
        return _obs(error="当前检索后端不支持一体化流水线,请改用 coarse_recall")
    query = query or ws.query
    result = full_recall(query=query, region_code=region_code, timeout=timeout)
    merged = result.get("merged", []) if isinstance(result, dict) else []
    if not merged and isinstance(result, dict) and result.get("error"):
        logger.warning("intergrate_all 流水线报错: %s", result["error"])
        return _obs(error=result["error"])
    if not merged and isinstance(result, dict) and result.get("message"):
        logger.warning("intergrate_all 零召回: %s (keywords=%s)",
                       result["message"], result.get("keywords"))
    chunks = merged_to_chunks(merged)
    ws.data["chunks"] = chunks
    ws.data["original_query"] = query
    ws.data["region_code"] = region_code
    ws.data["merged_results"] = merged
    rnd = ws.data.get("recall_round", 0) + 1
    ws.data["recall_round"] = rnd
    ws.tracer.log(f"{ws.stage}.round{rnd}", "recall",
                  channel="intergrate_all", region_code=region_code,
                  titles=[c.doc_title for c in chunks],
                  scores=[c.score for c in chunks])
    logger.info("intergrate_all 完成: query=%r region=%s merged=%d → chunks=%d",
                query, region_code, len(merged), len(chunks))
    return _obs(recalled=len(chunks), titles=[c.doc_title for c in chunks],
                scores=[c.score for c in chunks])



RETRIEVAL_TOOLS = [query_understanding, question_rewrite,
                   keyword_extraction, coarse_recall, intergrate_all]
