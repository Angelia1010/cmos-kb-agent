# -*- coding: utf-8 -*-
"""KB 智能体的工具集(LangChain @tool,供 uniagent 的 ReAct 子Agent 调用)。

对应原方案图两个子Agent 的 Tool List:
  检索: query_understanding / question_rewrite / keyword_extraction / coarse_recall
  处理: analyze_data / clean / denoise / dedupe / structure / sort / apply_business_skill

安全边界不变:LLM 只传结构化参数,coarse_recall 内部走
RetrievalParams 清洗 + build_dsl 字段白名单,LLM 永远不接触 ES DSL。
工具返回给 LLM 的是精简 JSON 字符串;Chunk 实体存放在 RunWorkspace。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from langchain_core.tools import tool

from . import lexicon
from .models import Chunk, RetrievalParams
from .search import build_dsl, rrf_fuse
from .workspace import get_workspace


def _obs(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False, default=str)


# ===========================================================================
# 检索子Agent 工具
# ===========================================================================

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

    params = RetrievalParams.from_llm_output({      # 类型清洗 + 值域夹紧
        "keywords": keywords,
        "expanded_terms": ws.data.get("expanded_terms", []),
        "filters": filters,
        "boost_fields": {"title": 2.0, "content": 1.0},
        "retrieval_mode": retrieval_mode,
    })
    dsl = build_dsl(params, size=ws.cfg.recall_size)          # 字段白名单
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


# ===========================================================================
# 数据处理子Agent 工具(无"截取片段" —— 职责已收归答案生成层)
# ===========================================================================

def _chunks() -> List[Chunk]:
    return get_workspace().data.get("chunks", [])


@tool
def analyze_data() -> str:
    """分析候选集的数量/类目分布/在售状态,为处理决策提供依据。建议首先调用。"""
    ws = get_workspace()
    chunks = _chunks()
    cats: Dict[str, int] = {}
    for c in chunks:
        cats[c.category] = cats.get(c.category, 0) + 1
    report = dict(count=len(chunks), categories=cats,
                  statuses=sorted({c.extra.get("status", "?") for c in chunks}))
    ws.data["analysis"] = report
    return _obs(**report)


@tool
def clean_data() -> str:
    """清洗:去除候选片段内容中的多余空白与控制字符。"""
    for c in _chunks():
        c.content = re.sub(r"\s+", " ", c.content).strip()
    return _obs(cleaned=len(_chunks()))


@tool
def denoise_data() -> str:
    """去噪:剔除下架/失效的知识片段。"""
    ws = get_workspace()
    before = len(_chunks())
    ws.data["chunks"] = [c for c in _chunks() if c.extra.get("status") != "下架"]
    return _obs(before=before, after=len(_chunks()))


@tool
def dedupe_data() -> str:
    """去重:同 chunk_id 只保留得分最高的一条。"""
    ws = get_workspace()
    before = len(_chunks())
    seen: Dict[str, Chunk] = {}
    for c in sorted(_chunks(), key=lambda x: x.score, reverse=True):
        seen.setdefault(c.chunk_id, c)
    ws.data["chunks"] = list(seen.values())
    return _obs(before=before, after=len(_chunks()))


@tool
def structure_data() -> str:
    """结构化:抽取金额/时限等业务字段挂到片段元数据(只增不删)。"""
    n = 0
    for c in _chunks():
        fees = re.findall(r"(\d+(?:\.\d+)?)元", c.content)
        if fees:
            c.extra["fees_yuan"] = fees
            n += 1
        hours = re.findall(r"(\d+)小时", c.content)
        if hours:
            c.extra["deadlines_hours"] = hours
    return _obs(structured=n)


@tool
def sort_data() -> str:
    """排序:得分优先,同分知识按更新时间新者优先。"""
    ws = get_workspace()
    ws.data["chunks"] = sorted(_chunks(),
                               key=lambda c: (c.score, c.updated_at), reverse=True)
    return _obs(sorted=len(_chunks()))


@tool
def apply_business_skill(category: str) -> str:
    """按业务类目做字段归一与业务噪音剔除(不裁剪内容)。category 取值: 套餐|宽带|账单|投诉。技能提示中会给出该类目的具体归一规则。"""
    if category not in ("套餐", "宽带", "账单", "投诉"):
        return _obs(error="category 必须是 套餐|宽带|账单|投诉 之一")
    for c in _chunks():
        if category == "套餐":
            c.content = c.content.replace("月费", "月费(每月)")
        c.extra["skill_applied"] = category
    return _obs(applied=category, count=len(_chunks()))


PROCESSING_TOOLS = [analyze_data, clean_data, denoise_data, dedupe_data,
                    structure_data, sort_data, apply_business_skill]


def run_fallback_pipeline() -> None:
    """确定性保底流水线:子Agent 无有效产出时由 pipeline 调用。"""
    clean_data.func()
    denoise_data.func()
    dedupe_data.func()
    structure_data.func()
    sort_data.func()
