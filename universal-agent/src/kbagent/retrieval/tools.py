# -*- coding: utf-8 -*-
"""检索子智能体工具集。

安全边界:LLM 只传结构化参数,intergrate_all / keyword_recall / vector_recall
内部经 keyword_search / vector_search 流水线召回,LLM 永远不接触 ES DSL。
coarse_recall 作为降级兜底工具,当主路径报错时由 agent 调用。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.tools import tool

from ..shared import lexicon
from ..shared.models import Chunk
from ..shared.search import kresult_to_chunks, vresult_to_chunks,get_kid_score
from ..shared.workspace import get_workspace

logger = logging.getLogger("kbagent.retrieval")


def _obs(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False, default=str)

@tool
def intergrate_all(query: str = "", region_code: str = "000",
                   timeout: int = 30, vector_mode: str = "new") -> str:
    """生产一体化检索流水线:同时调 keyword(槽位提取→知识主索引→原子表)与
    vector(在线 embedding)两路召回,跨路去重后产出最终候选片段。
    region_code 支持区号或省份名(如 000/福建)。
    """
    ws = get_workspace()
    query = query or ws.query
    keyword_search = getattr(ws.es, "keyword_search", None)
    vector_search = getattr(ws.es, "vector_search", None)

    kchunks: List[Chunk] = []
    kresult: dict = {}
    kerror: str = ""
    if keyword_search is not None:
        kresult = keyword_search(query=query, region_code=region_code, timeout=timeout)
        kmerged = kresult.get("merged", []) if isinstance(kresult, dict) else []
        if not kmerged and isinstance(kresult, dict) and kresult.get("error"):
            kerror = kresult["error"]
            logger.warning("intergrate_all keyword 流水线报错: %s", kerror)
        else:
            if not kmerged and isinstance(kresult, dict) and kresult.get("message"):
                logger.warning("intergrate_all keyword 零召回: %s (keywords=%s)",
                               kresult["message"], kresult.get("keywords"))
            kchunks = kresult_to_chunks(kresult)
    else:
        kerror = f"当前检索后端 {type(ws.es).__name__} 无 keyword_search"
        logger.warning("intergrate_all: %s,跳过 keyword 通道", kerror)

    vchunks: List[Chunk] = []
    vresult: Any = None
    verror: str = ""
    if vector_search is not None:
        try:
            vresult = vector_search(query, {"region": region_code} if region_code else {},
                                    ws.cfg.recall_size, vector_mode=vector_mode)
            vchunks = vresult_to_chunks(vresult)
        except Exception as exc:  # noqa: BLE001
            verror = repr(exc)
            logger.warning("intergrate_all vector 通道异常,降级为空: %r", exc)
    else:
        verror = f"当前检索后端 {type(ws.es).__name__} 无 vector_search"
        logger.warning("intergrate_all: %s,跳过 vector 通道", verror)

    seen: set = set()
    merged_chunks: List[Chunk] = []
    for c in kchunks + vchunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            merged_chunks.append(c)

    keyword_kid = list((kresult.get("knowledge_ids") if isinstance(kresult, dict) else []) or [])
    vector_kid = [c.doc_id for c in vchunks if c.doc_id and c.doc_id != "unknown"]
    kid_scores = get_kid_score(keyword_kid, vector_kid)
    sorted_kids = sorted(kid_scores.keys(), key=lambda k: kid_scores[k], reverse=True)
    kid_rank = {kid: rank for rank, kid in enumerate(sorted_kids)}
    merged_chunks.sort(key=lambda c: kid_rank.get(c.doc_id, len(sorted_kids)))

    chunks = merged_chunks
    logger.info("intergrate_all 合并: keyword=%d vector=%d → 去重后=%d",
                len(kchunks), len(vchunks), len(chunks))
    ws.data["chunks"] = chunks
    ws.data["original_query"] = query
    ws.data["region_code"] = region_code
    ws.data["keywords"] = list((kresult.get("keywords") if isinstance(kresult, dict) else []) or [])
    ws.data["merged_results"] = kresult.get("merged", []) if isinstance(kresult, dict) else []
    ws.data["keyword_chunks"] = kchunks
    ws.data["keyword_kid"] = keyword_kid
    ws.data["vector_results"] = vresult
    ws.data["vector_chunks"] = vchunks
    ws.data["vector_kid"] = vector_kid
    ws.data["kid_scores"] = kid_scores
    ws.data["ranked_kids"] = sorted_kids
    ws.data["example"] = kresult.get("example", {}) if isinstance(kresult, dict) else {}
    rnd = ws.data.get("recall_round", 0) + 1
    ws.data["recall_round"] = rnd
    ws.tracer.log(f"{ws.stage}.round{rnd}", "recall",
                  channel="intergrate_all", region_code=region_code,
                  keyword_count=len(kchunks), vector_count=len(vchunks),
                  merged_count=len(chunks),
                  titles=[c.doc_title for c in chunks],
                  scores=[c.score for c in chunks])
    logger.info("intergrate_all 完成: query=%r region=%s keyword=%d vector=%d → 去重=%d",
                query, region_code, len(kchunks), len(vchunks), len(chunks))

    if not chunks:
        errors = [e for e in (kerror, verror) if e]
        return _obs(error="keyword+vector 双路零召回" + (f"(原因: {'; '.join(errors)})" if errors else ""))
    return _obs(recalled=len(chunks), titles=[c.doc_title for c in chunks],
                scores=[c.score for c in chunks])


@tool
def vector_recall(query: str = "", region_code: str = "000", vector_mode: str = "new") -> str:
    """纯向量召回:直接走在线知识 embedding 向量检索服务,按语义相似度返回候选片段,
    不经过关键词/槽位提取。region_code 支持区号或省份名(如 000/福建),经 provinceId
    下推到向量服务。后端不支持向量检索时返回 error,由 agent 走兜底降级。
    """
    ws = get_workspace()
    vector_search = getattr(ws.es, "vector_search", None)
    if vector_search is None:
        logger.warning("vector_recall: 当前检索后端 %s 无 vector_search,"
                       "回退关键词召回", type(ws.es).__name__)
        return _obs(error="当前检索后端不支持向量召回,请改用 coarse_recall")
    query = query or ws.query
    region_code = region_code or ws.region_code
    try:
        result = vector_search(query, region_code, vector_mode=vector_mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vector_recall 异常: query=%r region=%s err=%r", query, region_code, exc)
        return _obs(error=f"向量召回异常: {exc!r}")
    chunks = vresult_to_chunks(result)
    for rank, chunk in enumerate(chunks):
        chunk.position["rank"] = rank
    ws.data["chunks"] = chunks
    ws.data["original_query"] = query
    ws.data["region_code"] = region_code
    ws.data["vector_results"] = result
    rnd = ws.data.get("recall_round", 0) + 1
    ws.data["recall_round"] = rnd
    ws.tracer.log(f"{ws.stage}.round{rnd}", "recall",
                  channel="vector_recall", region_code=region_code,
                  titles=[c.doc_title for c in chunks],
                  scores=[c.score for c in chunks])
    logger.info("vector_recall 完成: query=%r region=%s size=%d → chunks=%d",
                query, region_code, size, len(chunks))
    if not chunks:
        return _obs(error="vector_recall 零召回")
    return _obs(recalled=len(chunks), titles=[c.doc_title for c in chunks],
                scores=[c.score for c in chunks])


@tool
def keyword_recall(query: str = "", region_code: str = "000",
                   timeout: int = 30) -> str:
    """关键词一体化召回:走 keyword_search 流水线(槽位提取→知识主索引→原子表拼接)。
    region_code 支持区号或省份名(如 000/福建)。
    """
    ws = get_workspace()
    keyword_search = getattr(ws.es, "keyword_search", None)
    if keyword_search is None:
        logger.warning("keyword_recall: 当前检索后端 %s 无 keyword_search,"
                       "回退关键词召回", type(ws.es).__name__)
        return _obs(error="当前检索后端不支持一体化流水线,请改用 coarse_recall")
    query = query or ws.query
    result = keyword_search(query=query, region_code=region_code, timeout=timeout)
    merged = result.get("merged", []) if isinstance(result, dict) else []
    if not merged and isinstance(result, dict) and result.get("error"):
        logger.warning("keyword_recall 流水线报错: %s", result["error"])
        return _obs(error=result["error"])
    if not merged and isinstance(result, dict) and result.get("message"):
        logger.warning("keyword_recall 零召回: %s (keywords=%s)",
                       result["message"], result.get("keywords"))
    chunks = kresult_to_chunks(merged)
    logger.info("keyword_recall chunks=%d", len(chunks))
    ws.data["chunks"] = chunks
    ws.data["original_query"] = query
    ws.data["region_code"] = region_code
    ws.data["merged_results"] = merged
    ws.data["example"] = result.get("example", {})
    rnd = ws.data.get("recall_round", 0) + 1
    ws.data["recall_round"] = rnd
    ws.tracer.log(f"{ws.stage}.round{rnd}", "recall",
                  channel="keyword_recall", region_code=region_code,
                  titles=[c.doc_title for c in chunks],
                  scores=[c.score for c in chunks])
    logger.info("keyword_recall 完成: query=%r region=%s merged=%d → chunks=%d",
                query, region_code, len(merged), len(chunks))
    if not chunks:
        return _obs(error="keyword_recall 零召回")
    return _obs(recalled=len(chunks), titles=[c.doc_title for c in chunks],
                scores=[c.score for c in chunks])


RETRIEVAL_TOOLS = [intergrate_all, keyword_recall, vector_recall]
