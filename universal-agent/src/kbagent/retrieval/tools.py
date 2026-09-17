# -*- coding: utf-8 -*-
"""检索子智能体工具集。

安全边界:LLM 只传结构化参数,intergrate_all / keyword_recall / vector_recall
内部经 keyword_search / vector_search 流水线召回,LLM 永远不接触 ES DSL。
coarse_recall 作为降级兜底工具,当主路径报错时由 agent 调用。
query_rewrite 使用 LLM 对用户问题进行语义改写,产出利于检索的等价查询。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from ..shared import lexicon
from ..shared.models import Chunk
from ..shared.search import kresult_to_chunks, vresult_to_chunks,get_kid_score
from ..shared.workspace import get_workspace
# 原_QUERY_REWRITE_SYSTEM(查询重写,已弃用)
# from .prompt import _QUERY_REWRITE_SYSTEM
from .prompt import _KEYWORD_REWRITE_SYSTEM

logger = logging.getLogger("kbagent.retrieval")


def _obs(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False, default=str)

def _parse_json(raw: str) -> Dict[str, Any]:
    """解析 LLM 输出的 JSON;剥离 markdown 代码围栏,失败返回空字典。"""
    raw = re.sub(r"^```(json)?|```$", "", str(raw).strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning("query_rewrite LLM 输出不是 JSON 对象: %s", raw[:500])
            return {}
        return data
    except json.JSONDecodeError as exc:
        logger.warning("query_rewrite LLM 输出 JSON 解析失败 err=%s 原始输出=%s",
                       exc, raw[:500])
        return {}


def _invoke_json(model: Any, system: str, user: str) -> Dict[str, Any]:
    """用标准 model.invoke 调用 LLM 并解析 JSON 响应;记录耗时与原始输出。"""
    t0 = time.time()
    resp = model.invoke([SystemMessage(content=system),
                         HumanMessage(content=user)])
    elapsed = time.time() - t0
    raw = str(getattr(resp, "content", resp))
    logger.info("query_rewrite LLM 返回 耗时%.1fs 长度=%d 内容=%s",
                elapsed, len(raw), raw[:500].replace("\n", " "))
    return _parse_json(raw)


# 原 query_rewrite(查询重写,已弃用,改为关键词重写)
# @tool
# def query_rewrite(query: str = "", last_query: str = "",
#                   last_keywords: str = "",
#                   last_recall_summary: str = "") -> str:
#     """使用大模型对用户问题进行语义改写,产出 1-3 个语义等价但更利于
#     知识库检索的改写查询。首轮 intergrate_all 召回不足时调用,再用改写后的
#     查询调用 intergrate_all 重新召回。
#     可传入上一轮上下文(last_query/last_keywords/last_recall_summary)辅助改写;
#     未传时自动从工作区读取上一轮数据。
#     """
#     logger.info("[TOOL_CALL] query_rewrite 参数: query=%r last_query=%r last_keywords=%r last_recall_summary=%s",
#                 query, last_query, last_keywords, last_recall_summary)
#     ws = get_workspace()
#     model = getattr(ws, "model", None)
#     if model is None:
#         logger.info("[TOOL_RETURN] query_rewrite 返回: error=workspace 未注入 model")
#         return _obs(error="workspace 未注入 model,无法执行 query_rewrite")
#     query = query or ws.query
#     last_query = last_query or str(ws.data.get("original_query", ""))
#     last_keywords = last_keywords or ", ".join(
#         ws.data.get("keywords", []) or [])
#     if not last_recall_summary:
#         prev_chunks = ws.data.get("chunks", []) or []
#         if prev_chunks:
#             last_recall_summary = "; ".join(
#                 f"{c.doc_title}({c.score:.3f})" for c in prev_chunks[:10])
#         else:
#             last_recall_summary = "(无上一轮召回结果)"
#     context_lines = [
#         f"当前查询:{query}",
#         f"上一轮查询:{last_query or '(无)'}",
#         f"上一轮关键词:{last_keywords or '(无)'}",
#         f"上一轮召回摘要:{last_recall_summary}",
#     ]
#     user_msg = "\n".join(context_lines)
#     try:
#         data = _invoke_json(model, _QUERY_REWRITE_SYSTEM, user_msg)
#     except Exception as exc:  # noqa: BLE001
#         logger.warning("query_rewrite LLM 调用异常: %r", exc)
#         logger.info("[TOOL_RETURN] query_rewrite 返回: error=LLM 调用异常 %r", exc)
#         return _obs(error=f"LLM 调用异常: {exc!r}")
#     rewrites = data.get("rewrites", [])
#     if not isinstance(rewrites, list) or not rewrites:
#         logger.warning("query_rewrite 未产出有效改写, 原始=%r", data)
#         logger.info("[TOOL_RETURN] query_rewrite 返回: error=LLM 未产出有效改写查询")
#         return _obs(error="LLM 未产出有效改写查询")
#     rewrites = [str(r) for r in rewrites if r]
#     ws.data["rewritten_queries"] = rewrites
#     ws.data["original_query"] = query
#     rnd = ws.data.get("recall_round", 0)
#     ws.tracer.log(f"{ws.stage}.round{rnd}", "query_rewrite",
#                   original=query, last_query=last_query,
#                   last_keywords=last_keywords, rewrites=rewrites)
#     # logger.info("query_rewrite 完成: query=%r last_query=%r → rewrites=%s",
#     #             query, last_query, rewrites)
#     logger.info("[TOOL_RETURN] query_rewrite 返回: rewrites=%s", rewrites)
#     return _obs(rewrites=rewrites)
@tool
def query_rewrite(last_query: str = "",
                  last_keywords: str = "",
                  last_recall_summary: str = "") -> str:
    """使用大模型对上一轮检索关键词进行重写,产出 1-3 组语义等价但更利于
    知识库检索的改写关键词。首轮 intergrate_all 召回后调用,再用改写后的
    关键词作为 keywords 参数调用 intergrate_all 重新召回。
    入参:last_keywords(上一轮关键词)、last_query(上一轮查询)、
    last_recall_summary(上一轮召回摘要);未传时自动从工作区读取上一轮数据。
    """
    logger.info("[TOOL_CALL] query_rewrite 参数: last_query=%r last_keywords=%r last_recall_summary=%s",
                last_query, last_keywords, last_recall_summary)
    ws = get_workspace()
    model = getattr(ws, "model", None)
    if model is None:
        logger.info("[TOOL_RETURN] query_rewrite 返回: error=workspace 未注入 model")
        return _obs(error="workspace 未注入 model,无法执行 query_rewrite")
    last_query = last_query or str(ws.data.get("original_query", ""))
    last_keywords = last_keywords or ", ".join(
        ws.data.get("keywords", []) or [])
    if not last_recall_summary:
        prev_chunks = ws.data.get("chunks", []) or []
        if prev_chunks:
            last_recall_summary = "; ".join(
                f"{c.doc_title}({c.score:.3f})" for c in prev_chunks[:10])
        else:
            last_recall_summary = "(无上一轮召回结果)"
    context_lines = [
        f"上一轮查询:{last_query or '(无)'}",
        f"上一轮关键词:{last_keywords or '(无)'}",
        f"上一轮召回摘要:{last_recall_summary}",
    ]
    user_msg = "\n".join(context_lines)
    try:
        data = _invoke_json(model, _KEYWORD_REWRITE_SYSTEM, user_msg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("query_rewrite LLM 调用异常: %r", exc)
        logger.info("[TOOL_RETURN] query_rewrite 返回: error=LLM 调用异常 %r", exc)
        return _obs(error=f"LLM 调用异常: {exc!r}")
    rewritten_keywords = data.get("rewritten_keywords", [])
    if not isinstance(rewritten_keywords, list) or not rewritten_keywords:
        logger.warning("query_rewrite 未产出有效改写关键词, 原始=%r", data)
        logger.info("[TOOL_RETURN] query_rewrite 返回: error=LLM 未产出有效改写关键词")
        return _obs(error="LLM 未产出有效改写关键词")
    rewritten_keywords = [str(k) for k in rewritten_keywords if k]
    ws.data["rewritten_keywords"] = rewritten_keywords
    rnd = ws.data.get("recall_round", 0)
    ws.tracer.log(f"{ws.stage}.round{rnd}", "query_rewrite",
                  last_query=last_query,
                  last_keywords=last_keywords, rewritten_keywords=rewritten_keywords)
    logger.info("[TOOL_RETURN] query_rewrite 返回: rewritten_keywords=%s", rewritten_keywords)
    return _obs(rewritten_keywords=rewritten_keywords)

@tool
def intergrate_all(query: str = "", region_code: str = "000",
                   timeout: int = 30, vector_mode: str = "both",
                   keywords: list = None) -> str:
    """生产一体化检索流水线:同时调 keyword(槽位提取→知识主索引→原子表)与
    vector(在线 embedding)两路召回,跨路去重后产出最终候选片段。
    region_code 支持区号或省份名(如 000/福建)。
    可选传入 keywords 列表直接用于检索,跳过关键词提取。
    """
    logger.info("[TOOL_CALL] intergrate_all 参数: query=%r region_code=%s timeout=%s vector_mode=%s keywords=%s",
                query, region_code, timeout, vector_mode, keywords)
    if keywords is None:
        keywords = []
    ws = get_workspace()
    query = query or ws.query
    keyword_search = getattr(ws.es, "keyword_search", None)
    vector_search = getattr(ws.es, "vector_search", None)

    kchunks: List[Chunk] = []
    kresult: dict = {}
    kerror: str = ""
    if keyword_search is not None:
        kresult = keyword_search(query=query, region_code=region_code, timeout=timeout,keywords=keywords)
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
            # 旧调用沿用旧抽象签名 (query_text, filters, size, vector_mode),
            # 与 ProduceESClient.vector_search(query_text, region_code, vector_mode) 不匹配:
            # 第3位置参数 recall_size 占用 vector_mode 槽位,再传 vector_mode= 关键字导致
            # "got multiple values for argument 'vector_mode'"
            # vresult = vector_search(query, {"region": region_code} if region_code else {},
            #                         ws.cfg.recall_size, vector_mode=vector_mode)
            vresult = vector_search(query, region_code, vector_mode=vector_mode)
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
    # logger.info("intergrate_all 合并: keyword=%d vector=%d → 去重后=%d",
    #             len(kchunks), len(vchunks), len(chunks))
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
    # logger.info("intergrate_all 完成: query=%r region=%s keyword=%d vector=%d → 去重=%d",
    #             query, region_code, len(kchunks), len(vchunks), len(chunks))

    if not chunks:
        errors = [e for e in (kerror, verror) if e]
        logger.info("[TOOL_RETURN] intergrate_all 返回: 零召回 errors=%s", errors)
        return _obs(error="keyword+vector 双路零召回" + (f"(原因: {'; '.join(errors)})" if errors else ""))
    logger.info("[TOOL_RETURN] intergrate_all 返回: recalled=%d titles=%s",
                len(chunks), [c.doc_title for c in chunks])
    return _obs(recalled=len(chunks), titles=[c.doc_title for c in chunks],
                scores=[c.score for c in chunks])

@tool
def vector_recall(query: str = "", region_code: str = "000", vector_mode: str = "both") -> str:
    """纯向量召回:直接走在线知识 embedding 向量检索服务,按语义相似度返回候选片段,
    不经过关键词/槽位提取。region_code 支持区号或省份名(如 000/福建),经 provinceId
    下推到向量服务。后端不支持向量检索时返回 error,由 agent 走兜底降级。
    """
    logger.info("[TOOL_CALL] vector_recall 参数: query=%r region_code=%s vector_mode=%s",
                query, region_code, vector_mode)
    ws = get_workspace()
    vector_search = getattr(ws.es, "vector_search", None)
    if vector_search is None:
        logger.warning("vector_recall: 当前检索后端 %s 无 vector_search,"
                       "回退关键词召回", type(ws.es).__name__)
        logger.info("[TOOL_RETURN] vector_recall 返回: error=后端不支持向量召回")
        return _obs(error="当前检索后端不支持向量召回,请改用 coarse_recall")
    query = query or ws.query
    region_code = region_code or ws.region_code
    try:
        result = vector_search(query, region_code, vector_mode=vector_mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vector_recall 异常: query=%r region=%s err=%r", query, region_code, exc)
        logger.info("[TOOL_RETURN] vector_recall 返回: error=向量召回异常 %r", exc)
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
    # logger.info("vector_recall 完成: query=%r region=%s size=%d → chunks=%d",
    #             query, region_code, size, len(chunks))
    if not chunks:
        logger.info("[TOOL_RETURN] vector_recall 返回: 零召回")
        return _obs(error="vector_recall 零召回")
    logger.info("[TOOL_RETURN] vector_recall 返回: recalled=%d titles=%s",
                len(chunks), [c.doc_title for c in chunks])
    return _obs(recalled=len(chunks), titles=[c.doc_title for c in chunks],
                scores=[c.score for c in chunks])


@tool
def keyword_recall(query: str = "", region_code: str = "000",
                   timeout: int = 30) -> str:
    """关键词一体化召回:走 keyword_search 流水线(槽位提取→知识主索引→原子表拼接)。
    region_code 支持区号或省份名(如 000/福建)。
    """
    logger.info("[TOOL_CALL] keyword_recall 参数: query=%r region_code=%s timeout=%s",
                query, region_code, timeout)
    ws = get_workspace()
    keyword_search = getattr(ws.es, "keyword_search", None)
    if keyword_search is None:
        logger.warning("keyword_recall: 当前检索后端 %s 无 keyword_search,"
                       "回退关键词召回", type(ws.es).__name__)
        logger.info("[TOOL_RETURN] keyword_recall 返回: error=后端不支持一体化流水线")
        return _obs(error="当前检索后端不支持一体化流水线,请改用 coarse_recall")
    query = query or ws.query
    result = keyword_search(query=query, region_code=region_code, timeout=timeout)
    merged = result.get("merged", []) if isinstance(result, dict) else []
    if not merged and isinstance(result, dict) and result.get("error"):
        logger.warning("keyword_recall 流水线报错: %s", result["error"])
        logger.info("[TOOL_RETURN] keyword_recall 返回: error=%s", result["error"])
        return _obs(error=result["error"])
    if not merged and isinstance(result, dict) and result.get("message"):
        logger.warning("keyword_recall 零召回: %s (keywords=%s)",
                       result["message"], result.get("keywords"))
    chunks = kresult_to_chunks(merged)
    # logger.info("keyword_recall chunks=%d", len(chunks))
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
    # logger.info("keyword_recall 完成: query=%r region=%s merged=%d → chunks=%d",
    #             query, region_code, len(merged), len(chunks))
    if not chunks:
        logger.info("[TOOL_RETURN] keyword_recall 返回: 零召回")
        return _obs(error="keyword_recall 零召回")
    logger.info("[TOOL_RETURN] keyword_recall 返回: recalled=%d titles=%s",
                len(chunks), [c.doc_title for c in chunks])
    return _obs(recalled=len(chunks), titles=[c.doc_title for c in chunks],
                scores=[c.score for c in chunks])


RETRIEVAL_TOOLS = [intergrate_all, query_rewrite, keyword_recall, vector_recall]
