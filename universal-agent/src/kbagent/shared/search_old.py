# -*- coding: utf-8 -*-
"""检索层:结构化参数校验 → 模板拼装 ES DSL → 混合召回 → RRF 融合。

对应方案 3.2 / 3.3:
- LLM 永远不接触 DSL 字符串,只输出 RetrievalParams;
- 字段白名单 + 值域夹紧在 build_dsl 中强制执行,消除注入与语法错误;
- BM25 与向量 kNN 双通道并行,RRF 融合。

客户端实现:
- MockESClient    内置坐席知识库样例数据,离线演示/测试用;
- ProduceESClient 生产 ngkm 检索(槽位提取 → 知识主索引召回 → 原子表拼接),
                  一体化流水线经 ``full_recall`` 暴露,并映射为标准
                  ``ESClient`` 接口(keyword_search / vector_search);
                  vector 通道走在线知识 embedding 向量检索服务,新旧两套
                  请求模板经 ``vector_search`` 的 ``vector_mode`` 参数选路
                  (new/old/both,both 时按 ``vector_weights`` 比例分配返回名额)。
"""
from __future__ import annotations

import ast
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import requests
from jinja2 import Template as JinjaTemplate

from .models import Chunk, RetrievalParams, new_id

logger = logging.getLogger("kbagent.search")

# ---- 字段白名单(方案 3.2:代码侧校验) ----
ALLOWED_FILTER_FIELDS = {"category", "status", "region"}
ALLOWED_BOOST_FIELDS = {"title", "content", "keywords"}


# ── ngkm 检索请求模板(Jinja2 占位符 {{ var }}) ─────────────────────────
_INFO_RECALL_TEMPLATE = """{
  "beans": [],
  "params": {
    "indexType": "knowledges_info",
    "indexName": "ngkm.knowledges_{{ region_code }}",
    "page": "1",
    "size": "100",
    "keyWord": "{{ keyword }}",
    "searchInfo": "knowledgeName=10,klgAliasName=5",
    "analysisType": "smart",
    "relCalculus": "OR",
    "highlightField": "knowledgeName,klgAliasName"
  }
}"""

_ATOM_RECALL_TEMPLATE = """{
  "beans": [
    {"column": "knowledgeId", "value": "{{ knowledgeId }}", "type": "any"}
  ],
  "params": {
        "indexType": "_doc",
        "mandatoryField": "knowledgeId,paramType,klgAttrAtomId,paramName,content,wkuntt,srcTemplateAttrAtomId,channelCode,groupId,except,annotation,isSendMessage,srcTmpltGrpngId",
        "indexName": "ngkm.knowledge_atom_{{ region_code }}",
        "ignoreField": "_id"
    }
}"""

_PROVINCE_TO_REGION = {
        "福建": "591", "甘肃": "931", "海南": "898", "河北": "311",
        "黑龙江": "451", "河南": "371", "宁夏": "951", "四川": "280",
        "云南": "871", "全国": "000",
    }

_SLOT_EXTRACT_URL = "http://restapi.ly4.tyyt.cmos:20070/slot_extract_unified"
_NGKM_SEARCH_URL = ("http://restapi.ngkmsearch.cs.glb.cmos:20070"
                    "/ngkmSearch/ws/int/busiSearcher/busiSearcherInterService")
_VECTOR_SEARCH_URL = ("http://192.168.212.199:8902/group/online-knowledge-embedding"
                      "/open/embedding/search/vector")


# ── 在线知识 embedding 向量检索请求体(新旧两套模板) ──────────────────────
# 公共字段;每次请求动态写入: content(用户问题) / reqId / xTransId(重新生成),
# sessionId 暂时保持原请求中的固定值。
_VECTOR_PAYLOAD_COMMON: Dict[str, Any] = {
    "esTop": 0,
    "isEnableGroupContent": True,
    "faqThresholdScore": 0.75,
    "rerankModelType": "ZY_QW",
    "isEnableGroupName": False,
    "faqContentThresholdScore": 0.7,
    "knowledgeThresholdScore": 0.35,
    "isOn": False,
    "isEnableResetRerank": False,
    "knowledgeNameSeparateThresholdScore": 0.9,
    "isEnableTotalKnowledge": "1",
    "isEnableFaq": True,
    "recessivityFlag": "0",
    "isEnableQa": True,
    "intentNM": "",
    "provinceId": "",
    "isEnableKnowledgeName": True,
    "reqId": "",                       # 每次请求重新生成
    "groupBindQuestionThresholdScore": 0.8,
    "sysChnlCode": "lingxi",
    "isEnableAtomName": False,
    "qaType": "out",
    "groupQuestionThresholdScore": 0.8,
    "knowledgeNameGroupThresholdScore": 0.3,
    "knowledgePath": "",
    "klgState": "2",
    "isEnableGroupBindQuestion": True,
    "isEnableFaqContent": True,
    "qaThresholdScore": 0.6,
    "enableKeywordReplace": "true",
    "faqoutScore": 0.85,
    "content": "",                     # 每次请求替换为用户问题
    "isEnableReRank": False,
    "isEnableQaContent": True,
    "channelCode": "1,lingxi,zaixian",
    "maxTop": 100,
    "replyModelScore": 0.9,
    "xTransId": "",                    # 每次请求重新生成
    "thresholdScore": 0.3,
    "newRouteExpThresholdScore": 0.3,
    "knowledgeGroupThresholdScore": 0.72,
    "sessionId": "53NaJ6mPpiiIxSMkb7k3t3eBgPMNRHd9",
    "knowledgeNameCoefficient": 0.35,
    "isEnableGroupQuestion": True,
    "knowledgeTop": 10,
}

# 新模板:启用新路由实验(newRouteExp),top=1 精排,embeddingTop=10
_VECTOR_PAYLOAD_NEW: Dict[str, Any] = {
    **_VECTOR_PAYLOAD_COMMON,
    "knowledgeNameGroupTop": 2,
    "embeddingTop": 10,
    "isEnableNewRouteExp": True,
    "newRouteExpTop": 100,
    "groupBindQuestionTop": 5,
    "top": 1,
}

# 旧模板:关闭新路由实验,top=100 宽召回,embeddingTop=30,含 faq/qaContent 独立条数
_VECTOR_PAYLOAD_OLD: Dict[str, Any] = {
    **_VECTOR_PAYLOAD_COMMON,
    "knowledgeNameGroupTop": 20,
    "embeddingTop": 30,
    "isEnableNewRouteExp": False,
    "newRouteExpTop": 0,
    "faqTop": 20,
    "groupBindQuestionTop": 20,
    "top": 100,
    "qaContentTop": 20,
}

_VECTOR_PAYLOADS: Dict[str, Dict[str, Any]] = {
    "new": _VECTOR_PAYLOAD_NEW,
    "old": _VECTOR_PAYLOAD_OLD,
}

# 向量响应命中条目字段(在线知识 embedding 服务固定返回)
_VECTOR_ID_FIELD = "knowledgeId"
_VECTOR_TITLE_FIELD = "knowledgeName"
_VECTOR_CONTENT_FIELD = "content"


def _region_code(value: str) -> str:
    """省份名 → 区号;已是区号(或其他值)原样返回。"""
    return _PROVINCE_TO_REGION.get(value, value)


def _preview(value: Any, limit: int = 800) -> str:
    """日志安全预览:dict/list 转 JSON,控制台换行压成空格,超长截断。"""
    try:
        s = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        s = str(value)
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= limit else s[:limit] + f"...(共{len(s)}字符)"


def _extract_doc_list(parsed: Any) -> List[dict]:
    """从 ngkm 检索响应的 object 解析结果中提取知识条目列表。

    生产响应结构:object 是 JSON 对象,真正的条目列表在 ``docment`` 字段;
    兼容解析结果本身就是 list 的旧结构;dict 且无 docment 字段时按单条处理。
    """
    if isinstance(parsed, list):
        return [d for d in parsed if isinstance(d, dict)]
    if isinstance(parsed, dict):
        for key in ("docment", "document", "documents", "docs"):
            docs = parsed.get(key)
            if isinstance(docs, list):
                logger.info("ngkm 响应从字段 %r 提取条目列表,共 %d 条",
                            key, len(docs))
                return [d for d in docs if isinstance(d, dict)]
        return [parsed] if parsed else []
    return []


def build_dsl(params: RetrievalParams, size: int = 10) -> Dict[str, Any]:
    """由结构化参数拼装 keyword 通道的 ES DSL。只认白名单字段。"""
    terms = [t for t in (params.keywords + params.expanded_terms) if t]
    boosts = {k: v for k, v in params.boost_fields.items() if k in ALLOWED_BOOST_FIELDS}
    if not boosts:
        boosts = {"title": 2.0, "content": 1.0}
    fields = [f"{name}^{weight}" for name, weight in boosts.items()]

    must: List[Dict[str, Any]] = [{
        "multi_match": {
            "query": " ".join(terms) if terms else "*",
            "fields": fields,
            "type": "best_fields",
        }
    }]
    filt: List[Dict[str, Any]] = [
        {"term": {field: value}}
        for field, value in params.filters.items()
        if field in ALLOWED_FILTER_FIELDS          # 白名单外的过滤字段直接丢弃
    ]
    return {"size": size, "query": {"bool": {"must": must, "filter": filt}}}


def rrf_fuse(keyword_hits: List[Chunk], vector_hits: List[Chunk],
             k: int = 60, top_n: int = 8) -> List[Chunk]:
    """Reciprocal Rank Fusion:score = Σ 1/(k + rank)。"""
    scores: Dict[str, float] = {}
    pool: Dict[str, Chunk] = {}
    for hits in (keyword_hits, vector_hits):
        for rank, chunk in enumerate(hits):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            pool.setdefault(chunk.chunk_id, chunk)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    out: List[Chunk] = []
    for cid, s in ranked:
        c = pool[cid]
        c.score = round(s * 30, 4)   # 归一到与阈值同量纲(Mock 用)
        out.append(c)
    return out


def merged_to_chunks(merged: List[Dict[str, Any]]) -> List[Chunk]:
    """一体化流水线的知识条目(info+atoms)→ 标准 Chunk 列表。

    一条知识映射一个 Chunk:content 由原子字段拼接(参数名:内容),
    原始条目完整保留在 extra 供溯源;生产侧无显式相关性得分,按出现顺序衰减。
    """
    chunks: List[Chunk] = []
    for rank, entry in enumerate(merged or []):
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("knowledgeId") or entry.get("knowledge_id") or "")
        title = str(entry.get("knowledgeName") or entry.get("knowledge_name") or "")
        lines: List[str] = []
        atoms = entry.get("atoms") or []
        for atom in atoms:
            if not isinstance(atom, dict) or atom.get("error"):
                continue
            name = str(atom.get("paramName") or "").strip()
            text = str(atom.get("content") or "").strip()
            if not text:
                continue
            lines.append(f"{name}:{text}" if name else text)
        content = "\n".join(lines) or str(entry.get("content") or "") or title
        if not content:
            continue
        updated_at = ""
        for key in ("updateTime", "update_time", "srcTime", "createTime"):
            if entry.get(key):
                updated_at = str(entry[key])
                break
        chunks.append(Chunk(
            chunk_id=f"ngkm_{kid}" if kid else new_id("ngkm"),
            doc_id=kid or "unknown",
            doc_title=title,
            content=content,
            category=str(entry.get("category") or ""),
            position={"knowledge_id": kid},
            updated_at=updated_at,
            score=round(max(0.5, 1.0 - 0.05 * rank), 4),
            extra={"status": str(entry.get("status") or ""),
                   "source": "ngkm",
                   "atoms": atoms},
        ))
    if len(chunks) < len(merged or []):
        logger.info("merged_to_chunks: %d 条知识条目 → %d 条有效 Chunk"
                    "(无内容/原子全失败的条目被丢弃)",
                    len(merged or []), len(chunks))
    for c in chunks[:5]:
        logger.info("merged_to_chunks 产出: id=%s title=%r content_len=%d",
                    c.chunk_id, c.doc_title, len(c.content))
    return chunks


# ---------------------------------------------------------------------------
# 向量检索:原始返回 → 标准 Chunk(在线知识 embedding 服务)
# ---------------------------------------------------------------------------
def _vector_raw_to_chunks(resp: Any, source: str) -> List[Chunk]:
    """向量检索原始返回 → 标准 Chunk 列表(按响应顺序,即服务端相关性顺序)。

    条目标定为含 knowledgeId/knowledgeName 之一且含 content 的 dict,命中即收、
    不再下钻;兼容 ngkm 风格 ``object``(JSON 字符串)包裹;每个 Chunk 的
    ``extra.raw`` 保留对应条目原文供溯源,``source`` 标记来源模板(new/old)。
    """
    chunks: List[Chunk] = []
    seen: set = set()

    def add_entry(entry: Dict[str, Any]) -> None:
        kid = str(entry.get(_VECTOR_ID_FIELD) or "")
        title = str(entry.get(_VECTOR_TITLE_FIELD) or "")
        content = str(entry.get(_VECTOR_CONTENT_FIELD) or "") or title
        # 有 knowledgeId 用它去重;无 id 的条目用随机 id,不互相合并
        key = kid or new_id("vec")
        if kid and key in seen:
            return
        seen.add(key)
        chunks.append(Chunk(
            chunk_id=f"vec_{kid}" if kid else key,
            doc_id=kid or "unknown",
            doc_title=title,
            content=content,
            category="",
            position={"knowledge_id": kid, "rank": len(chunks)},
            score=0.0,  # 最终位置分由 vector_search 按返回顺序统一编号
            extra={"source": "vector",
                   "vector_channel": source,
                   "raw": entry},
        ))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            raw_obj = node.get("object")
            if isinstance(raw_obj, str):
                try:
                    walk(json.loads(raw_obj))
                except (json.JSONDecodeError, TypeError):
                    pass
            has_id = node.get(_VECTOR_ID_FIELD) not in (None, "")
            has_title = node.get(_VECTOR_TITLE_FIELD) not in (None, "")
            has_content = node.get(_VECTOR_CONTENT_FIELD) not in (None, "")
            if (has_id or has_title) and has_content:
                add_entry(node)
                return
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(resp)
    return chunks


class ESClient(ABC):
    @abstractmethod
    def keyword_search(self, dsl: Dict[str, Any]) -> List[Chunk]: ...

    @abstractmethod
    def vector_search(self, query_text: str, filters: Dict[str, str],
                      size: int = 10, vector_mode: str = "new",
                      vector_weights: Optional[Dict[str, float]] = None
                      ) -> List[Chunk]: ...


# ---------------------------------------------------------------------------
# Mock ES:内置坐席知识库样例(套餐/宽带/账单/投诉)
# ---------------------------------------------------------------------------
_KB: List[Dict[str, Any]] = [
    dict(chunk_id="kb_0001#p1", doc_id="kb_0001", doc_title="5G畅享套餐资费说明",
         category="套餐", status="在售", updated_at="2026-06-10", version="v3.2",
         content="5G畅享套餐月费59元,含30GB全国流量、500分钟通话。达量后限速至1Mbps,不额外收费。"
                 "次月1日生效,当月按天折算。"),
    dict(chunk_id="kb_0001#p2", doc_id="kb_0001", doc_title="5G畅享套餐资费说明",
         category="套餐", status="在售", updated_at="2026-06-10", version="v3.2",
         content="办理条件:实名客户,无欠费。合约期内客户需先解除原合约再变更套餐。"),
    dict(chunk_id="kb_0002#p1", doc_id="kb_0002", doc_title="流量加油包推荐话术",
         category="套餐", status="在售", updated_at="2026-05-20", version="v1.4",
         content="客户反映流量不够用时,优先推荐10元5GB加油包,当月有效,立即生效。"
                 "月流量长期超量的客户建议升级更高档位套餐。"),
    dict(chunk_id="kb_0003#p1", doc_id="kb_0003", doc_title="家庭宽带新装流程",
         category="宽带", status="在售", updated_at="2026-04-01", version="v2.0",
         content="家庭宽带新装需客户提供安装地址与实名信息,预约后48小时内上门。300M宽带月费30元。"),
    dict(chunk_id="kb_0004#p1", doc_id="kb_0004", doc_title="话费账单查询指引",
         category="账单", status="在售", updated_at="2026-03-15", version="v1.1",
         content="客户可通过APP查询近6个月账单明细。争议扣费由坐席发起账单复核工单,3个工作日内答复。"),
    dict(chunk_id="kb_0005#p1", doc_id="kb_0005", doc_title="投诉处理时限规范",
         category="投诉", status="在售", updated_at="2026-01-08", version="v1.0",
         content="一般投诉48小时内首次回复,资费争议类投诉24小时内升级处理。"),
    dict(chunk_id="kb_0006#p1", doc_id="kb_0006", doc_title="旧版4G套餐说明(已下架)",
         category="套餐", status="下架", updated_at="2024-11-01", version="v0.9",
         content="4G飞享套餐月费38元,含5GB流量,已停止办理。"),
]


def _to_chunk(row: Dict[str, Any], score: float) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"], doc_id=row["doc_id"], doc_title=row["doc_title"],
        content=row["content"], category=row["category"],
        position={"para": int(row["chunk_id"].split("#p")[-1])},
        version=row["version"], updated_at=row["updated_at"], score=score,
        extra={"status": row["status"]},
    )


class MockESClient(ESClient):
    def keyword_search(self, dsl: Dict[str, Any]) -> List[Chunk]:
        query_terms = dsl["query"]["bool"]["must"][0]["multi_match"]["query"].split()
        filters = {list(f["term"].keys())[0]: list(f["term"].values())[0]
                   for f in dsl["query"]["bool"].get("filter", [])}
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in _KB:
            if any(row.get(k) != v for k, v in filters.items()):
                continue
            text = row["doc_title"] * 2 + row["content"]   # 标题加权
            s = sum(text.count(t) for t in query_terms)
            if s > 0:
                scored.append((float(s), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_to_chunk(r, s) for s, r in scored[: dsl.get("size", 10)]]

    def vector_search(self, query_text: str, filters: Dict[str, str],
                      size: int = 10, vector_mode: str = "new",
                      vector_weights: Optional[Dict[str, float]] = None
                      ) -> List[Chunk]:
        # Mock 无新旧模板之分,vector_mode/vector_weights 仅为对齐 ESClient 接口
        # 用字符集合重叠率模拟语义相似度,兜住口语化改写
        q = set(query_text) - set(" ,。?？!")
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in _KB:
            if any(row.get(k) != v for k, v in filters.items()):
                continue
            t = set(row["doc_title"] + row["content"])
            sim = len(q & t) / max(len(q), 1)
            if sim > 0.15:
                scored.append((sim, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_to_chunk(r, round(s, 4)) for s, r in scored[:size]]


class ProduceESClient(ESClient):
    """生产 ngkm 检索客户端。

    keyword 通道:一体化流水线(槽位提取 → 知识主索引召回 → 原子表拼接),
    经 ``full_recall`` 返回原始结构,``keyword_search`` 将其映射为标准 Chunk。
    vector 通道:在线知识 embedding 向量检索服务(新旧两套请求模板),
    选路参数由 ``vector_search`` 逐次传入、不做客户端级配置:
    - ``vector_mode="new"``  仅新模板(新路由实验,top=1 精排,默认);
    - ``vector_mode="old"``  仅旧模板(top=100 宽召回);
    - ``vector_mode="both"`` 新旧两路均请求,日志给出两路总条数与新旧比例,
      按 ``vector_weights`` 比例分配返回名额(默认 new:old=0.6:0.4),
      跨路同一知识去重(新模板优先),某路不足由另一路补齐。
    任一通道异常时该路降级为空,不影响另一路;两路全空时 RRF 退化为纯关键词。
    """

    def __init__(self, region_code: str = "000", timeout: int = 30):
        self.region_code = region_code      # 支持省份名,内部自动转区号
        self.timeout = timeout

    # ------------------------------------------------------------------
    # ESClient 标准接口
    # ------------------------------------------------------------------
    def keyword_search(self, dsl: Dict[str, Any]) -> List[Chunk]:
        """按 DSL 中的关键词走 info → atom 召回(关键词已在上游提取,跳过槽位抽取)。"""
        query = dsl["query"]["bool"]["must"][0]["multi_match"]["query"]
        keywords = [t for t in query.split() if t and t != "*"]
        region = self.region_code
        for f in dsl["query"]["bool"].get("filter", []):
            term = f.get("term") or {}
            if "region" in term:
                region = str(term["region"])
        if not keywords:
            return []
        result = self._info_atom_recall(keywords, region, self.timeout)
        return merged_to_chunks(result.get("merged", []))[: dsl.get("size", 10)]

    def vector_search(self, query_text: str, filters: Dict[str, str],
                      size: int = 10, vector_mode: str = "new",
                      vector_weights: Optional[Dict[str, float]] = None
                      ) -> List[Chunk]:
        """向量通道(ESClient 接口):按 ``vector_mode`` 调度召回,返回标准 Chunk。

        vector_mode/vector_weights 逐次调用传入(默认 new 单模板;both 时按
        vector_weights 分配新旧两路返回名额),召回/选路见 ``_vector_recall``;
        过滤条件中 region 经请求体 provinceId 下推,向量响应仅含
        knowledgeId/knowledgeName/content,不做 category/status 客户端过滤。
        返回顺序即最终相关性顺序,位置分按 1/(rank+1) 统一编号。
        """
        if not query_text or not query_text.strip():
            return []
        mode = (vector_mode or "new").strip().lower()
        weights = vector_weights or {"new": 0.6, "old": 0.4}
        province = str(filters.get("region") or "")
        chunks = self._vector_recall(query_text, province, self.timeout, size,
                                     mode, weights)
        for rank, chunk in enumerate(chunks):
            chunk.position["rank"] = rank
            chunk.score = round(1.0 / (rank + 1), 4)
        return chunks

    # ------------------------------------------------------------------
    # 向量召回:新模板请求 / 旧模板请求 / 模式调度
    # ------------------------------------------------------------------
    def _vector_recall_new(self, query_text: str, province: str,
                           timeout: int) -> Optional[dict]:
        """按新模板请求(newRouteExp,top=1 精排),返回接口完整原始返回;失败降级为 None。"""
        try:
            resp = self._post_vector_search("new", query_text, timeout,
                                            province=province)
            logger.info("向量召回(新模板) query=%r 完成", query_text)
            return resp
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量召回(新模板)失败,降级为空 query=%r err=%r",
                           query_text, exc)
            return None

    def _vector_recall_old(self, query_text: str, province: str,
                           timeout: int) -> Optional[dict]:
        """按旧模板请求(top=100 宽召回),返回接口完整原始返回;失败降级为 None。"""
        try:
            resp = self._post_vector_search("old", query_text, timeout,
                                            province=province)
            logger.info("向量召回(旧模板) query=%r 完成", query_text)
            return resp
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量召回(旧模板)失败,降级为空 query=%r err=%r",
                           query_text, exc)
            return None

    def _vector_recall(self, query_text: str, province: str, timeout: int,
                       size: int, vector_mode: str,
                       vector_weights: Dict[str, float]) -> List[Chunk]:
        """按 ``vector_mode``(vector_search 入参)调度召回并转 Chunk,返回去重/配额后的 Chunk 列表。

        - new/old: 单模板请求,原始返回经 ``_vector_raw_to_chunks`` 转 Chunk;
        - both: 新旧两路均请求,日志给出两路总条数与新旧比例,按
          ``vector_weights`` 把 size 个返回名额分配给两路(默认新 6 旧 4);
          跨路同一知识(chunk_id)去重,新模板优先保留;某路名额不足时
          由另一路剩余 Chunk 补齐。权重为 0 的路不请求、不占名额。
        """
        if vector_mode == "new":
            return _vector_raw_to_chunks(
                self._vector_recall_new(query_text, province, timeout),
                "new")[:size]
        if vector_mode == "old":
            return _vector_raw_to_chunks(
                self._vector_recall_old(query_text, province, timeout),
                "old")[:size]

        # both: 新旧两路均请求;权重为 0 的路不请求、不占名额、不参与去重
        w_new = vector_weights.get("new", 0.5)
        w_old = vector_weights.get("old", 0.5)
        new_chunks = (_vector_raw_to_chunks(
            self._vector_recall_new(query_text, province, timeout), "new")
            if w_new > 0 else [])
        old_chunks = (_vector_raw_to_chunks(
            self._vector_recall_old(query_text, province, timeout), "old")
            if w_old > 0 else [])

        # 跨路去重(新模板优先保留;无 knowledgeId 的 chunk_id 随机,不合并)
        seen: set = set()
        new_unique: List[Chunk] = []
        old_unique: List[Chunk] = []
        for c in new_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                new_unique.append(c)
        for c in old_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                old_unique.append(c)

        # 按新旧比例分配返回名额
        n_new = round(size * w_new) if w_new > 0 else 0
        n_old = size - n_new
        picked = new_unique[:n_new] + old_unique[:n_old]
        # 某路不足名额:用另一路剩余 Chunk 补齐到 size
        if len(picked) < size:
            rest = new_unique[n_new:] + old_unique[n_old:]
            picked += rest[:size - len(picked)]

        logger.info("向量混合召回: 新模板=%d条 旧模板=%d条 两路总条数=%d; "
                    "新旧比例=%.2f:%.2f → 分配名额 新%d/旧%d(size=%d), 实际返回=%d条",
                    len(new_chunks), len(old_chunks),
                    len(new_chunks) + len(old_chunks),
                    w_new, w_old, n_new, n_old, size, len(picked))
        return picked[:size]

    def raw_vector_search(self, query_text: str, mode: str = "new",
                          province: str = "", timeout: int = 0
                          ) -> Dict[str, Any]:
        """向量检索原始调用(批量评测/排障用),固定返回三字段、绝不抛异常:

        ``{"response": 接口完整原始返回(dict), "success": bool, "error": str}``

        - mode: ``new`` / ``old`` 单模板请求(both 为两路融合,无单一原始返回,
          请在批量脚本中按模式分文件保存);
        - 网络/超时/HTTP 非 2xx/参数非法等全部落入 ``success=False`` +
          ``error``(异常类型与信息),``response`` 为 None,批量任务不中断。
        """
        mode = (mode or "new").strip().lower()
        if mode not in _VECTOR_PAYLOADS:
            return {"response": None, "success": False,
                    "error": f"mode 仅支持 new/old,收到: {mode!r}"}
        if not query_text or not query_text.strip():
            return {"response": None, "success": False, "error": "query 为空"}
        try:
            resp = self._post_vector_search(mode, query_text,
                                            timeout or self.timeout,
                                            province=province)
            return {"response": resp, "success": True, "error": ""}
        except Exception as exc:  # noqa: BLE001
            return {"response": None, "success": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _post_vector_search(self, mode: str, query_text: str,
                            timeout: int, province: str = "") -> dict:
        """按模板(mode=new/old)发向量检索请求;reqId/xTransId 每次重新生成。

        province 非空时写入请求体 ``provinceId``(测试集"省份"列透传)。
        """
        payload = dict(_VECTOR_PAYLOADS[mode])
        payload["content"] = query_text
        if province:
            payload["provinceId"] = province
        payload["reqId"] = uuid.uuid4().hex
        payload["xTransId"] = uuid.uuid4().hex
        logger.info("向量检索请求 mode=%s content=%r top=%s embeddingTop=%s "
                    "isEnableNewRouteExp=%s",
                    mode, query_text, payload.get("top"),
                    payload.get("embeddingTop"), payload.get("isEnableNewRouteExp"))
        try:
            resp = requests.post(
                _VECTOR_SEARCH_URL,
                headers={"Content-Type": "application/json"},
                json=payload, timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量检索请求失败(网络/超时/DNS) mode=%s url=%s err=%r",
                           mode, _VECTOR_SEARCH_URL, exc)
            raise
        logger.info("向量检索响应 mode=%s status=%s body=%s",
                    mode, resp.status_code, resp.text[:800])
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # 一体化流水线:槽位提取 → info 召回 → atom 召回 → 合并
    # ------------------------------------------------------------------
    def full_recall(self, query: str, region_code: str = "",
                    timeout: int = 0) -> dict:
        """完整流水线,返回 {keywords, knowledge_ids, info, atom, merged_count, merged}。"""
        raw_region = region_code or self.region_code
        region_code = _region_code(raw_region)
        timeout = timeout or self.timeout
        logger.info("full_recall 开始 query=%r region=%r→%r "
                    "索引=ngkm.knowledges_%s / ngkm.knowledge_atom_%s timeout=%ss",
                    query, raw_region, region_code, region_code, region_code, timeout)
        try:
            keywords = self._extract_keywords(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("full_recall 槽位提取异常,流水线终止: %r", exc)
            return {"error": f"keyword 调用失败: {exc}", "merged": []}
        if not keywords:
            logger.warning("full_recall 未提取到有效关键词 → 零召回 query=%r", query)
            return {"keywords": [], "info": [], "atom": [], "merged": [],
                    "message": "未提取到有效关键词"}
        return self._info_atom_recall(keywords, region_code, timeout)

    def _extract_keywords(self, query: str) -> List[str]:
        """Step 1:槽位抽取服务提取检索关键词。"""
        payload = {
            "query": query,
            "context": {
                "app_id": "hint_server",
                "province_id": "test_pro",
                "channel_id": "web",
            },
            "confidence_threshold": 0.5,
        }
        try:
            resp = requests.post(_SLOT_EXTRACT_URL,
                                 headers={"Content-Type": "application/json"},
                                 json=payload, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("槽位提取请求失败(网络/超时/DNS) url=%s err=%r",
                           _SLOT_EXTRACT_URL, exc)
            raise
        if resp.status_code != 200:
            logger.warning("槽位提取返回非200 status=%s body=%s",
                           resp.status_code, resp.text[:300])
        resp.raise_for_status()
        data = resp.json()
        logger.info("槽位提取原始响应 query=%r: %s", query, _preview(data))
        slots = data.get("slots", []) if isinstance(data, dict) else []
        keywords: List[str] = []
        for slot in slots:
            raw = slot.get("slot_value", "") if isinstance(slot, dict) else ""
            if not raw:
                continue
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)):
                    keywords.extend(str(v).strip() for v in parsed if v)
                else:
                    keywords.append(str(parsed).strip())
            except (ValueError, SyntaxError):
                keywords.append(raw.strip())
        seen: set = set()
        deduped = [k for k in keywords if k and not (k in seen or seen.add(k))]
        logger.info("槽位提取完成 query=%r slots=%d个 → keywords=%s",
                    query, len(slots), deduped)
        if not deduped:
            logger.warning("槽位提取返回空关键词,响应体=%s", str(data)[:300])
        return deduped

    def _info_atom_recall(self, keywords: List[str], region_code: str,
                          timeout: int) -> dict:
        """Step 2-4:info 召回 → 按 knowledgeId 拉 atom → 合并。"""
        region_code = _region_code(region_code)

        # ---- Step 2: 收集所有 info 条目(跨所有 keyword) ----
        all_infos: List[dict] = []
        for kw in keywords:
            try:
                info_resp = self._get_info(keyword=kw, region_code=region_code,
                                           timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("info 召回失败,跳过该关键词 keyword=%r "
                               "索引=ngkm.knowledges_%s err=%r",
                               kw, region_code, exc)
                continue
            logger.info("info 原始响应 keyword=%r: %s", kw, _preview(info_resp))
            raw_obj = info_resp.get("object", "") if isinstance(info_resp, dict) else ""
            try:
                parsed = json.loads(raw_obj) if isinstance(raw_obj, str) else raw_obj or {}
            except json.JSONDecodeError as exc:
                logger.warning("info 响应 object 非 JSON,跳过 keyword=%r err=%r raw=%s",
                               kw, exc, str(raw_obj)[:200])
                continue
            infos = _extract_doc_list(parsed)
            for info in infos:
                if not isinstance(info, dict):
                    continue
                info["_keyword"] = kw
                all_infos.append(info)
            logger.info("info 召回 keyword=%r → %d 条", kw, len(infos))
            for i, info in enumerate(infos[:5]):
                if isinstance(info, dict):
                    logger.info("  info[%d]: knowledgeId=%s name=%r keys=%s", i,
                                info.get("knowledgeId") or info.get("knowledge_id"),
                                info.get("knowledgeName") or info.get("knowledge_name"),
                                sorted(info.keys())[:12])

        # 收集所有 knowledgeId(去重)
        seen_kids: set = set()
        kid_order: List[str] = []
        for info in all_infos:
            kid = info.get("knowledgeId") or info.get("knowledge_id") or ""
            if kid and kid not in seen_kids:
                seen_kids.add(kid)
                kid_order.append(kid)
        logger.info("info 召回汇总: keywords=%s 总条目=%d knowledgeIds=%s",
                    keywords, len(all_infos), kid_order)
        if not all_infos:
            logger.warning("所有关键词均无 info 召回——请检查索引 "
                           "ngkm.knowledges_%s 是否存在、其中有无匹配知识",
                           region_code)

        # ---- Step 3: 按 knowledgeId 检索 atom(去重复用) ----
        atoms_cache: Dict[str, List[dict]] = {}
        for kid in kid_order:
            try:
                atom_resp = self._get_atom(knowledgeId=kid, region_code=region_code,
                                           timeout=timeout)
                logger.info("atom 原始响应 knowledgeId=%s: %s", kid, _preview(atom_resp))
                raw_obj = atom_resp.get("object", "") if isinstance(atom_resp, dict) else ""
                parsed = json.loads(raw_obj) if isinstance(raw_obj, str) else raw_obj or {}
                atoms = _extract_doc_list(parsed)
                for a in atoms:
                    if isinstance(a, dict):
                        a["knowledgeId"] = kid
                logger.info("atom 召回 knowledgeId=%s → %d 条", kid, len(atoms))
            except Exception as exc:  # noqa: BLE001
                logger.warning("atom 召回失败 knowledgeId=%s "
                               "索引=ngkm.knowledge_atom_%s err=%r",
                               kid, region_code, exc)
                atoms = [{"knowledgeId": kid, "error": f"atom 调用失败: {exc}"}]
            atoms_cache[kid] = atoms

        # ---- Step 4: 合并 info + atom ----
        merged: List[dict] = []
        all_info_clean: List[dict] = []
        for info in all_infos:
            kid = info.get("knowledgeId") or info.get("knowledge_id") or ""
            entry = dict(info)
            entry["atoms"] = atoms_cache.get(kid, []) if kid else []
            entry.pop("_keyword", None)
            merged.append(entry)
            all_info_clean.append(entry)

        all_atoms: List[dict] = [a for atoms in atoms_cache.values() for a in atoms]
        logger.info("full_recall 完成: merged=%d 条 (info=%d, atom=%d) knowledgeIds=%s",
                    len(merged), len(all_info_clean), len(all_atoms), kid_order)
        return {
            "keywords": keywords,
            "knowledge_ids": kid_order,
            "info": all_info_clean,
            "atom": all_atoms,
            "merged_count": len(merged),
            "merged": merged,
        }

    # ------------------------------------------------------------------
    # ngkm HTTP 调用
    # ------------------------------------------------------------------
    def _get_info(self, keyword: str, region_code: str = "000",
                  timeout: int = 30) -> dict:
        """知识主索引关键词检索(ngkm.knowledges_{region_code})。"""
        rendered = JinjaTemplate(_INFO_RECALL_TEMPLATE).render(
            keyword=keyword, region_code=_region_code(region_code))
        logger.info("ngkm info 请求体 keyword=%r: %s", keyword,
                    rendered.replace("\n", " "))
        try:
            resp = requests.post(
                _NGKM_SEARCH_URL,
                headers={"Content-Type": "application/json"},
                json=json.loads(rendered), timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ngkm info 请求失败(网络/超时/DNS) url=%s keyword=%r err=%r",
                           _NGKM_SEARCH_URL, keyword, exc)
            raise
        logger.info("ngkm info 响应 keyword=%r status=%s body=%s",
                    keyword, resp.status_code, resp.text[:800])
        if resp.status_code != 200:
            logger.warning("ngkm info 检索非200 keyword=%r 索引=ngkm.knowledges_%s "
                           "status=%s", keyword, _region_code(region_code),
                           resp.status_code)
        resp.raise_for_status()
        return resp.json()

    def _get_atom(self, knowledgeId: str, region_code: str = "000",
                  timeout: int = 30) -> dict:
        """原子表按 knowledgeId 检索(ngkm.knowledge_atom_{region_code})。"""
        rendered = JinjaTemplate(_ATOM_RECALL_TEMPLATE).render(
            knowledgeId=knowledgeId, region_code=_region_code(region_code))
        logger.info("ngkm atom 请求体 knowledgeId=%s: %s", knowledgeId,
                    rendered.replace("\n", " "))
        try:
            resp = requests.post(
                _NGKM_SEARCH_URL,
                headers={"Content-Type": "application/json"},
                json=json.loads(rendered), timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ngkm atom 请求失败(网络/超时/DNS) url=%s knowledgeId=%s err=%r",
                           _NGKM_SEARCH_URL, knowledgeId, exc)
            raise
        logger.info("ngkm atom 响应 knowledgeId=%s status=%s body=%s",
                    knowledgeId, resp.status_code, resp.text[:800])
        if resp.status_code != 200:
            logger.warning("ngkm atom 检索非200 knowledgeId=%s "
                           "索引=ngkm.knowledge_atom_%s status=%s",
                           knowledgeId, _region_code(region_code),
                           resp.status_code)
        resp.raise_for_status()
        return resp.json()
