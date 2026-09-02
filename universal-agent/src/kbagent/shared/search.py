# -*- coding: utf-8 -*-
"""检索层:结构化参数校验 → 模板拼装 ES DSL → 混合召回 → RRF 融合。

对应方案 3.2 / 3.3:
- LLM 永远不接触 DSL 字符串,只输出 RetrievalParams;
- 字段白名单 + 值域夹紧在 build_dsl 中强制执行,消除注入与语法错误;
- BM25 与向量 kNN 双通道并行,RRF 融合。

生产接入:实现 ESClient,内部用 elasticsearch-py 执行 build_dsl 产出的
DSL(keyword 通道)与 knn 检索(vector 通道)。MockESClient 内置一份
坐席知识库样例数据,用词面/字符重叠模拟两个通道的打分。
"""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import json
import requests
from jinja2 import Template as JinjaTemplate

from .models import Chunk, RetrievalParams

# ---- 字段白名单(方案 3.2:代码侧校验) ----
ALLOWED_FILTER_FIELDS = {"category", "status", "region"}
ALLOWED_BOOST_FIELDS = {"title", "content", "keywords"}


# ── ngkm 检索请求模板(string.Template 占位符 {{ var }}) ──────────────────
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
        "云南": "871","全国": "000"
    }

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


class ESClient(ABC):
    @abstractmethod
    def keyword_search(self, dsl: Dict[str, Any]) -> List[Chunk]: ...

    @abstractmethod
    def vector_search(self, query_text: str, filters: Dict[str, str],
                      size: int = 10) -> List[Chunk]: ...


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
                      size: int = 10) -> List[Chunk]:
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
    def _get_region_code(self,province: str) -> str:
        return _PROVINCE_TO_REGION.get(province, "200")

    def _get_atom_recall_template(knowledgeId: str) -> str:
        return _ATOM_RECALL_TEMPLATE.substitute(knowledgeId=knowledgeId)

    def _get_info_recall_template(self,province: str) -> str:
        return _INFO_RECALL_TEMPLATE.substitute(region_code=self._get_region_code(province))

    def _get_keyword(self,query: str) -> str:
        payload = {
            "query": query,
            "context": {
                "app_id": "hint_server",
                "province_id": "test_pro",
                "channel_id": "web",
            },
            "confidence_threshold": 0.5,
        }
        resp = requests.post(
            "http://restapi.ly4.tyyt.cmos:20070/slot_extract_unified",
            headers={"Content-Type": "application/json"},
            json=payload, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_info(self,keyword: str, region_code: str = "200", timeout: int = 30) -> dict:
        """知识主索引关键词检索(ngkm.knowledges_{region_code})。
        参考 tools.py 的 info_recall:省份映射 → 渲染 _INFO_RECALL_TEMPLATE →
        POST ngkm 检索接口 → 返回 json。
        region_code 支持省份名(福建/甘肃/... 自动转区号,如 "福建" -> "591")。
        """
        region_code = _PROVINCE_TO_REGION.get(region_code, region_code)
        rendered = JinjaTemplate(_INFO_RECALL_TEMPLATE).render(
            keyword=keyword, region_code=region_code)
        payload = json.loads(rendered)
        resp = requests.post(
            "http://restapi.ngkmsearch.cs.glb.cmos:20070"
            "/ngkmSearch/ws/int/busiSearcher/busiSearcherInterService",
            headers={"Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_atom(self,knowledgeId: str, region_code: str = "200", timeout: int = 30) -> dict:
        """原子表按 knowledgeId 检索(ngkm.knowledge_atom_{region_code})。
        参考 tools.py 的 atom_recall:省份映射 → 渲染 _ATOM_RECALL_TEMPLATE →
        POST ngkm 检索接口 → 返回 json。
        region_code 支持省份名(自动转区号)。
        """
        region_code = _PROVINCE_TO_REGION.get(region_code, region_code)
        rendered = JinjaTemplate(_ATOM_RECALL_TEMPLATE).render(
            knowledgeId=knowledgeId, region_code=region_code)
        payload = json.loads(rendered)
        resp = requests.post(
            "http://restapi.ngkmsearch.cs.glb.cmos:20070"
            "/ngkmSearch/ws/int/busiSearcher/busiSearcherInterService",
            headers={"Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    def keyword_search(self, query: str = "", region_code: str = "200",
                   timeout: int = 30) -> dict:
        """一体化检索流水线: keyword → info → atom → 合并。
        参考 tools.py 的 intergrate_all: 槽位提取关键词 → 知识主索引检索 →
        原子表按 knowledgeId 检索 → 拼接返回。
        """
        # ── Step 1: 槽位提取关键词 ───────────────────────────────────
        try:
            kw_resp = self._get_keyword(query=query)
        except Exception as e:
            return {"error": f"keyword 调用失败: {e}", "merged": []}

        slots = kw_resp.get("slots", []) if isinstance(kw_resp, dict) else []
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
        keywords = [k for k in keywords if k and not (k in seen or seen.add(k))]

        if not keywords:
            return {"keywords": [], "info": [], "atom": [], "merged": [],
                    "message": "未提取到有效关键词"}

        # ── Step 2: 收集所有 info 条目(跨所有 keyword) ─────────────────
        all_infos: List[dict] = []
        for kw in keywords:
            try:
                info_resp = self._get_info(keyword=kw, region_code=region_code,
                                          timeout=timeout)
            except Exception:
                continue
            raw_obj = info_resp.get("object", "") if isinstance(info_resp, dict) else ""
            parsed = json.loads(raw_obj) if isinstance(raw_obj, str) else raw_obj or {}
            infos = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
            for info in infos:
                if not isinstance(info, dict):
                    continue
                info["_keyword"] = kw
                all_infos.append(info)

        # 收集所有 knowledgeId(去重)
        seen_kids: set = set()
        kid_order: List[str] = []
        for info in all_infos:
            kid = info.get("knowledgeId") or info.get("knowledge_id") or ""
            if kid and kid not in seen_kids:
                seen_kids.add(kid)
                kid_order.append(kid)

        # ── Step 3: 按 knowledgeId 检索 atom(去重复用) ─────────────────
        atoms_cache: Dict[str, List[dict]] = {}
        for kid in kid_order:
            try:
                atom_resp = self._get_atom(knowledgeId=kid, region_code=region_code,
                                           timeout=timeout)
                raw_obj = atom_resp.get("object", "") if isinstance(atom_resp, dict) else ""
                parsed = json.loads(raw_obj) if isinstance(raw_obj, str) else raw_obj or {}
                atoms = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
                for a in atoms:
                    if isinstance(a, dict):
                        a["knowledgeId"] = kid
            except Exception as e:
                atoms = [{"knowledgeId": kid, "error": f"atom 调用失败: {e}"}]
            atoms_cache[kid] = atoms

        # ── Step 4: 合并 info + atom ──────────────────────────────────
        result: List[dict] = []
        all_info_clean: List[dict] = []
        for info in all_infos:
            kid = info.get("knowledgeId") or info.get("knowledge_id") or ""
            entry = dict(info)
            entry["atoms"] = atoms_cache.get(kid, []) if kid else []
            entry.pop("_keyword", None)
            result.append(entry)
            all_info_clean.append(entry)

        all_atoms: List[dict] = [a for atoms in atoms_cache.values() for a in atoms]

        return {
            "keywords": keywords,
            "knowledge_ids": kid_order,
            "info": all_info_clean,
            "atom": all_atoms,
            "merged_count": len(result),
            "merged": result,
        }
