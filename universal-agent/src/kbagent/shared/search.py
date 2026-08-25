# -*- coding: utf-8 -*-
"""检索层:结构化参数校验 → 模板拼装 ES DSL → 混合召回 → RRF 融合。

LLM 永远不接触 DSL 字符串,只输出 RetrievalParams;
字段白名单在 build_dsl 中强制执行。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from .models import Chunk, RetrievalParams

ALLOWED_FILTER_FIELDS = {"category", "status", "region"}
ALLOWED_BOOST_FIELDS = {"title", "content", "keywords"}


def build_dsl(params: RetrievalParams, size: int = 10) -> Dict[str, Any]:
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
        if field in ALLOWED_FILTER_FIELDS
    ]
    return {"size": size, "query": {"bool": {"must": must, "filter": filt}}}


def rrf_fuse(keyword_hits: List[Chunk], vector_hits: List[Chunk],
             k: int = 60, top_n: int = 8) -> List[Chunk]:
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
        c.score = round(s * 30, 4)
        out.append(c)
    return out


class ESClient(ABC):
    @abstractmethod
    def keyword_search(self, dsl: Dict[str, Any]) -> List[Chunk]: ...

    @abstractmethod
    def vector_search(self, query_text: str, filters: Dict[str, str],
                      size: int = 10) -> List[Chunk]: ...


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
            text = row["doc_title"] * 2 + row["content"]
            s = sum(text.count(t) for t in query_terms)
            if s > 0:
                scored.append((float(s), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_to_chunk(r, s) for s, r in scored[: dsl.get("size", 10)]]

    def vector_search(self, query_text: str, filters: Dict[str, str],
                      size: int = 10) -> List[Chunk]:
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
