# -*- coding: utf-8 -*-
"""高频问题缓存(方案 二·快速通道)。

命中标准:归一化 query 的相似度高于阈值,且缓存条目引用的知识版本未失效。
生产接入:相似度替换为 embedding 余弦(可复用向量召回的 embedding 服务),
存储替换为 Redis;invalidate_doc 由知识库变更事件(MQ)触发。
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .models import FinalAnswer


def normalize_query(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"[,。,.!?！？\s]+", "", q)
    for filler in ("你好", "请问", "麻烦问下", "帮我看看", "一下"):
        q = q.replace(filler, "")
    return q


def _similarity(a: str, b: str) -> float:
    """字符集合 Jaccard(Mock);生产替换为 embedding 余弦。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class CacheEntry:
    normalized_query: str
    answer: FinalAnswer
    doc_ids: Set[str] = field(default_factory=set)
    valid: bool = True


@dataclass
class AnswerCache:
    sim_threshold: float = 0.85
    entries: List[CacheEntry] = field(default_factory=list)

    def lookup(self, normalized_query: str) -> Optional[FinalAnswer]:
        best, best_sim = None, 0.0
        for e in self.entries:
            if not e.valid:
                continue
            sim = _similarity(normalized_query, e.normalized_query)
            if sim > best_sim:
                best, best_sim = e, sim
        if best and best_sim >= self.sim_threshold:
            return copy.deepcopy(best.answer)      # 防调用方修改污染缓存
        return None

    def put(self, normalized_query: str, answer: FinalAnswer) -> None:
        if answer.degraded:      # 降级结果不入缓存
            return
        doc_ids = {s.chunk_id.split("#")[0] for s in answer.sources}
        self.entries.append(
            CacheEntry(normalized_query, copy.deepcopy(answer), doc_ids))

    def invalidate_doc(self, doc_id: str) -> int:
        """知识库变更事件触发:失效引用了该文档的所有缓存。"""
        n = 0
        for e in self.entries:
            if doc_id in e.doc_ids and e.valid:
                e.valid = False
                n += 1
        return n
