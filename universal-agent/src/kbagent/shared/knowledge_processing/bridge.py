# -*- coding: utf-8 -*-
"""检索产物 → Processing 标准候选的边界桥接。

主链路检索阶段会产出两类工件:
- merged_results: 生产一体化流水线(intergrate_all)的原始 camelCase 条目;
- chunks:         merged_to_chunks / coarse_recall 的标准 Chunk 列表。

处理阶段(analyze → filter → build_markdown → rerank)只认蛇形命名的
标准候选(knowledge_id/knowledge_name/atoms[...]),本模块完成映射:
有 merged 时按知识拆原子(camelCase → snake_case);
离线环境无 merged 时把每个 Chunk 适配为一条单原子候选。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from ..models import Chunk


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _applicability_of(raw: Dict[str, Any]) -> Dict[str, Any]:
    """从 ngkm 条目/原子映射适用性字段;只映射明确存在的字段,不猜测。"""
    applicability: Dict[str, Any] = {}
    status = _text(raw.get("statusCode") or raw.get("status"))
    if status:
        applicability["status"] = status
    channels = raw.get("channelCodes") or raw.get("channelCode") \
        or raw.get("channel_codes")
    if channels not in (None, ""):
        applicability["channel_codes"] = channels
    return applicability


def _merged_atom_to_dict(knowledge_id: str, position: int,
                         atom: Dict[str, Any]) -> Dict[str, Any]:
    except_value = atom.get("except", atom.get("except_rules"))
    annotation = atom.get("annotation")
    return {
        "atom_id": _text(atom.get("klgAttrAtomId") or atom.get("atom_id"))
                   or f"{knowledge_id}-ATOM-{position + 1:03d}",
        "param_name": _text(atom.get("paramName") or atom.get("param_name")),
        "param_type": _text(atom.get("paramType") or atom.get("param_type")),
        "group_id": _text(atom.get("groupId") or atom.get("group_id")),
        "content": copy.deepcopy(atom.get("content"))
                   if atom.get("content") is not None else "",
        "except_rules": copy.deepcopy(except_value)
                        if except_value not in (None, "") else [],
        "annotation": copy.deepcopy(annotation)
                      if annotation not in (None, "", [], {}) else None,
        "arrange_seq_number": atom.get(
            "arrangeSeqNumber", atom.get("arrange_seq_number")) or position + 1,
        "wkuntt": _text(atom.get("wkuntt") or atom.get("unit")),
        "applicability": _applicability_of(atom),
    }


def _merged_entry_to_candidate(index: int, entry: Dict[str, Any]) -> Dict[str, Any]:
    kid = _text(entry.get("knowledgeId") or entry.get("knowledge_id")) \
        or f"MERGED-{index + 1:03d}"
    atoms: List[Dict[str, Any]] = []
    for atom in entry.get("atoms") or []:
        if not isinstance(atom, dict) or atom.get("error"):
            continue
        atoms.append(_merged_atom_to_dict(kid, len(atoms), atom))
    return {
        "knowledge_id": kid,
        "knowledge_name": _text(
            entry.get("knowledgeName") or entry.get("knowledge_name"))
            or f"未命名知识-{kid}",
        # 生产侧无显式相关性得分,留空由下游按缺分处理
        "retrieval_rank": index + 1,
        "retrieval_score": None,
        "applicability": _applicability_of(entry),
        "atoms": atoms,
    }


def _chunk_to_candidate(index: int, chunk: Chunk) -> Dict[str, Any]:
    kid = chunk.chunk_id or f"CHUNK-{index + 1:03d}"
    return {
        "knowledge_id": kid,
        "knowledge_name": chunk.doc_title or kid,
        "retrieval_rank": index + 1,
        "retrieval_score": chunk.score,
        "applicability": _applicability_of(chunk.extra or {}),
        "atoms": [{
            "atom_id": f"{kid}-ATOM-001",
            "param_name": "业务内容",
            "content": chunk.content,
            "arrange_seq_number": 1,
            "applicability": {},
        }],
    }


def retrieval_to_candidates(merged: Optional[List[Dict[str, Any]]] = None,
                            chunks: Optional[List[Chunk]] = None,
                            ) -> List[Dict[str, Any]]:
    """检索产物 → Processing 标准候选列表。

    优先使用一体化流水线的 merged 原始条目;无 merged(离线 coarse_recall
    路径)时退化为 Chunk 适配,一条 Chunk 映射一条单原子候选。
    """
    if merged:
        return [_merged_entry_to_candidate(i, e)
                for i, e in enumerate(merged) if isinstance(e, dict)]
    return [_chunk_to_candidate(i, c) for i, c in enumerate(chunks or [])]
