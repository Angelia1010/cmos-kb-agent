# -*- coding: utf-8 -*-
"""Retrieval Chunk → Processing 内部候选的边界桥接。"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from ..models import Chunk


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _applicability_of(raw: Dict[str, Any]) -> Dict[str, Any]:
    """从 ngkm 条目/原子映射适用性字段;只映射明确存在的字段,不猜测。"""
    nested = raw.get("applicability")
    applicability: Dict[str, Any] = (
        copy.deepcopy(nested) if isinstance(nested, dict) else {}
    )
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


def _chunk_to_candidate(index: int, chunk: Chunk) -> Dict[str, Any]:
    extra = chunk.extra if isinstance(chunk.extra, dict) else {}
    raw_atoms = extra.get("atoms")
    atoms: List[Dict[str, Any]] = []
    if isinstance(raw_atoms, (list, tuple)):
        for atom in raw_atoms:
            if not isinstance(atom, dict) or atom.get("error"):
                continue
            atoms.append(_merged_atom_to_dict(chunk.doc_id, len(atoms), atom))
    # 候选正文:结构化 ngkm 条目走 extra["atoms"](已映射进 atoms);
    # 离线 MockES / 关键词回退等非结构化 chunk 只有 chunk.content —— 必须
    # 落到候选 content,否则 has_renderable_candidate_content 判空,filter
    # 会以 empty_content 把全部候选丢弃,导致 Top3 恒空、验证 loop 永远走不通。
    return {
        "chunk_id": chunk.chunk_id,
        "knowledge_id": chunk.doc_id,
        "knowledge_name": chunk.doc_title,
        "content": _text(chunk.content) or "",
        "retrieval_rank": index + 1,
        "source_index": index,
        "retrieval_score": chunk.score,
        "matched_atom_ids": copy.deepcopy(
            extra.get("matched_atom_ids", extra.get("matchedAtomIds", []))
        ),
        "source_routes": copy.deepcopy(
            extra.get("source_routes", extra.get("sourceRoutes", []))
        ),
        "knowledge_type": _text(
            extra.get("knowledge_type", extra.get("knowledgeType"))
        ),
        "template_id": _text(
            extra.get("template_id", extra.get("templateId"))
        ),
        "applicability": _applicability_of(extra),
        "atoms": atoms,
    }


def index_source_chunks(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    """按 chunk_id 建立当前 Processing 调用的原 Chunk 映射并拒绝重复 ID。"""
    source_chunks: Dict[str, Chunk] = {}
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Chunk):
            raise TypeError(f"chunks[{index}] 不是 Chunk")
        if not chunk.chunk_id:
            raise ValueError(f"chunks[{index}] 缺少 chunk_id")
        if chunk.chunk_id in source_chunks:
            raise ValueError(f"重复 chunk_id: {chunk.chunk_id}")
        source_chunks[chunk.chunk_id] = chunk
    return source_chunks


def retrieval_to_candidates(
    merged: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[Sequence[Chunk]] = None,
) -> List[Dict[str, Any]]:
    """按输入顺序把 Retrieval Chunk 映射为 Processing 候选。

    ``merged`` 仅为保持 MainAgent 已提交调用签名而保留；当前主链只允许
    ``merged_results → chunks → candidates``，不再直接转换 merged。
    """
    del merged
    if chunks is None:
        raise ValueError("缺少 Retrieval chunks，不能直接从 merged_results 构造候选")
    index_source_chunks(chunks)
    return [_chunk_to_candidate(index, chunk) for index, chunk in enumerate(chunks)]
