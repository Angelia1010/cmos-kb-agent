# -*- coding: utf-8 -*-
"""数据处理子智能体工具集。"""
from __future__ import annotations

import json
import re
from typing import Dict, List

from langchain_core.tools import tool

from ..shared.models import Chunk
from ..shared.workspace import get_workspace


def _obs(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False, default=str)


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
    """确定性保底流水线:子智能体无有效产出时调用。"""
    clean_data.func()
    denoise_data.func()
    dedupe_data.func()
    structure_data.func()
    sort_data.func()
