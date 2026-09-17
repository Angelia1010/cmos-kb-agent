# -*- coding: utf-8 -*-
"""文档内证据片段定位(答案子智能体新增能力)。

背景:检索阶段把**整篇知识文档**作为 chunk.content 传入。本模块对每篇文档,
让大模型判断哪些片段能回答用户问题,并把片段**原文逐字**摘出(带 start/end 偏移)。

可溯源保证:模型返回的片段文本一律经代码侧校验——必须是文档原文的连续子串;
命中后以 content[start:end] 回填,从定义上保证逐字。非原文(改写/概括/幻觉)的
片段直接丢弃,绝不流入下游。

与答案生成相互独立(additive):本模块只产出定位结果,不改 [TASK:answer] 的素材。
使用标准 model.invoke([SystemMessage, HumanMessage]) 调用,任意 BaseChatModel 可直连。
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

from ..shared.models import Chunk, DocFragments, LocatedFragment
from .generate import _invoke_json

logger = logging.getLogger("kbagent.answer")

# 单篇文档最多保留的证据片段数,防止模型返回过多导致响应膨胀
MAX_FRAGMENTS_PER_DOC = 5

_LOCATE_SYSTEM = (
    "[TASK:locate_fragments] 你是10086坐席辅助助手。给定用户问题和一篇知识文档全文,"
    "判断文档中哪些连续片段能回答该问题,并把这些片段从原文中逐字摘出。输出 JSON:"
    '{"answerable": bool(该文档是否能回答用户问题), '
    '"relevance": int(该文档对回答用户问题的支撑程度,0-100 整数:'
    "能直接完整回答→80-100,只覆盖部分→40-79,仅沾边→1-39,无关→0), "
    '"fragments": [{"text": str(能回答问题的原文连续片段), "reason": str(该片段为何能回答问题,简述)}]}。'
    "硬性要求:text 必须是文档中真实出现的连续原文,**逐字复制**,禁止改写、概括、翻译、"
    "拼接或添加省略号;只摘与问题直接相关的片段,无关内容不要摘;"
    "若文档无法回答该问题,answerable=false 且 fragments=[]。只输出 JSON。"
    "字符串值内禁止出现未转义的英文双引号(引用词语请用中文引号“”),值内禁止换行。"
)


def _parse_relevance(data: Any) -> int:
    """解析模型自评相关度,夹紧到 0-100;非法值按 0(调用方另有兜底)。"""
    try:
        return max(0, min(100, int(float((data or {}).get("relevance", 0)))))
    except (TypeError, ValueError, AttributeError):
        return 0


def _locate_span(content: str, text: str) -> Optional[Tuple[int, int]]:
    """在 content 中定位 text 的逐字位置,返回 (start, end) 或 None。

    两级匹配:
      1) 精确子串:content.find(text);
      2) 空白弹性兜底:把片段按空白切词、用 \\s+ 连接成正则再 search,
         以吸收换行/多空格差异(命中后取原文 span,仍保证逐字)。
    都失败则返回 None(视为非原文,调用方丢弃)。
    """
    text = text.strip()
    if not text or not content:
        return None
    idx = content.find(text)
    if idx >= 0:
        return idx, idx + len(text)
    # —— 空白弹性兜底 ——
    tokens = text.split()
    if len(tokens) > 1:
        pattern = r"\s+".join(re.escape(t) for t in tokens)
        m = re.search(pattern, content)
        if m:
            return m.start(), m.end()
    return None


def locate_fragments(model: Any, query: str, chunk: Chunk) -> DocFragments:
    """对单篇文档定位能回答 query 的原文逐字片段。

    返回 DocFragments(answerable + 已校验的逐字片段列表)。
    answerable 以**可验证片段**为准:模型自评 answerable 但摘不出原文片段时,
    视为不可用(answerable=False),保证暴露给坐席的片段都真实可溯源。
    """
    data = _invoke_json(
        model, _LOCATE_SYSTEM,
        f"用户问题:{query}\n文档内容:\n{chunk.content}")

    raw_fragments = data.get("fragments") if isinstance(data, dict) else None
    raw_fragments = raw_fragments if isinstance(raw_fragments, list) else []
    llm_answerable = bool(data.get("answerable", False)) if isinstance(data, dict) else False

    fragments: List[LocatedFragment] = []
    for item in raw_fragments:
        if len(fragments) >= MAX_FRAGMENTS_PER_DOC:
            break
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        span = _locate_span(chunk.content, text)
        if span is None:
            logger.warning("locate 片段非原文逐字,已丢弃 chunk_id=%s text=%r",
                           chunk.chunk_id, text[:60])
            continue
        start, end = span
        fragments.append(LocatedFragment(
            text=chunk.content[start:end],   # 以原文回填,从定义上保证逐字
            start=start, end=end,
            reason=str(item.get("reason", "")).strip(),
        ))

    answerable = bool(fragments)
    if llm_answerable and not answerable:
        logger.info("locate 模型自评可答但无可验证原文片段 → 判为不可答 chunk_id=%s",
                    chunk.chunk_id)
    relevance = _parse_relevance(data)
    # 无可验证片段时相关度不应偏高(模型自评不可信),压到 39 以下
    if not answerable:
        relevance = min(relevance, 39)
    logger.info("locate 完成 chunk_id=%s answerable=%s relevance=%d 片段=%d/%d",
                chunk.chunk_id, answerable, relevance, len(fragments),
                len(raw_fragments))
    return DocFragments(
        chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, doc_title=chunk.doc_title,
        answerable=answerable, relevance=relevance, fragments=fragments,
    )
