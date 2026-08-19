"""基于锚点的拓扑排序中间件链组装。"""

from __future__ import annotations

import logging
from typing import Sequence

from uniagent.middleware.base import Middleware
from uniagent.middleware.positioning import get_after_anchors, get_before_anchors

logger = logging.getLogger(__name__)


class MiddlewareChainError(Exception):
    """当中间件链存在冲突或循环时抛出。"""


def assemble_middleware_chain(
    built_in: Sequence[Middleware],
    extras: Sequence[Middleware] | None = None,
    tail_type: type | None = None,
) -> list[Middleware]:
    """从 *built_in* 和 *extras* 构建有序的中间件列表。

    步骤：
    1. 以 *built_in* 列表为基础（默认保留顺序）。
    2. 按照 ``@after`` / ``@before`` 锚点约束依次插入 *extras* 中的中间件。
    3. 若指定了 *tail_type*，确保匹配的中间件排在最后。
    4. 检测循环依赖，若发现则抛出 ``MiddlewareChainError``。

    返回
    -------
    list[Middleware]
        按执行顺序排列的中间件链。
    """
    chain = list(built_in)
    if extras:
        for mw in extras:
            _insert_with_anchors(chain, mw)

    # 若指定了尾部类型，将其移至末尾
    if tail_type is not None:
        tail_items = [m for m in chain if isinstance(m, tail_type)]
        if tail_items:
            for t in tail_items:
                chain.remove(t)
            chain.extend(tail_items)

    _detect_cycles(chain)
    logger.debug(
        "中间件链：%s",
        " → ".join(m.name or type(m).__name__ for m in chain),
    )
    return chain


def _insert_with_anchors(chain: list[Middleware], mw: Middleware) -> None:
    """按照 @after/@before 约束将 *mw* 插入 *chain*。"""
    cls = type(mw)
    next_anchors = get_after_anchors(cls)  # 必须排在这些类型之后
    prev_anchors = get_before_anchors(cls)  # 必须排在这些类型之前

    # 找到最晚的"必须排在其后"位置
    after_idx = -1
    for anchor in next_anchors:
        for i, existing in enumerate(chain):
            if isinstance(existing, anchor):
                after_idx = max(after_idx, i)

    # 找到最早的"必须排在其前"位置
    before_idx = len(chain)
    for anchor in prev_anchors:
        for i, existing in enumerate(chain):
            if isinstance(existing, anchor):
                before_idx = min(before_idx, i)

    if after_idx >= before_idx:
        raise MiddlewareChainError(
            f"{cls.__name__} 的锚点冲突：必须在索引 {after_idx} 之后，"
            f"但也必须在索引 {before_idx} 之前。"
        )

    insert_pos = after_idx + 1
    chain.insert(insert_pos, mw)


def _detect_cycles(chain: list[Middleware]) -> None:
    """验证最终链满足所有锚点约束。"""
    index_map: dict[type, int] = {}
    for i, mw in enumerate(chain):
        index_map[type(mw)] = i

    for i, mw in enumerate(chain):
        cls = type(mw)
        for anchor in get_after_anchors(cls):
            anchor_idx = index_map.get(anchor)
            if anchor_idx is not None and anchor_idx >= i:
                raise MiddlewareChainError(
                    f"循环/违规：{cls.__name__}（位置 {i}）必须排在 "
                    f"{anchor.__name__}（位置 {anchor_idx}）之后。"
                )
        for anchor in get_before_anchors(cls):
            anchor_idx = index_map.get(anchor)
            if anchor_idx is not None and anchor_idx <= i:
                raise MiddlewareChainError(
                    f"循环/违规：{cls.__name__}（位置 {i}）必须排在 "
                    f"{anchor.__name__}（位置 {anchor_idx}）之前。"
                )
