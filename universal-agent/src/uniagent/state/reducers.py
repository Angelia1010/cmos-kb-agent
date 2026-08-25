"""Annotated 状态字段的通用归约器工厂。"""

from __future__ import annotations

import json
from typing import Any, TypeVar

T = TypeVar("T")


def last_wins(existing: T, new: T) -> T:
    """简单的最后写入胜出归约器。"""
    return new


def idempotent_merge(
    existing: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:
    """将 *new* 合并到 *existing* 中。已有键除非被覆盖，否则保持不变。"""
    merged = {**existing, **new}
    return merged


def dedup_list_merge(existing: list[T], new: list[T]) -> list[T]:
    """将 *new* 中不在 *existing* 里的项追加进去（保持顺序）。"""
    seen = set()
    result: list[T] = []
    for item in existing:
        key = _hashable(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    for item in new:
        key = _hashable(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result



def _hashable(item: Any) -> Any:
    """尽力而为的可哈希键，用于去重。

    M7: 对 dict/list 使用 json 序列化而非 id()，确保内容相同的对象
    被正确识别为重复项。
    """
    try:
        hash(item)
        return item
    except TypeError:
        try:
            return json.dumps(item, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return id(item)
