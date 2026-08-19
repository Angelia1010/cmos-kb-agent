"""用于中间件排序约束的 @after / @before 装饰器。"""

from __future__ import annotations

from typing import Any

# 用于在中间件类上存储定位元数据的属性名。
_NEXT_ATTR = "__uniagent_next__"
_PREV_ATTR = "__uniagent_prev__"


def after(anchor: type) -> Any:
    """类装饰器：此中间件必须排在 *anchor* **之后**。

    使用示例::

        @after(SandboxMiddleware)
        class MyMiddleware(Middleware):
            ...
    """

    def decorator(cls: type) -> type:
        existing = getattr(cls, _NEXT_ATTR, set())
        setattr(cls, _NEXT_ATTR, existing | {anchor})
        return cls

    return decorator


def before(anchor: type) -> Any:
    """类装饰器：此中间件必须排在 *anchor* **之前**。

    使用示例::

        @before(TokenUsageMiddleware)
        class MyMiddleware(Middleware):
            ...
    """

    def decorator(cls: type) -> type:
        existing = getattr(cls, _PREV_ATTR, set())
        setattr(cls, _PREV_ATTR, existing | {anchor})
        return cls

    return decorator


def get_after_anchors(cls: type) -> set[type]:
    """返回 *cls* 必须排在其后的类型集合。"""
    return getattr(cls, _NEXT_ATTR, set())


def get_before_anchors(cls: type) -> set[type]:
    """返回 *cls* 必须排在其前的类型集合。"""
    return getattr(cls, _PREV_ATTR, set())
