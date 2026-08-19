"""基于 ContextVar 的用户上下文 IoC — 请求级作用域，无需参数透传。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Protocol, runtime_checkable


@runtime_checkable
class CurrentUser(Protocol):
    """在上下文中传递的用户身份对象协议。"""

    @property
    def user_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...


_current_user: ContextVar[CurrentUser | None] = ContextVar(
    "uniagent_current_user", default=None
)


def get_current_user() -> CurrentUser | None:
    """从 ContextVar 中返回当前用户（未设置时返回 None）。"""
    return _current_user.get()


def set_current_user(user: CurrentUser | None) -> None:
    """在 ContextVar 作用域中设置当前用户。"""
    _current_user.set(user)
