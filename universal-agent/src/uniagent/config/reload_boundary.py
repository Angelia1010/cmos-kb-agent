"""热重载边界的启动期字段注册表。

在此注册的字段在首次加载配置后将被冻结。热重载时若检测到这些字段发生变化，
将发出警告并忽略更改，而不是静默应用（后者可能破坏运行时状态，例如沙箱句柄）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 热重载时不允许变更的字段路径集合。
_STARTUP_ONLY_FIELDS: set[str] = set()


def register_startup_only(*field_paths: str) -> None:
    """将 *field_paths* 标记为启动期字段（首次加载后不可变）。"""
    _STARTUP_ONLY_FIELDS.update(field_paths)


def get_startup_only_fields() -> frozenset[str]:
    return frozenset(_STARTUP_ONLY_FIELDS)


def _deep_get(d: dict[str, Any], path: str) -> Any:
    """H13: 支持点分路径深度查找，如 "sandbox.type"。"""
    keys = path.split(".")
    current: Any = d
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def check_reload_safety(
    old: dict[str, object],
    new: dict[str, object],
) -> list[str]:
    """返回在 *old* 与 *new* 之间发生变化的启动期字段列表。

    H13: 支持点分路径（如 "sandbox.type"）进行深度字段查找。
    """
    violations: list[str] = []
    for field in _STARTUP_ONLY_FIELDS:
        old_val = _deep_get(old, field)
        new_val = _deep_get(new, field)
        if old_val != new_val:
            violations.append(field)
            logger.warning(
                "启动期字段 %r 在热重载时发生了变化（旧值=%r，新值=%r）。"
                "本次变更已被忽略——请重启服务以使其生效。",
                field,
                old_val,
                new_val,
            )
    return violations


# 注册默认的启动期字段。
register_startup_only("sandbox")
