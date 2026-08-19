"""在运行时将 ``"pkg.mod:name"`` 格式的字符串解析为真实的 Python 对象。"""

from __future__ import annotations

import importlib
from typing import TypeVar, overload

T = TypeVar("T")


class ResolveError(Exception):
    """当点分路径无法被解析时抛出。"""


def _split_path(path: str) -> tuple[str, str]:
    """将 ``"pkg.mod:name"`` 拆分为模块路径和属性名。"""
    if ":" not in path:
        raise ResolveError(
            f"无效的导入路径 {path!r}，期望格式为 'pkg.mod:name'。"
        )
    module_path, _, attr_name = path.partition(":")
    if not module_path or not attr_name:
        raise ResolveError(
            f"无效的导入路径 {path!r}，模块名和属性名均不能为空。"
        )
    return module_path, attr_name


def _import_attr(path: str) -> object:
    module_path, attr_name = _split_path(path)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ResolveError(
            f"无法导入模块 {module_path!r}（来自路径 {path!r}）。"
            f"该包是否已安装？请执行：pip install {module_path.split('.')[0]}"
        ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ResolveError(
            f"模块 {module_path!r} 中不存在属性 {attr_name!r}。"
        ) from exc


@overload
def resolve_variable(path: str) -> object: ...
@overload
def resolve_variable(path: str, expected_type: type[T]) -> T: ...


def resolve_variable(path: str, expected_type: type[T] | None = None) -> T | object:
    """从 *path* 导入并返回任意对象。

    Parameters
    ----------
    path:
        ``"pkg.mod:name"`` 格式的点分导入路径字符串。
    expected_type:
        若提供，则使用 ``isinstance`` 验证解析到的对象类型。

    Raises
    ------
    ResolveError
        导入失败或类型不匹配时抛出。
    """
    obj = _import_attr(path)
    if expected_type is not None and not isinstance(obj, expected_type):
        raise ResolveError(
            f"路径 {path!r} 解析到的对象类型为 {type(obj).__name__}，"
            f"期望 {expected_type.__name__} 的实例。"
        )
    return obj  # type: ignore[return-value]


def resolve_class(path: str, base_class: type[T] | None = None) -> type[T]:
    """从 *path* 导入并返回一个**类**。

    Parameters
    ----------
    path:
        ``"pkg.mod:ClassName"`` 格式的点分导入路径字符串。
    base_class:
        若提供，则要求解析到的类必须是其子类。

    Raises
    ------
    ResolveError
        导入失败、解析结果不是类，或子类关系不匹配时抛出。
    """
    cls = _import_attr(path)
    if not isinstance(cls, type):
        raise ResolveError(f"路径 {path!r} 解析到的对象不是类（实际类型为 {type(cls).__name__}）。")
    if base_class is not None and not issubclass(cls, base_class):
        raise ResolveError(
            f"类 {cls.__name__} 不是 {base_class.__name__} 的子类。"
        )
    return cls  # type: ignore[return-value]
