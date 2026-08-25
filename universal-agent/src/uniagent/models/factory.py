"""模型工厂 — 从 ``ModelConfig`` 构建并缓存 ``BaseChatModel`` 实例。"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from uniagent.imports.resolvers import resolve_class

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from uniagent.config.sub_configs import ModelConfig


class ModelFactory:
    """从 ``ModelConfig`` 构建 LangChain ``BaseChatModel`` 的工厂。

    实例级别的缓存：相同 *name* 的模型只构建一次，直到 ``invalidate()`` 被调用。
    线程安全（内部使用 ``threading.Lock``）。
    """

    def __init__(self) -> None:
        self._cache: dict[str, "BaseChatModel"] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def build(self, config: "ModelConfig") -> "BaseChatModel":
        """从 ``ModelConfig`` 构建一个新的模型实例（不使用缓存）。

        参数注入优先级（低 → 高）：
        ``api_key / base_url / timeout / max_retries / extra_headers``
        → 最后被 ``config.kwargs`` 覆盖。
        """
        model_cls = resolve_class(config.use)
        init_kwargs: dict[str, Any] = {}

        if config.api_key:
            init_kwargs["api_key"] = config.api_key
        if config.base_url:
            init_kwargs["base_url"] = config.base_url
        if config.timeout > 0:
            init_kwargs["timeout"] = config.timeout
        if config.max_retries >= 0:
            init_kwargs["max_retries"] = config.max_retries
        if config.extra_headers:
            init_kwargs["default_headers"] = config.extra_headers

        # kwargs 优先级最高，可覆盖上述任意字段
        init_kwargs.update(config.kwargs)

        return model_cls(
            model=config.model,
            temperature=config.temperature,
            **init_kwargs,
        )

    def get(self, name: str, config: "ModelConfig") -> "BaseChatModel":
        """按名称获取模型实例（带缓存）。

        若缓存中已存在同名实例则直接返回，否则调用 ``build()`` 并缓存结果。
        """
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self.build(config)
            return self._cache[name]

    def invalidate(self, name: str | None = None) -> None:
        """清除缓存（用于热重载）。

        Parameters
        ----------
        name:
            为 ``None`` 时清除所有缓存；否则仅清除指定名称的缓存。
        """
        with self._lock:
            if name is None:
                self._cache.clear()
            else:
                self._cache.pop(name, None)


# ---------------------------------------------------------------------------
# 模块级单例与便捷函数
# ---------------------------------------------------------------------------

_default_factory = ModelFactory()


def build_model(config: "ModelConfig") -> "BaseChatModel":
    """使用默认工厂从 ``ModelConfig`` 构建新模型实例（不使用缓存）。"""
    return _default_factory.build(config)


def get_model(name: str = "default", cfg: Any = None) -> "BaseChatModel":
    """使用默认工厂按名称获取模型实例（带缓存）。

    Parameters
    ----------
    name:
        模型名称，对应 ``AppConfig.models[*].name``。
    cfg:
        ``AppConfig`` 实例；为 ``None`` 时延迟加载全局配置（避免循环依赖）。
    """
    if cfg is None:
        # 延迟导入，避免 models ←→ config 循环依赖
        from uniagent.config.app_config import get_app_config
        cfg = get_app_config()

    if not cfg.models:
        raise ValueError(
            "未配置任何模型。请在配置的 'models' 中至少添加一个条目。"
        )

    # 优先按 name 字段查找，找不到时回退到第一个模型
    mc = None
    for m in cfg.models:
        if getattr(m, "name", None) == name:
            mc = m
            break
    if mc is None:
        mc = cfg.models[0]

    return _default_factory.get(name, mc)
