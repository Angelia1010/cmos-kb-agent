"""应用级配置，支持 YAML 热重载与 ContextVar 栈隔离。"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from uniagent.config.sub_configs import (
    LoopConfig,
    ModelConfig,
    SkillConfig,
    StateConfig,
    ToolConfig,
    VerificationConfig,
)
from uniagent.config.reload_boundary import check_reload_safety

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _substitute_env(value: str) -> str:
    """将 *value* 中的 ``${VAR}`` 或 ``${VAR:default}`` 替换为对应的环境变量值。"""

    def _replacer(m: re.Match[str]) -> str:
        var_name = m.group(1)
        default = m.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        return m.group(0)  # 若无环境变量且无默认值则保留原样

    return _ENV_VAR_RE.sub(_replacer, value)


def _walk_substitute(obj: Any) -> Any:
    """递归地将 dict/list 结构中所有字符串的环境变量占位符展开。"""
    if isinstance(obj, str):
        return _substitute_env(obj)
    if isinstance(obj, dict):
        return {k: _walk_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_substitute(item) for item in obj]
    return obj


class AppConfig(BaseModel):
    """应用根配置模型。"""

    log_level: str = "info"
    models: list[ModelConfig] = Field(default_factory=lambda: [])
    tools: list[ToolConfig] = Field(default_factory=list)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    skills: SkillConfig = Field(default_factory=SkillConfig)
    extra_middleware: list[str] = Field(
        default_factory=list,
        description="额外中间件类的点分导入路径列表。",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """从 YAML 文件加载配置，并自动展开环境变量。"""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        raw = _walk_substitute(raw)
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# 基于文件签名的热重载
# ---------------------------------------------------------------------------

_FileSignature = tuple[float, int, str]  # (mtime, size, sha256-hex)


def _file_signature(path: Path) -> _FileSignature:
    stat = path.stat()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return (stat.st_mtime, stat.st_size, sha)


class _ConfigCache:
    """线程安全的单例缓存，用于支持热重载。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._sig: _FileSignature | None = None
        self._config: AppConfig | None = None

    def get(self, path: Path) -> AppConfig:
        # M9: 先在锁外做快速签名检查，减少锁内阻塞时间
        try:
            sig = _file_signature(path)
        except FileNotFoundError:
            with self._lock:
                if self._config is not None:
                    return self._config
                return AppConfig()

        with self._lock:
            if self._config is not None and self._path == path and self._sig == sig:
                return self._config

        # M9: 文件 I/O 和 YAML 解析在锁外执行
        new_config = AppConfig.from_yaml(path)

        with self._lock:
            # 双重检查，防止并发重复加载
            if self._config is not None and self._path == path and self._sig == sig:
                return self._config

            # 检查热重载边界安全性
            if self._config is not None:
                old_dict = self._config.model_dump()
                new_dict = new_config.model_dump()
                violations = check_reload_safety(old_dict, new_dict)
                if violations:
                    # 对违规字段保留旧值
                    for field_name in violations:
                        if field_name in old_dict:
                            new_dict[field_name] = old_dict[field_name]
                    new_config = AppConfig.model_validate(new_dict)

            self._path = path
            self._sig = sig
            self._config = new_config
            return self._config

    def invalidate(self) -> None:
        with self._lock:
            self._config = None
            self._sig = None


_cache = _ConfigCache()

# 默认配置文件路径（可通过环境变量覆盖）
_DEFAULT_CONFIG_PATH = Path(os.environ.get("UNIAGENT_CONFIG", "config.yaml"))


def get_app_config(path: str | Path | None = None) -> AppConfig:
    """返回当前的 ``AppConfig``，若 YAML 文件有变更则自动热重载。

    若 YAML 文件不存在，则返回默认的 ``AppConfig()``。
    """
    # ContextVar 中的覆盖配置优先级最高
    stack = _config_stack.get()
    if stack:
        return stack[-1]

    p = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    return _cache.get(p)


# ---------------------------------------------------------------------------
# 用于请求级别配置隔离的 ContextVar 栈
# ---------------------------------------------------------------------------

_config_stack: ContextVar[list[AppConfig]] = ContextVar(
    "uniagent_config_stack", default=[]
)


def push_current_app_config(config: AppConfig) -> None:
    """将 *config* 压入 ContextVar 栈（请求级别的配置覆盖）。"""
    stack = _config_stack.get().copy()
    stack.append(config)
    _config_stack.set(stack)


def pop_current_app_config() -> AppConfig:
    """从 ContextVar 栈中弹出顶部配置。"""
    stack = _config_stack.get().copy()
    if not stack:
        raise RuntimeError("配置栈为空——push 与 pop 调用不匹配。")
    config = stack.pop()
    _config_stack.set(stack)
    return config
