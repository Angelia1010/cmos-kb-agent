"""状态后端协议 —— 外部状态的可插拔持久化层。

默认实现使用本地文件系统（JSON 文件）。
可替换为 S3、Redis、数据库等。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateBackend(ABC):
    """外部状态的抽象持久化后端。"""

    @abstractmethod
    async def load(self, key: str) -> dict[str, Any] | None:
        """按键加载状态。未找到时返回 None。"""

    @abstractmethod
    async def save(self, key: str, data: dict[str, Any]) -> None:
        """将状态持久化到指定键。"""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """按键删除状态。"""

    @abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """列出所有键，支持可选前缀过滤。"""


class LocalFileBackend(StateBackend):
    """本地文件系统后端 —— 将每个键存储为一个 JSON 文件。

    H9 修复：
    - 使用 SHA256 编码 key 为文件名，避免 "a/b" 和 "a_b" 碰撞。
    - 使用原子写入（先写临时文件再重命名），防止崩溃导致文件损坏。
    - 维护 key→hash 的映射文件用于 list_keys 反查。

    目录结构::

        {state_dir}/
        ├── _keymap.json            ← key→hash 映射
        ├── a1b2c3d4e5f6....json    ← SHA256 编码的文件名
        └── ...
    """

    def __init__(self, state_dir: str = ".uniagent/state") -> None:
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keymap_path = self._dir / "_keymap.json"
        self._keymap: dict[str, str] = self._load_keymap()

    def _load_keymap(self) -> dict[str, str]:
        """加载 key→hash 映射。"""
        if self._keymap_path.is_file():
            try:
                return json.loads(self._keymap_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_keymap(self) -> None:
        """原子写入 keymap。"""
        self._atomic_write(
            self._keymap_path,
            json.dumps(self._keymap, ensure_ascii=False, indent=2),
        )

    def _path(self, key: str) -> Path:
        # H9: 用 SHA256 编码 key 为安全文件名，避免碰撞
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self._dir / f"{key_hash}.json"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """H9: 先写临时文件再原子重命名，防止崩溃导致文件损坏。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".state_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            # Windows 上需要先删除目标文件才能重命名
            tmp = Path(tmp_path)
            tmp.replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    async def load(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("加载状态 %r 失败：%s", key, exc)
            return None

    async def save(self, key: str, data: dict[str, Any]) -> None:
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        self._atomic_write(self._path(key), content)
        # 更新 keymap
        self._keymap[key] = self._path(key).stem
        self._save_keymap()

    async def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
        self._keymap.pop(key, None)
        self._save_keymap()

    async def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        for key in self._keymap:
            if not prefix or key.startswith(prefix):
                keys.append(key)
        return keys
