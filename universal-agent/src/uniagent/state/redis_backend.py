"""Redis 状态后端 —— 将每个逻辑键存为 Redis 字符串（JSON 序列化）。

依赖（可选）：``pip install redis``（>=4.2.0，已内置 asyncio 支持）。

键命名规则：
    Redis 键 = ``{key_prefix}:{logical_key}``
    例：``uniagent:state:agent:thread-abc123``

特性：
- 连接池复用（from_url 内置连接池）；
- 可选 TTL：0 = 永不过期，>0 = 自动淘汰（适合会话级状态）；
- list_keys 使用 SCAN 分页，不使用 KEYS * 阻塞命令；
- 连接失败 / 序列化错误均记录日志后传播，不静默吞掉。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from uniagent.state.backend import StateBackend

logger = logging.getLogger(__name__)


class RedisBackend(StateBackend):
    """基于 redis.asyncio 的 Redis 状态后端。

    参数
    ------
    url:
        Redis 连接 URL。
        明文：``redis://[:password@]host[:port][/db]``
        TLS ：``rediss://[:password@]host[:port][/db]``
        默认 ``redis://localhost:6379/0``。
    key_prefix:
        所有键的命名空间前缀，隔离不同应用/环境。
        默认 ``uniagent:state``。
    ttl:
        键生存时间（秒）。0 = 永不过期；>0 则写入时附带 EX 参数。
        默认 ``0``。

    使用示例
    --------
    直接实例化::

        from uniagent.state.redis_backend import RedisBackend
        backend = RedisBackend("redis://localhost:6379/0",
                               key_prefix="myapp:state", ttl=3600)
        await backend.save("session:xyz", {"step": 3})
        data = await backend.load("session:xyz")
        await backend.close()

    通过 StateConfig / get_backend 工厂（推荐）::

        from uniagent.state.backend import get_backend
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig(backend="redis",
                          redis_url="redis://localhost:6379/0",
                          key_prefix="myapp:state",
                          ttl=3600)
        backend = get_backend(cfg)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "uniagent:state",
        ttl: int = 0,
    ) -> None:
        try:
            import redis.asyncio as aioredis  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "RedisBackend 需要安装 redis 包：pip install redis"
            ) from exc

        self._prefix = key_prefix.rstrip(":")
        self._ttl = max(0, int(ttl))
        # from_url 内部维护连接池，无需手动管理
        self._client = aioredis.from_url(url, decode_responses=True)
        logger.debug(
            "RedisBackend 初始化完成：url=%r  prefix=%r  ttl=%ds",
            url, self._prefix, self._ttl,
        )

    # ── 内部键转换 ──────────────────────────────────────────────────────── #

    def _rkey(self, logical_key: str) -> str:
        """逻辑键 → Redis 键（附加命名空间前缀）。"""
        return f"{self._prefix}:{logical_key}"

    def _lkey(self, redis_key: str) -> str:
        """Redis 键 → 逻辑键（去掉命名空间前缀）。"""
        prefix_sep = f"{self._prefix}:"
        return redis_key[len(prefix_sep):] if redis_key.startswith(prefix_sep) else redis_key

    # ── StateBackend 接口实现 ─────────────────────────────────────────────── #

    async def load(self, key: str) -> dict[str, Any] | None:
        """从 Redis 读取并反序列化状态；键不存在时返回 None。"""
        try:
            raw = await self._client.get(self._rkey(key))
        except Exception as exc:
            logger.warning("RedisBackend.load(%r) 失败：%s", key, exc)
            raise
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "RedisBackend.load(%r)：JSON 解析失败，原始值已损坏：%s", key, exc
            )
            return None

    async def save(self, key: str, data: dict[str, Any]) -> None:
        """将状态序列化为 JSON 并写入 Redis（可选 TTL）。"""
        raw = json.dumps(data, ensure_ascii=False, default=str)
        rkey = self._rkey(key)
        try:
            if self._ttl > 0:
                await self._client.set(rkey, raw, ex=self._ttl)
            else:
                await self._client.set(rkey, raw)
        except Exception as exc:
            logger.error("RedisBackend.save(%r) 失败：%s", key, exc)
            raise

    async def delete(self, key: str) -> None:
        """删除指定键；键不存在时静默跳过。"""
        try:
            await self._client.delete(self._rkey(key))
        except Exception as exc:
            logger.error("RedisBackend.delete(%r) 失败：%s", key, exc)
            raise

    async def list_keys(self, prefix: str = "") -> list[str]:
        """使用 SCAN 分页枚举所有匹配的逻辑键（避免阻塞服务器）。

        参数
        ------
        prefix:
            逻辑键前缀过滤，例如 ``"agent:"`` 只返回以 "agent:" 开头的键。
        """
        pattern = f"{self._prefix}:{prefix}*"
        keys: list[str] = []
        try:
            async for rkey in self._client.scan_iter(match=pattern, count=100):
                keys.append(self._lkey(rkey))
        except Exception as exc:
            logger.error(
                "RedisBackend.list_keys(prefix=%r) 失败：%s", prefix, exc
            )
            raise
        return keys

    # ── 生命周期 ─────────────────────────────────────────────────────────── #

    async def close(self) -> None:
        """关闭 Redis 连接池（进程退出前调用）。"""
        await self._client.aclose()
        logger.debug("RedisBackend 连接池已关闭")
