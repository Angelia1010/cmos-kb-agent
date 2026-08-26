# -*- coding: utf-8 -*-
"""RedisBackend 测试（使用 AsyncMock，无需真实 Redis 服务）

覆盖范围：
  T1  键名转换（_rkey / _lkey）
  T2  load  — GET、不存在返回 None、JSON 损坏返回 None、连接异常向上传播
  T3  save  — SET / SET EX（有 TTL）
  T4  delete — DEL
  T5  list_keys — SCAN 分页、前缀过滤、去前缀
  T6  get_backend 工厂 — local / redis / 无效别名
  T7  StateConfig 新字段默认值与校验

运行方式：
  PYTHONPATH=src python -m unittest tests.test_state_redis -v
"""
import asyncio
import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, "src")

# ─────────────────────────────── helpers ────────────────────────────────── #

def _run(coro):
    return asyncio.run(coro)


def _make_backend(url="redis://localhost:6379/0",
                  key_prefix="uniagent:state",
                  ttl=0,
                  mock_client=None):
    """创建 RedisBackend 并注入 mock 客户端，不真正连接 Redis。"""
    from uniagent.state.redis_backend import RedisBackend

    if mock_client is None:
        mock_client = MagicMock()

    with patch("redis.asyncio.from_url", return_value=mock_client):
        backend = RedisBackend(url=url, key_prefix=key_prefix, ttl=ttl)
    return backend, mock_client


# ══════════════════════════════════════════════════════════════════════════ #
#  T1  键名转换                                                              #
# ══════════════════════════════════════════════════════════════════════════ #

class TestKeyTransform(unittest.TestCase):

    def setUp(self):
        self.backend, _ = _make_backend(key_prefix="myapp:state")

    def test_rkey_adds_prefix(self):
        self.assertEqual(
            self.backend._rkey("agent:123"),
            "myapp:state:agent:123",
        )

    def test_rkey_simple_key(self):
        self.assertEqual(self.backend._rkey("foo"), "myapp:state:foo")

    def test_lkey_strips_prefix(self):
        self.assertEqual(
            self.backend._lkey("myapp:state:agent:123"),
            "agent:123",
        )

    def test_lkey_no_prefix_passthrough(self):
        """不以 prefix: 开头的 redis key 原样返回，防止崩溃。"""
        self.assertEqual(self.backend._lkey("other:key"), "other:key")

    def test_prefix_trailing_colon_stripped(self):
        """构造时 key_prefix 末尾的 ':' 应被去掉，避免双冒号。"""
        backend, _ = _make_backend(key_prefix="ns:")
        self.assertEqual(backend._prefix, "ns")
        self.assertEqual(backend._rkey("k"), "ns:k")


# ══════════════════════════════════════════════════════════════════════════ #
#  T2  load                                                                  #
# ══════════════════════════════════════════════════════════════════════════ #

class TestLoad(unittest.TestCase):

    def test_load_returns_dict_on_hit(self):
        data = {"step": 3, "score": 0.87}
        mc = MagicMock()
        mc.get = AsyncMock(return_value=json.dumps(data))
        backend, _ = _make_backend(mock_client=mc)
        result = _run(backend.load("sess:abc"))
        self.assertEqual(result, data)
        mc.get.assert_awaited_once_with("uniagent:state:sess:abc")

    def test_load_returns_none_on_miss(self):
        mc = MagicMock()
        mc.get = AsyncMock(return_value=None)
        backend, _ = _make_backend(mock_client=mc)
        result = _run(backend.load("sess:miss"))
        self.assertIsNone(result)

    def test_load_returns_none_on_json_error(self):
        """Redis 中存了损坏数据时，load 应返回 None 而非崩溃。"""
        mc = MagicMock()
        mc.get = AsyncMock(return_value="this is not json{{")
        backend, _ = _make_backend(mock_client=mc)
        result = _run(backend.load("sess:bad"))
        self.assertIsNone(result)

    def test_load_propagates_connection_error(self):
        """Redis 连接失败应向上传播，不被静默吞掉。"""
        from redis.exceptions import ConnectionError as RedisConnErr
        mc = MagicMock()
        mc.get = AsyncMock(side_effect=RedisConnErr("timeout"))
        backend, _ = _make_backend(mock_client=mc)
        with self.assertRaises(RedisConnErr):
            _run(backend.load("sess:abc"))


# ══════════════════════════════════════════════════════════════════════════ #
#  T3  save                                                                  #
# ══════════════════════════════════════════════════════════════════════════ #

class TestSave(unittest.TestCase):

    def test_save_calls_set_without_ttl(self):
        mc = MagicMock()
        mc.set = AsyncMock()
        backend, _ = _make_backend(ttl=0, mock_client=mc)
        data = {"k": "v", "n": 42}
        _run(backend.save("agent:001", data))
        mc.set.assert_awaited_once()
        call_args = mc.set.call_args
        # 第一个位置参数是 redis key
        self.assertEqual(call_args.args[0], "uniagent:state:agent:001")
        # 值是 JSON 字符串
        saved = json.loads(call_args.args[1])
        self.assertEqual(saved, data)
        # 无 TTL 时不传 ex 参数
        self.assertNotIn("ex", call_args.kwargs)

    def test_save_calls_set_with_ex_when_ttl_set(self):
        mc = MagicMock()
        mc.set = AsyncMock()
        backend, _ = _make_backend(ttl=3600, mock_client=mc)
        _run(backend.save("sess:xyz", {"foo": "bar"}))
        call_kwargs = mc.set.call_args.kwargs
        self.assertEqual(call_kwargs.get("ex"), 3600)

    def test_save_serializes_non_json_types_with_str(self):
        """datetime 等不可序列化类型应通过 default=str 转换，不崩溃。"""
        from datetime import datetime
        mc = MagicMock()
        mc.set = AsyncMock()
        backend, _ = _make_backend(mock_client=mc)
        _run(backend.save("t", {"ts": datetime(2026, 1, 1)}))
        raw = mc.set.call_args.args[1]
        self.assertIn("2026", raw)

    def test_save_propagates_connection_error(self):
        from redis.exceptions import ConnectionError as RedisConnErr
        mc = MagicMock()
        mc.set = AsyncMock(side_effect=RedisConnErr("refused"))
        backend, _ = _make_backend(mock_client=mc)
        with self.assertRaises(RedisConnErr):
            _run(backend.save("x", {}))


# ══════════════════════════════════════════════════════════════════════════ #
#  T4  delete                                                                #
# ══════════════════════════════════════════════════════════════════════════ #

class TestDelete(unittest.TestCase):

    def test_delete_calls_del_with_correct_key(self):
        mc = MagicMock()
        mc.delete = AsyncMock()
        backend, _ = _make_backend(mock_client=mc)
        _run(backend.delete("agent:to_remove"))
        mc.delete.assert_awaited_once_with("uniagent:state:agent:to_remove")

    def test_delete_propagates_error(self):
        from redis.exceptions import ConnectionError as RedisConnErr
        mc = MagicMock()
        mc.delete = AsyncMock(side_effect=RedisConnErr("refused"))
        backend, _ = _make_backend(mock_client=mc)
        with self.assertRaises(RedisConnErr):
            _run(backend.delete("x"))


# ══════════════════════════════════════════════════════════════════════════ #
#  T5  list_keys                                                             #
# ══════════════════════════════════════════════════════════════════════════ #

async def _async_generator(items):
    for item in items:
        yield item


class TestListKeys(unittest.TestCase):

    def _make_scan_client(self, redis_keys):
        mc = MagicMock()
        mc.scan_iter = MagicMock(return_value=_async_generator(redis_keys))
        return mc

    def test_list_keys_strips_prefix(self):
        redis_keys = [
            "uniagent:state:agent:001",
            "uniagent:state:agent:002",
        ]
        mc = self._make_scan_client(redis_keys)
        backend, _ = _make_backend(mock_client=mc)
        keys = _run(backend.list_keys())
        self.assertIn("agent:001", keys)
        self.assertIn("agent:002", keys)

    def test_list_keys_uses_correct_pattern(self):
        mc = self._make_scan_client([])
        backend, _ = _make_backend(key_prefix="ns", mock_client=mc)
        _run(backend.list_keys("agent:"))
        mc.scan_iter.assert_called_once_with(match="ns:agent:*", count=100)

    def test_list_keys_empty_prefix_matches_all(self):
        redis_keys = ["uniagent:state:a", "uniagent:state:b"]
        mc = self._make_scan_client(redis_keys)
        backend, _ = _make_backend(mock_client=mc)
        keys = _run(backend.list_keys())
        self.assertEqual(sorted(keys), ["a", "b"])

    def test_list_keys_returns_empty_list_when_none(self):
        mc = self._make_scan_client([])
        backend, _ = _make_backend(mock_client=mc)
        keys = _run(backend.list_keys("nonexistent:"))
        self.assertEqual(keys, [])

    def test_list_keys_propagates_error(self):
        from redis.exceptions import ConnectionError as RedisConnErr

        async def _fail():
            raise RedisConnErr("refused")
            yield  # make it a generator

        mc = MagicMock()
        mc.scan_iter = MagicMock(return_value=_fail())
        backend, _ = _make_backend(mock_client=mc)
        with self.assertRaises(RedisConnErr):
            _run(backend.list_keys())


# ══════════════════════════════════════════════════════════════════════════ #
#  T6  get_backend 工厂                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class TestGetBackend(unittest.TestCase):

    def test_local_backend_returns_local_file_backend(self):
        import tempfile, os
        from uniagent.state.backend import get_backend
        from uniagent.config.sub_configs import StateConfig
        with tempfile.TemporaryDirectory() as d:
            cfg = StateConfig(backend="local", state_dir=d)
            backend = get_backend(cfg)
            from uniagent.state.backend import LocalFileBackend
            self.assertIsInstance(backend, LocalFileBackend)

    def test_redis_backend_returns_redis_backend(self):
        from uniagent.state.backend import get_backend
        from uniagent.state.redis_backend import RedisBackend
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig(backend="redis",
                          redis_url="redis://localhost:6379/0",
                          key_prefix="test:ns", ttl=60)
        with patch("redis.asyncio.from_url", return_value=MagicMock()):
            backend = get_backend(cfg)
        self.assertIsInstance(backend, RedisBackend)
        self.assertEqual(backend._prefix, "test:ns")
        self.assertEqual(backend._ttl, 60)

    def test_redis_backend_config_applied(self):
        from uniagent.state.backend import get_backend
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig(backend="redis", key_prefix="custom:prefix", ttl=7200)
        with patch("redis.asyncio.from_url", return_value=MagicMock()):
            backend = get_backend(cfg)
        self.assertEqual(backend._prefix, "custom:prefix")
        self.assertEqual(backend._ttl, 7200)


# ══════════════════════════════════════════════════════════════════════════ #
#  T7  StateConfig 新字段                                                    #
# ══════════════════════════════════════════════════════════════════════════ #

class TestStateConfig(unittest.TestCase):

    def test_default_backend_is_local(self):
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig()
        self.assertEqual(cfg.backend, "local")

    def test_default_redis_url(self):
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig()
        self.assertEqual(cfg.redis_url, "redis://localhost:6379/0")

    def test_default_key_prefix(self):
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig()
        self.assertEqual(cfg.key_prefix, "uniagent:state")

    def test_default_ttl_is_zero(self):
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig()
        self.assertEqual(cfg.ttl, 0)

    def test_ttl_cannot_be_negative(self):
        from uniagent.config.sub_configs import StateConfig
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            StateConfig(ttl=-1)

    def test_redis_config_fields_settable(self):
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig(
            backend="redis",
            redis_url="redis://myhost:6380/3",
            key_prefix="myapp",
            ttl=1800,
        )
        self.assertEqual(cfg.redis_url, "redis://myhost:6380/3")
        self.assertEqual(cfg.key_prefix, "myapp")
        self.assertEqual(cfg.ttl, 1800)

    def test_local_state_dir_settable(self):
        from uniagent.config.sub_configs import StateConfig
        cfg = StateConfig(backend="local", state_dir="/data/state")
        self.assertEqual(cfg.state_dir, "/data/state")


if __name__ == "__main__":
    unittest.main(verbosity=2)
