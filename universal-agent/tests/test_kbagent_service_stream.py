# -*- coding: utf-8 -*-
"""kbagent_service 流式接口(/retrieve/stream)离线契约测试。

覆盖:
  - SSE 帧格式:``data: {json}\\n\\n``,trace 帧逐条推送 + 唯一 final 结束帧
  - trace 帧与 final.object.processTrace 同源同序
  - final.response 业务字段与同步 /retrieve 一致(ScriptedChatModel 确定性)
  - 参数校验(无用户消息 / appId 白名单)返回 JSON 错误信封而非 SSE
  - KB_SERVICE_EXPOSE_TRACE=0 时不推 trace 帧、processTrace 为空

运行(离线,ScriptedChatModel + MockESClient):
  PYTHONPATH="src;services" python -m unittest tests.test_kbagent_service_stream -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services"))

from kbagent import MockESClient, ScriptedChatModel  # noqa: E402

try:  # 本地未安装服务依赖时,核心测试仍可执行。
    from fastapi.testclient import TestClient
    from kbagent_service.app import create_app
except ModuleNotFoundError:
    TestClient = None
    create_app = None

BASE = "/api/kb-agent-service/test"
QUERY = "用户想办理流量套餐,如何推荐?"


def _body(query: str = QUERY, app_id: str = "kb-test",
          role: int = 1) -> dict:
    return {"params": {
        "appId": app_id,
        "requestId": "req-stream-test",
        "sessionId": "sess-stream-test",
        "userInfo": {"phone": "test-phone", "province": "000",
                     "location": None},
        "extInfo": {},
        "conversations": [{"role": role, "content": query}],
    }}


def _parse_sse(text: str) -> list:
    """按 SSE 空行分帧解析 data: 行。"""
    frames = []
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                frames.append(json.loads(line[5:].strip()))
    return frames


@unittest.skipUnless(TestClient is not None, "缺少 fastapi/TestClient 服务依赖")
class TestRetrieveStream(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(create_app(
            model=ScriptedChatModel(), es=MockESClient(), base_path=BASE))
        self.client.__enter__()          # 触发 lifespan,注入 app.state

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def _stream(self, body: dict):
        with self.client.stream("POST", f"{BASE}/retrieve/stream",
                                json=body) as resp:
            return (resp.status_code,
                    resp.headers.get("content-type", ""),
                    _parse_sse("".join(resp.iter_text())))

    # ── SSE 契约 ────────────────────────────────────────────────────────
    def test_stream_frames_contract(self):
        status, ctype, frames = self._stream(_body())
        self.assertEqual(200, status)
        self.assertIn("text/event-stream", ctype)

        traces = [f for f in frames if f["type"] == "trace"]
        finals = [f for f in frames if f["type"] == "final"]
        self.assertGreaterEqual(len(traces), 3, "正常链路应推送多条 trace 帧")
        self.assertEqual(1, len(finals), "有且仅有一个 final 帧")
        self.assertEqual("final", frames[-1]["type"], "final 必须是最后一帧")

        for t in traces:
            self.assertIn("ts_ms", t)
            self.assertIn("stage", t)
            self.assertIn("event", t)
        stages = [t["stage"] for t in traces]
        self.assertEqual("run", stages[0], "首帧应为 run.start")
        self.assertTrue(any(s.startswith("retrieval") for s in stages))
        self.assertIn("finalize", stages)

        resp = finals[0]["response"]
        self.assertEqual("0", resp["rtnCode"])
        obj = resp["object"]
        # trace 帧与最终信封 processTrace 同源同序
        self.assertEqual(len(traces), len(obj["processTrace"]))
        self.assertEqual(stages, [e["stage"] for e in obj["processTrace"]])
        # sources 修复回归:正常链路必须带出全部素材
        self.assertGreater(len(obj["sources"]), 0)
        self.assertTrue(all(s["content"] for s in obj["sources"]),
                        "sources.content 不应为空")
        self.assertEqual(100, max(s["relevance"] for s in obj["sources"]),
                         "最相关一篇的 relevance 应归一化为 100")

    # ── 与同步 /retrieve 的一致性 ───────────────────────────────────────
    def test_stream_final_matches_plain_retrieve(self):
        plain = self.client.post(f"{BASE}/retrieve", json=_body()).json()
        _, _, frames = self._stream(_body())
        stream = [f for f in frames if f["type"] == "final"][0]["response"]

        self.assertEqual(plain["rtnCode"], stream["rtnCode"])
        po, so = plain["object"], stream["object"]
        for field in ("degraded", "script", "handlingSuggestion"):
            self.assertEqual(po[field], so[field], f"{field} 应一致")
        self.assertEqual([s["chunkId"] for s in po["sources"]],
                         [s["chunkId"] for s in so["sources"]])
        self.assertEqual(po["usability"]["level"], so["usability"]["level"])

    # ── 参数校验:JSON 错误信封而非 SSE ─────────────────────────────────
    def test_no_user_message_returns_json_error(self):
        resp = self.client.post(f"{BASE}/retrieve/stream",
                                json=_body(role=2))
        self.assertIn("application/json", resp.headers.get("content-type", ""))
        self.assertEqual("40001", resp.json()["rtnCode"])

    def test_appid_whitelist_rejected(self):
        with patch.dict(os.environ, {"KB_SERVICE_APP_IDS": "other-app"}):
            resp = self.client.post(f"{BASE}/retrieve/stream", json=_body())
        self.assertIn("application/json", resp.headers.get("content-type", ""))
        self.assertEqual("40001", resp.json()["rtnCode"])

    # ── trace 开关 ─────────────────────────────────────────────────────
    def test_expose_trace_off_only_final_frame(self):
        with patch.dict(os.environ, {"KB_SERVICE_EXPOSE_TRACE": "0"}):
            status, ctype, frames = self._stream(_body())
        self.assertEqual(200, status)
        self.assertIn("text/event-stream", ctype)
        self.assertEqual([], [f for f in frames if f["type"] == "trace"])
        self.assertEqual(1, len(frames))
        resp = frames[0]["response"]
        self.assertEqual("0", resp["rtnCode"])
        self.assertEqual([], resp["object"]["processTrace"])


@unittest.skipUnless(TestClient is not None, "缺少 fastapi/TestClient 服务依赖")
class TestEnvOverrides(unittest.TestCase):
    """50002 超时治理的环境变量覆盖:TIMEOUT_S / MAX_RETRIEVAL_ROUNDS。"""

    def test_timeout_env_override_and_explicit_precedence(self):
        from kbagent_service.app import DEFAULT_TIMEOUT_S, _resolve_timeout_s
        with patch.dict(os.environ, {"KB_SERVICE_TIMEOUT_S": "300"}):
            self.assertEqual(300.0, _resolve_timeout_s())
            self.assertEqual(45.0, _resolve_timeout_s(45.0))  # 显式入参优先
        with patch.dict(os.environ, {"KB_SERVICE_TIMEOUT_S": "not-a-number"}):
            self.assertEqual(DEFAULT_TIMEOUT_S, _resolve_timeout_s())
        os.environ.pop("KB_SERVICE_TIMEOUT_S", None)
        self.assertEqual(DEFAULT_TIMEOUT_S, _resolve_timeout_s())

    def test_max_rounds_env_override(self):
        from kbagent.shared.config import DEFAULT_CONFIG
        from kbagent_service.app import _resolve_cfg
        with patch.dict(os.environ, {"KB_SERVICE_MAX_RETRIEVAL_ROUNDS": "2"}):
            self.assertEqual(2, _resolve_cfg().max_retrieval_rounds)
        os.environ.pop("KB_SERVICE_MAX_RETRIEVAL_ROUNDS", None)
        self.assertEqual(DEFAULT_CONFIG.max_retrieval_rounds,
                         _resolve_cfg().max_retrieval_rounds)

    def test_app_state_picks_up_env(self):
        with patch.dict(os.environ, {"KB_SERVICE_TIMEOUT_S": "123",
                                     "KB_SERVICE_MAX_RETRIEVAL_ROUNDS": "1"}):
            client = TestClient(create_app(
                model=ScriptedChatModel(), es=MockESClient(), base_path=BASE))
            with client:      # 触发 lifespan
                self.assertEqual(123.0, client.app.state.timeout_s)
                self.assertEqual(1, client.app.state.cfg.max_retrieval_rounds)


if __name__ == "__main__":
    unittest.main(verbosity=2)
