"""llm_397b 直通服务的离线契约与 HTTP 测试(注入 mock 模型,不依赖灵犀网关)。"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services"))

from llm_service.app import _model_name, _text_of, create_app  # noqa: E402

try:  # 本地未安装服务依赖时，核心服务测试仍可执行。
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    TestClient = None


class _FakeModel:
    """最小 BaseChatModel 替身:invoke 返回可控内容,可注入延迟/异常。"""

    model_name = "fake-397b"

    def __init__(self, content="这是模拟预测文本", delay=0.0, error=None):
        self.content = content
        self.delay = delay
        self.error = error
        self.received = []

    def invoke(self, messages, **kwargs):
        self.received.append(messages)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return AIMessage(
            content=self.content,
            usage_metadata={"input_tokens": 5, "output_tokens": 7,
                             "total_tokens": 12},
        )


@unittest.skipUnless(TestClient is not None, "缺少 fastapi/TestClient 服务依赖")
class TestLlmServiceHttp(unittest.TestCase):

    def test_health(self):
        with TestClient(create_app(model=_FakeModel())) as client:
            resp = client.get("/health")
        self.assertEqual(200, resp.status_code)
        self.assertEqual({"status": "ok"}, resp.json())

    def test_post_returns_predicted_text(self):
        model = _FakeModel(content="5G套餐59元档含20GB流量")
        with TestClient(create_app(model=model)) as client:
            resp = client.post("/llm_397b_api",
                               json={"query": "5G套餐怎么办理?"})
        self.assertEqual(200, resp.status_code)
        body = resp.json()
        self.assertEqual("0", body["rtnCode"])
        self.assertEqual("success", body["rtnMsg"])
        obj = body["object"]
        self.assertEqual("5G套餐59元档含20GB流量", obj["answer"])
        self.assertEqual("fake-397b", obj["model"])
        self.assertTrue(obj["requestId"])
        self.assertGreaterEqual(obj["elapsedMs"], 0)
        # 透传给模型的应是单条 HumanMessage(query 原文)
        self.assertEqual(1, len(model.received))
        messages = model.received[0]
        self.assertEqual(1, len(messages))
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertEqual("5G套餐怎么办理?", messages[0].content)

    def test_request_ids_are_unique(self):
        with TestClient(create_app(model=_FakeModel())) as client:
            first = client.post("/llm_397b_api", json={"query": "问题一"})
            second = client.post("/llm_397b_api", json={"query": "问题二"})
        self.assertNotEqual(first.json()["object"]["requestId"],
                            second.json()["object"]["requestId"])

    def test_missing_query_returns_40001(self):
        with TestClient(create_app(model=_FakeModel())) as client:
            resp = client.post("/llm_397b_api", json={})
        body = resp.json()
        self.assertEqual("40001", body["rtnCode"])
        self.assertEqual({}, body["object"])

    def test_empty_query_returns_40001(self):
        with TestClient(create_app(model=_FakeModel())) as client:
            resp = client.post("/llm_397b_api", json={"query": ""})
        self.assertEqual("40001", resp.json()["rtnCode"])

    def test_gateway_error_returns_50001(self):
        model = _FakeModel(error=ConnectionError("gateway unreachable"))
        with TestClient(create_app(model=model)) as client:
            resp = client.post("/llm_397b_api", json={"query": "你好"})
        body = resp.json()
        self.assertEqual("50001", body["rtnCode"])
        self.assertEqual({}, body["object"])

    def test_timeout_returns_50002(self):
        model = _FakeModel(delay=0.5)
        with TestClient(create_app(model=model, timeout_s=0.05)) as client:
            resp = client.post("/llm_397b_api", json={"query": "你好"})
        self.assertEqual("50002", resp.json()["rtnCode"])


class TestHelpers(unittest.TestCase):

    def test_text_of_str(self):
        self.assertEqual("abc", _text_of("abc"))

    def test_text_of_content_blocks(self):
        content = [{"type": "text", "text": "你好"}, "世界",
                   {"type": "text", "text": "!"}]
        self.assertEqual("你好世界!", _text_of(content))

    def test_text_of_none(self):
        self.assertEqual("", _text_of(None))

    def test_model_name_fallback(self):
        class _OnlyModel:
            model = "m-1"

        class _Nothing:
            pass

        self.assertEqual("fake-397b", _model_name(_FakeModel()))
        self.assertEqual("m-1", _model_name(_OnlyModel()))
        self.assertEqual("", _model_name(_Nothing()))


if __name__ == "__main__":
    unittest.main()
