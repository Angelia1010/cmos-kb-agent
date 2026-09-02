"""独立 Processing 服务的离线契约、隔离与 HTTP 测试。"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services"))

from kbagent.processing.agent import KnowledgeProcessingOrchestrator  # noqa: E402
from kbagent.scripted_model import ScriptedChatModel  # noqa: E402
from kbagent.shared.models import Chunk  # noqa: E402
from kbagent.shared.workspace import RunWorkspace, get_workspace, workspace_scope  # noqa: E402
from processing_service.models import ProcessingRequest  # noqa: E402
from processing_service.runner import run_processing_request  # noqa: E402

try:  # 本地未安装服务依赖时，核心服务测试仍可执行。
    from fastapi.testclient import TestClient
    from processing_service.app import create_app
except ModuleNotFoundError:
    TestClient = None
    create_app = None


SENSITIVE_MARKER = "SYNTHETIC_SECRET_DO_NOT_EXPOSE"


def _candidate(prefix: str, index: int, topic: str) -> dict:
    return {
        "knowledge_id": f"{prefix}-{index:03d}",
        "knowledge_name": f"{topic}知识{index}",
        "content": "",
        "retrieval_rank": index,
        "retrieval_score": round(1 - index / 100, 2),
        "matched_atom_ids": [f"{prefix}-A-{index:03d}"],
        "source_routes": ["synthetic"],
        "applicability": {"status": "1"},
        "atoms": [{
            "atom_id": f"{prefix}-A-{index:03d}",
            "group_id": "G001",
            "param_name": "业务内容",
            "param_type": "text",
            "content": f"{topic}办理说明第{index}条",
            "except_rules": [],
            "annotation": None,
            "arrange_seq_number": index,
            "wkuntt": None,
            "applicability": {"status": "1"},
            "metadata": {"private_atom_marker": SENSITIVE_MARKER},
            "raw": {"private_atom_raw": SENSITIVE_MARKER},
        }],
        "metadata": {"private_candidate_marker": SENSITIVE_MARKER},
        "raw": {"private_candidate_raw": SENSITIVE_MARKER},
    }


def _request(prefix: str = "A", topic: str = "流量") -> ProcessingRequest:
    return ProcessingRequest.model_validate({
        "query": f"{topic}查询",
        "retrieval_query": f"查询{topic}办理说明",
        "processing_context": {
            "region_id": "200",
            "region_name": "广东",
            "channel_code": "1",
            "request_time": "2026-09-02T10:00:00+08:00",
            "audience": "agent",
            "customer_type": "个人客户",
        },
        "candidates": [_candidate(prefix, index, topic) for index in range(1, 5)],
    })


class _BrokenScriptedModel(ScriptedChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError(SENSITIVE_MARKER)


class TestProcessingServiceCore(unittest.IsolatedAsyncioTestCase):
    async def test_request_validation_and_empty_context_defaults(self):
        with self.assertRaises(ValidationError):
            ProcessingRequest.model_validate({
                "query": "",
                "retrieval_query": "有效检索问题",
                "processing_context": {},
                "candidates": [],
            })

        request = ProcessingRequest.model_validate({
            "query": "流量查询",
            "retrieval_query": "查询流量",
            "processing_context": {},
            "candidates": [],
        })
        result = await run_processing_request(
            request,
            model=ScriptedChatModel(),
            request_id="request-default-context",
        )
        self.assertEqual("no_valid_candidates", result.outcome)

    async def test_standard_request_and_response_whitelist(self):
        request = _request()
        original = copy.deepcopy(request.model_dump())

        result = await run_processing_request(
            request,
            model=ScriptedChatModel(),
            request_id="request-standard",
        )

        self.assertEqual("request-standard", result.request_id)
        self.assertEqual("scripted", result.model_mode)
        self.assertEqual(3, len(result.top3_candidates))
        self.assertEqual(3, len(result.processed_chunks))
        self.assertEqual([1, 2, 3], [item.rerank_rank for item in result.top3_candidates])
        self.assertTrue(all(item.content_md for item in result.top3_candidates))
        self.assertEqual(
            [item.knowledge_id for item in result.top3_candidates],
            [item.chunk_id for item in result.processed_chunks],
        )
        self.assertEqual(
            [item.content_md for item in result.top3_candidates],
            [item.content for item in result.processed_chunks],
        )
        self.assertTrue(all(
            isinstance(Chunk(**item.model_dump()), Chunk) for item in result.processed_chunks
        ))
        self.assertEqual(original, request.model_dump())
        serialized = result.model_dump_json()
        self.assertNotIn("raw", serialized)
        self.assertNotIn("metadata", serialized)
        self.assertNotIn(SENSITIVE_MARKER, serialized)

    async def test_empty_candidates_are_normal_degradation(self):
        request = _request().model_copy(update={"candidates": []})
        result = await run_processing_request(
            request,
            model=ScriptedChatModel(),
            request_id="request-empty",
        )

        self.assertEqual("no_valid_candidates", result.outcome)
        self.assertTrue(result.degraded)
        self.assertEqual([], result.top3_candidates)
        self.assertEqual([], result.processed_chunks)
        self.assertIn("insufficient_candidates", result.processing_meta.degradation_reasons)

    async def test_model_failure_uses_existing_rerank_fallback_safely(self):
        result = await run_processing_request(
            _request(),
            model=_BrokenScriptedModel(),
            request_id="request-fallback",
        )

        self.assertEqual("degraded", result.outcome)
        self.assertTrue(result.degraded)
        self.assertEqual(3, len(result.top3_candidates))
        self.assertEqual(3, len(result.processed_chunks))
        self.assertTrue(any(item.code == "rerank_model_error" for item in result.warnings))
        self.assertNotIn(SENSITIVE_MARKER, result.model_dump_json())

    async def test_concurrent_requests_do_not_share_workspace(self):
        first, second = await asyncio.gather(
            run_processing_request(
                _request("FLOW", "流量"),
                model=ScriptedChatModel(),
                request_id="request-flow",
            ),
            run_processing_request(
                _request("BROADBAND", "宽带"),
                model=ScriptedChatModel(),
                request_id="request-broadband",
            ),
        )

        self.assertNotEqual(first.trace_id, second.trace_id)
        self.assertTrue(all(item.knowledge_id.startswith("FLOW-") for item in first.top3_candidates))
        self.assertTrue(all(item.knowledge_id.startswith("BROADBAND-") for item in second.top3_candidates))
        self.assertTrue(all(item.chunk_id.startswith("FLOW-") for item in first.processed_chunks))
        self.assertTrue(all(item.chunk_id.startswith("BROADBAND-") for item in second.processed_chunks))
        with self.assertRaises(RuntimeError):
            get_workspace()

    async def test_workspace_is_restored_when_orchestrator_raises(self):
        with patch(
            "processing_service.runner.KnowledgeProcessingOrchestrator.run",
            new=AsyncMock(side_effect=RuntimeError("synthetic runner failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic runner failure"):
                await run_processing_request(
                    _request(),
                    model=ScriptedChatModel(),
                    request_id="request-error",
                )
        with self.assertRaises(RuntimeError):
            get_workspace()

    async def test_missing_workspace_processed_chunks_is_not_hidden_as_empty(self):
        with patch(
            "processing_service.runner.KnowledgeProcessingOrchestrator.run",
            new=AsyncMock(return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "processed_chunks"):
                await run_processing_request(
                    _request(),
                    model=ScriptedChatModel(),
                    request_id="request-missing-answer-chunks",
                )

    async def test_service_result_matches_direct_orchestrator(self):
        request = _request()
        service_result = await run_processing_request(
            request,
            model=ScriptedChatModel(),
            request_id="request-service",
        )

        ws = RunWorkspace(
            query=request.query,
            data={
                "retrieval_query": request.retrieval_query,
                "processing_context": request.processing_context.model_dump(),
                "knowledge_candidates": copy.deepcopy(request.candidates),
            },
        )
        with workspace_scope(ws):
            direct = await KnowledgeProcessingOrchestrator(ScriptedChatModel()).run()
            direct_chunks = [item.to_dict() for item in ws.data["processed_chunks"]]

        self.assertEqual(
            [item.knowledge_id for item in direct],
            [item.knowledge_id for item in service_result.top3_candidates],
        )
        self.assertEqual(
            [item.content_md for item in direct],
            [item.content_md for item in service_result.top3_candidates],
        )
        self.assertEqual(
            [item.rerank_rank for item in direct],
            [item.rerank_rank for item in service_result.top3_candidates],
        )
        self.assertEqual(
            direct_chunks,
            [item.model_dump() for item in service_result.processed_chunks],
        )

    async def test_response_serializes_the_same_workspace_processed_chunks(self):
        captured_chunks = []
        original_run = KnowledgeProcessingOrchestrator.run

        async def capturing_run(orchestrator):
            result = await original_run(orchestrator)
            captured_chunks.extend(copy.deepcopy(get_workspace().data["processed_chunks"]))
            return result

        with patch(
            "processing_service.runner.KnowledgeProcessingOrchestrator.run",
            new=capturing_run,
        ):
            service_result = await run_processing_request(
                _request(),
                model=ScriptedChatModel(),
                request_id="request-same-workspace-chunks",
            )

        self.assertEqual(
            [item.to_dict() for item in captured_chunks],
            [item.model_dump() for item in service_result.processed_chunks],
        )


@unittest.skipUnless(TestClient is not None, "缺少 fastapi/TestClient 服务依赖")
class TestProcessingServiceHttp(unittest.TestCase):
    def test_health_and_standard_post(self):
        with TestClient(create_app(base_path="/api/processing-service/test")) as client:
            health = client.get("/health")
            response = client.post(
                "/api/processing-service/test/process",
                headers={"X-Request-ID": "postman-smoke"},
                json=_request().model_dump(),
            )

        self.assertEqual(200, health.status_code)
        self.assertEqual({"status": "ok", "model_mode": "scripted"}, health.json())
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("0", body["rtnCode"])
        self.assertEqual("postman-smoke", body["object"]["request_id"])
        self.assertEqual("scripted", body["object"]["model_mode"])
        self.assertEqual(3, len(body["object"]["processed_chunks"]))
        self.assertEqual(
            [item["knowledge_id"] for item in body["object"]["top3_candidates"]],
            [item["chunk_id"] for item in body["object"]["processed_chunks"]],
        )
        self.assertTrue(all(
            isinstance(Chunk(**item), Chunk) for item in body["object"]["processed_chunks"]
        ))
        self.assertNotIn(SENSITIVE_MARKER, json.dumps(body, ensure_ascii=False))

    def test_openapi_exposes_typed_processed_chunks(self):
        with TestClient(create_app()) as client:
            schema = client.get("/openapi.json").json()

        response_object = schema["components"]["schemas"]["ProcessingResponseObject"]
        processed_chunks = response_object["properties"]["processed_chunks"]
        self.assertEqual("array", processed_chunks["type"])
        self.assertEqual(
            "#/components/schemas/ProcessedChunk",
            processed_chunks["items"]["$ref"],
        )

    def test_invalid_request_returns_422_without_echoing_body(self):
        with TestClient(create_app()) as client:
            response = client.post(
                "/api/processing-service/prod/process",
                json={"query": SENSITIVE_MARKER},
            )

        self.assertEqual(422, response.status_code)
        serialized = json.dumps(response.json(), ensure_ascii=False)
        self.assertNotIn(SENSITIVE_MARKER, serialized)
        self.assertEqual("40001", response.json()["rtnCode"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
