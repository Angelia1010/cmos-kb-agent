"""Processing 可视化 Demo 的离线数据、执行器、HTTP 与前端安全测试。"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services"))

from fastapi.testclient import TestClient  # noqa: E402

from kbagent.scripted_model import ScriptedChatModel  # noqa: E402
try:  # demo 系列模块尚未随 11d6773 提交,缺失时整模块跳过而非报错
    from processing_service.demo_app import create_demo_app  # noqa: E402
    from processing_service.demo_data import get_scenario, list_scenarios  # noqa: E402
    from processing_service.demo_models import DemoExecutionRequest  # noqa: E402
    from processing_service.demo_runner import run_demo_execution  # noqa: E402
    import processing_service.demo_runner as demo_runner_module  # noqa: E402
except ModuleNotFoundError as _exc:
    raise unittest.SkipTest(
        f"processing_service demo 模块未提交,等待 processing-dev 补交: {_exc}")
from processing_service.models import ProcessingRequest  # noqa: E402
from processing_service.runner import run_processing_request  # noqa: E402


FIXTURE_PATH = PROJECT_ROOT / "demo-inputs" / "processing_web_fixture_library.json"
SCENARIO_PATH = PROJECT_ROOT / "demo-inputs" / "processing_web_scenarios.json"
STATIC_ROOT = PROJECT_ROOT / "services" / "processing_service" / "static" / "processing_demo"


def _execution(scenario_id: str) -> DemoExecutionRequest:
    return DemoExecutionRequest.model_validate(get_scenario(scenario_id)["execution"])


class TestProcessingWebDemoData(unittest.TestCase):
    def test_fixture_library_is_large_complex_and_explicitly_synthetic(self):
        payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        fixtures = payload["fixtures"]
        atoms = [
            atom
            for fixture in fixtures
            for atom in fixture["chunk"]["extra"]["atoms"]
        ]

        self.assertEqual(12, len(fixtures))
        self.assertEqual(63, len(atoms))
        self.assertTrue(all(row["fixture_id"].startswith(("CONTROL-", "TARGET-")) for row in fixtures))
        self.assertTrue(all(row["chunk"]["doc_id"].startswith("DEMO-") for row in fixtures))
        self.assertTrue(any("except" in atom for atom in atoms))
        self.assertTrue(any("annotation" in atom for atom in atoms))
        self.assertTrue(any("applicability" in atom for atom in atoms))
        fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("http://", fixture_text)
        self.assertEqual(2, fixture_text.count("https://example.invalid/"))
        self.assertEqual(2, fixture_text.count("https://"))

    def test_exactly_23_scenarios_materialize_to_processing_requests(self):
        raw = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))["scenarios"]
        summaries = list_scenarios()

        self.assertEqual(23, len(raw))
        self.assertEqual(23, len(summaries))
        self.assertEqual(23, len({item.id for item in summaries}))
        for item in summaries:
            materialized = get_scenario(item.id)
            request = ProcessingRequest.model_validate(materialized["execution"]["request"])
            self.assertEqual(item.chunk_count, len(request.chunks))
            self.assertGreaterEqual(item.atom_count, 0)

    def test_frontend_is_self_contained_and_uses_safe_text_rendering(self):
        index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn("Processing 可视化验证台", index)
        self.assertIn("/assets/app.js", index)
        self.assertIn("场景预期校验", index)
        self.assertIn('id="diagnosticSection"', index)
        self.assertIn("全部诊断（较长）", index)
        self.assertNotIn("http://", index)
        self.assertNotIn("https://", index)
        self.assertNotIn("innerHTML", script)
        self.assertIn("textContent", script)
        self.assertIn("renderKnowledgeTarget", script)
        self.assertIn("renderAtomComparison", script)
        self.assertIn('row.addEventListener("click"', script)
        self.assertIn("diagnosticPayload", script)
        self.assertIn("checkDetail", script)
        self.assertIn("已在处理后 Markdown 中找到", script)
        self.assertIn("--forest: #174a8b", stylesheet)
        self.assertNotIn("#144f3d", stylesheet)
        self.assertIn("@media", stylesheet)

    def test_demo_runner_has_no_retrieval_or_network_client_dependency(self):
        source = inspect.getsource(demo_runner_module)

        self.assertNotIn("kbagent.retrieval", source)
        self.assertNotIn("ProduceESClient", source)
        self.assertNotIn("httpx", source)
        self.assertNotIn("requests.", source)


class TestProcessingWebDemoExecution(unittest.IsolatedAsyncioTestCase):
    async def test_all_fixed_scenarios_satisfy_their_assertions(self):
        failures: list[str] = []
        for summary in list_scenarios():
            result = await run_demo_execution(_execution(summary.id))
            if not result.all_checks_passed:
                failed = [item.name for item in result.checks if not item.passed]
                failures.append(f"{summary.id}: {failed}")

        self.assertEqual([], failures)

    async def test_demo_matches_the_standard_processing_service_result(self):
        execution = _execution("except-region-override")
        demo = await run_demo_execution(execution, model=ScriptedChatModel())
        standard = await run_processing_request(
            execution.request,
            model=ScriptedChatModel(),
            request_id="web-demo-equivalence",
        )

        self.assertTrue(demo.ok)
        self.assertEqual(
            demo.stages["processed_chunks"],
            [item.model_dump() for item in standard.processed_chunks],
        )
        self.assertEqual(
            [item["knowledge_id"] for item in demo.stages["top3"]],
            [item.knowledge_id for item in standard.top3_candidates],
        )

    async def test_parallel_runs_keep_workspace_results_isolated(self):
        match, mismatch = await asyncio.gather(
            run_demo_execution(_execution("candidate-region-id-match")),
            run_demo_execution(_execution("candidate-region-id-mismatch")),
        )

        self.assertEqual("candidate-region-id-match", match.scenario_id)
        self.assertEqual("candidate-region-id-mismatch", mismatch.scenario_id)
        self.assertTrue(match.all_checks_passed)
        self.assertTrue(mismatch.all_checks_passed)
        match_ids = {item["knowledge_id"] for item in match.stages["filtered"]}
        mismatch_ids = {item["knowledge_id"] for item in mismatch.stages["filtered"]}
        self.assertIn("DEMO-KLG-20260915-004", match_ids)
        self.assertNotIn("DEMO-KLG-20260915-004", mismatch_ids)


class TestProcessingWebDemoHttp(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(create_demo_app())
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health_index_assets_and_scenario_routes(self):
        health = self.client.get("/health")
        self.assertEqual(200, health.status_code)
        self.assertEqual("offline-synthetic", health.json()["mode"])

        index = self.client.get("/")
        self.assertEqual(200, index.status_code)
        self.assertIn("Processing 可视化验证台", index.text)
        self.assertEqual(200, self.client.get("/assets/app.js").status_code)

        scenarios = self.client.get("/api/scenarios")
        self.assertEqual(200, scenarios.status_code)
        self.assertEqual(23, len(scenarios.json()))

        scenario = self.client.get("/api/scenarios/except-region-override")
        self.assertEqual(200, scenario.status_code)
        self.assertEqual("except-region-override", scenario.json()["id"])
        self.assertEqual(404, self.client.get("/api/scenarios/not-found").status_code)

    def test_run_and_validation_routes(self):
        execution = get_scenario("annotation-no-leak")["execution"]
        response = self.client.post("/api/run", json=execution)

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json()["ok"])
        self.assertTrue(response.json()["all_checks_passed"])

        invalid = self.client.post("/api/run", json={"request": {"query": ""}})
        self.assertEqual(422, invalid.status_code)


if __name__ == "__main__":
    unittest.main()
