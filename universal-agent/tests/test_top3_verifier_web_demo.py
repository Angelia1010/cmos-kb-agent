"""Top3 Verifier 可视化 Demo 的场景、Runner、HTTP 与前端测试。"""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services"))

from fastapi.testclient import TestClient  # noqa: E402

from processing_service.demo_app import create_demo_app  # noqa: E402
from processing_service.verifier_demo_data import (  # noqa: E402
    get_verifier_scenario,
    list_verifier_scenarios,
)
from processing_service.verifier_demo_models import (  # noqa: E402
    VerifierDemoExecutionRequest,
)
from processing_service.verifier_demo_runner import (  # noqa: E402
    run_verifier_demo_execution,
)
import processing_service.verifier_demo_runner as verifier_runner_module  # noqa: E402


FIXTURE_PATH = PROJECT_ROOT / "demo-inputs" / "top3_verifier_fixture_library.json"
SCENARIO_PATH = PROJECT_ROOT / "demo-inputs" / "top3_verifier_scenarios.json"
STATIC_ROOT = PROJECT_ROOT / "services" / "processing_service" / "static" / "verifier_demo"


def _execution(scenario_id: str) -> VerifierDemoExecutionRequest:
    return VerifierDemoExecutionRequest.model_validate(
        get_verifier_scenario(scenario_id)["execution"]
    )


class TestVerifierWebDemoData(unittest.TestCase):
    def test_fixture_and_scenario_libraries_are_complete_and_synthetic(self):
        fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["fixtures"]
        scenarios = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))["scenarios"]
        summaries = list_verifier_scenarios()

        self.assertEqual(13, len(fixtures))
        self.assertEqual(55, len(scenarios))
        self.assertEqual(55, len(summaries))
        self.assertEqual(55, len({item.id for item in summaries}))
        self.assertEqual(
            {"正常通过", "业务失败", "技术异常", "输出契约", "Evidence 映射", "输入边界", "Scripted Model"},
            {item.group for item in summaries},
        )
        combined = FIXTURE_PATH.read_text(encoding="utf-8") + SCENARIO_PATH.read_text(encoding="utf-8")
        self.assertNotIn("http://", combined)
        self.assertNotIn("https://", combined)

    def test_every_scenario_materializes_to_strict_request(self):
        for summary in list_verifier_scenarios():
            with self.subTest(scenario=summary.id):
                execution = _execution(summary.id)
                self.assertEqual(summary.id, execution.scenario_id)
                self.assertEqual(summary.candidate_count, len(execution.candidates))

    def test_runner_has_no_processing_retrieval_or_network_dependency(self):
        source = inspect.getsource(verifier_runner_module)

        self.assertNotIn("KnowledgeProcessingOrchestrator", source)
        self.assertNotIn("kbagent.retrieval", source)
        self.assertNotIn("httpx", source)
        self.assertNotIn("requests.", source)
        self.assertIn("Top3AnswerabilityVerifier", source)

    def test_frontend_is_safe_and_exposes_all_diagnostics(self):
        index = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        stylesheet = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn("Verifier 可视化验证台", index)
        self.assertIn("检索反馈", index)
        self.assertIn("Evidence 映射", index)
        self.assertIn("模型协议", index)
        self.assertIn("完整诊断", index)
        self.assertIn("/verifier-assets/app.js", index)
        self.assertNotIn("innerHTML", script)
        self.assertIn("textContent", script)
        self.assertIn("renderEvidence", script)
        self.assertIn("renderProtocol", script)
        self.assertIn("fixed_response", script)
        self.assertIn(".status-hero.passed", stylesheet)
        self.assertIn("@media", stylesheet)


class TestVerifierWebDemoExecution(unittest.IsolatedAsyncioTestCase):
    async def test_all_55_scenarios_satisfy_their_assertions(self):
        failures: list[str] = []
        for summary in list_verifier_scenarios():
            result = await run_verifier_demo_execution(_execution(summary.id))
            if not result.all_checks_passed:
                failed = [item.name for item in result.checks if not item.passed]
                failures.append(f"{summary.id}: {failed}; result={result.result}")
        self.assertEqual([], failures)

    async def test_empty_candidates_bypass_model_with_usable_feedback(self):
        result = await run_verifier_demo_execution(_execution("B01-failed-empty-top3"))

        self.assertTrue(result.ok)
        self.assertFalse(result.trace["model_called"])
        self.assertEqual("failed", result.result["status"])
        self.assertEqual(["no_valid_candidates"], result.result["reason_codes"])
        feedback = result.result["retrieval_feedback"]
        self.assertTrue(feedback["suggested_query"])
        self.assertTrue(feedback["missing_aspects"])
        self.assertTrue(feedback["suggested_keywords"])

    async def test_prompt_hides_real_ids_and_sensitive_metadata(self):
        mapping = await run_verifier_demo_execution(_execution("E01-evidence-real-mapping"))
        sensitive = await run_verifier_demo_execution(_execution("E07-sensitive-metadata"))

        mapping_prompt = mapping.model_protocol["user_prompt"]
        sensitive_prompt = sensitive.model_protocol["user_prompt"]
        self.assertNotIn("chunk-plan-fee", mapping_prompt)
        self.assertNotIn("DEMO-SECRET-METADATA-MARKER", sensitive_prompt)
        self.assertNotIn("DEMO-SECRET-RAW-MARKER", sensitive_prompt)
        self.assertEqual(
            ["E001", "E002"],
            [item["evidence_id"] for item in mapping.model_protocol["evidence_candidates"]],
        )

    async def test_request_error_is_not_disguised_as_unknown(self):
        result = await run_verifier_demo_execution(_execution("E09-empty-query"))

        self.assertFalse(result.ok)
        self.assertIsNone(result.result)
        self.assertEqual("ValueError", result.error["type"])
        self.assertFalse(result.trace["model_called"])
        self.assertTrue(result.all_checks_passed)


class TestVerifierWebDemoHttp(unittest.TestCase):
    def setUp(self):
        self.client_context = TestClient(create_demo_app())
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_page_assets_and_scenario_routes(self):
        page = self.client.get("/verifier")
        self.assertEqual(200, page.status_code)
        self.assertIn("Verifier 可视化验证台", page.text)
        self.assertEqual(200, self.client.get("/verifier-assets/app.js").status_code)

        scenarios = self.client.get("/api/verifier/scenarios")
        self.assertEqual(200, scenarios.status_code)
        self.assertEqual(55, len(scenarios.json()))

        scenario = self.client.get("/api/verifier/scenarios/A01-passed-single")
        self.assertEqual(200, scenario.status_code)
        self.assertEqual("A01-passed-single", scenario.json()["id"])
        self.assertEqual(404, self.client.get("/api/verifier/scenarios/not-found").status_code)

    def test_run_and_request_validation_routes(self):
        execution = get_verifier_scenario("A01-passed-single")["execution"]
        response = self.client.post("/api/verifier/run", json=execution)

        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json()["ok"])
        self.assertTrue(response.json()["all_checks_passed"])
        self.assertEqual("passed", response.json()["result"]["status"])

        invalid = self.client.post("/api/verifier/run", json={"query": "x", "unknown": 1})
        self.assertEqual(422, invalid.status_code)

    def test_existing_processing_routes_still_work(self):
        self.assertEqual(200, self.client.get("/").status_code)
        self.assertEqual(200, self.client.get("/api/scenarios").status_code)
        self.assertEqual(200, self.client.get("/assets/app.js").status_code)


if __name__ == "__main__":
    unittest.main()

