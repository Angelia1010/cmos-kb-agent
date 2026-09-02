"""Processing 本地 Demo 的入口、输出和故障模式测试。"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import uuid
import unittest
from pathlib import Path

from scripts.run_processing_demo import DemoConfig, main, run_demo
from tests.processing_mock_data import make_top100_candidates


class TestProcessingDemo(unittest.TestCase):
    _TEMP_ROOT = Path(__file__).resolve().parent

    def _test_directory(self) -> Path:
        path = self._TEMP_ROOT / f".processing_demo_test_{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(self._remove_test_directory, path)
        return path

    @staticmethod
    def _remove_test_directory(path: Path) -> None:
        logger = logging.getLogger("processing_demo")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        if path.exists():
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
            path.rmdir()

    def _run(self, config: DemoConfig):
        with contextlib.redirect_stdout(io.StringIO()):
            return asyncio.run(run_demo(config))

    def test_normal_demo_writes_all_snapshots_and_prompt_whitelist(self):
        output_dir = self._test_directory()
        result = self._run(DemoConfig(count=10, output_dir=output_dir))
        self.assertTrue(result.input_unchanged)
        self.assertEqual(len(result.workspace.data["top3_candidates"]), 3)
        self.assertIn("rerank_evidence_map", result.workspace.data)
        expected_files = {
            "01_mock_input.json", "02_normalized_candidates.json",
            "03_filtered_candidates.json", "04_processed_candidates.json",
            "05_sample_content.md", "06_rerank_prompt.json",
            "07_top3_result.json", "processing_demo.log",
        }
        self.assertEqual({path.name for path in output_dir.iterdir()}, expected_files)

        prompt_output = json.loads(
            (output_dir / "06_rerank_prompt.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(prompt_output["calls"]), 2)
        serialized = json.dumps(prompt_output, ensure_ascii=False)
        for forbidden in ('"knowledge_id"', '"evidence_map"', '"raw"', '"metadata"', '"attributes"'):
            self.assertNotIn(forbidden, serialized)
        self.assertIn("100元、30GB、ID、1、E001", serialized)
        first_payload = prompt_output["calls"][0]["payload"]
        self.assertEqual(set(first_payload["candidates"][0]), {
            "evidence_id", "title", "content_md",
        })
        self.assertRegex(first_payload["candidates"][0]["evidence_id"], r"^E\d{3}$")

        top3_output = json.loads(
            (output_dir / "07_top3_result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(top3_output["rerank_metadata"]["mode"], "model")
        self.assertFalse(top3_output["rerank_metadata"]["degraded"])
        self.assertEqual([item["rerank_rank"] for item in top3_output["top3"]], [1, 2, 3])

    def test_failure_simulations_finish_with_top3_and_expected_reasons(self):
        expectations = {
            "timeout": "rerank_timeout",
            "invalid_json": "rerank_invalid_json",
            "insufficient_results": "rerank_wrong_count",
        }
        for simulation, expected_reason in expectations.items():
            with self.subTest(simulation=simulation):
                result = self._run(DemoConfig(count=30, simulate=simulation))
                ws = result.workspace
                self.assertEqual(len(ws.data["top3_candidates"]), 3)
                self.assertTrue(ws.data["processing_meta"].degraded)
                self.assertIn(
                    expected_reason,
                    ws.data["rerank_details"]["fallback_reasons"],
                )
                self.assertEqual([item.rerank_rank for item in ws.data["top3_candidates"]], [1, 2, 3])

    def test_input_json_list_and_candidates_object_are_supported_without_mutation(self):
        candidates = make_top100_candidates(10)
        root = self._test_directory()
        for name, payload in (
            ("list.json", candidates),
            ("object.json", {"candidates": candidates, "source": "retrieval"}),
        ):
            path = root / name
            original = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            path.write_text(original, encoding="utf-8")
            result = self._run(DemoConfig(input_json=path))
            self.assertTrue(result.input_unchanged)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(len(result.workspace.data["top3_candidates"]), 3)

        protected_input = root / "01_mock_input.json"
        protected_content = json.dumps(candidates, ensure_ascii=False) + "\n"
        protected_input.write_text(protected_content, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "覆盖 --input-json"):
            self._run(DemoConfig(count=0, input_json=protected_input, output_dir=root))
        self.assertEqual(protected_input.read_text(encoding="utf-8"), protected_content)

    def test_enveloped_input_writes_query_and_context_without_mutating_file(self):
        root = self._test_directory()
        input_path = root / "retrieval.json"
        payload = {
            "query": "流量查询",
            "retrieval_query": "查询手机剩余流量",
            "processing_context": {
                "region_id": "010",
                "region_name": "北京",
                "channel_code": "APP",
                "request_time": "2026-09-01T09:30:00+08:00",
                "audience": "customer",
                "customer_type": "个人客户",
            },
            "candidates": make_top100_candidates(10),
        }
        original = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        input_path.write_text(original, encoding="utf-8")

        result = self._run(DemoConfig(input_json=input_path))

        self.assertEqual(result.workspace.query, payload["query"])
        self.assertEqual(result.workspace.data["retrieval_query"], payload["retrieval_query"])
        self.assertEqual(
            result.workspace.data["processing_context"], payload["processing_context"]
        )
        self.assertEqual(input_path.read_text(encoding="utf-8"), original)

    def test_cli_values_override_json_and_external_summary_hides_candidate_body(self):
        root = self._test_directory()
        input_path = root / "retrieval.json"
        candidates = make_top100_candidates(10)
        sensitive_marker = "SENSITIVE-CANDIDATE-BODY-DO-NOT-LOG"
        candidates[0]["content"] = sensitive_marker
        payload = {
            "query": "JSON问题",
            "retrieval_query": "JSON检索词",
            "processing_context": {
                "region_id": "JSON-ID",
                "region_name": "JSON地区",
                "channel_code": "JSON渠道",
                "request_time": "2026-01-01T00:00:00+08:00",
                "audience": "json-audience",
                "customer_type": "json-customer",
            },
            "candidates": candidates,
        }
        original = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        input_path.write_text(original, encoding="utf-8")

        argv = [
            "--input-json", str(input_path), "--output-dir", str(root),
            "--query", "CLI问题", "--retrieval-query", "CLI检索词",
            "--region-id", "CLI-ID", "--region-name", "CLI地区",
            "--channel-code", "CLI渠道",
            "--request-time", "2026-09-01T12:00:00+08:00",
            "--audience", "cli-audience", "--customer-type", "cli-customer",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = main(argv)

        self.assertEqual(exit_code, 0)
        summary_text = (root / "07_top3_result.json").read_text(encoding="utf-8")
        summary = json.loads(summary_text)
        self.assertEqual(summary["runtime_input"], {
            "query": "CLI问题",
            "retrieval_query": "CLI检索词",
            "processing_context": {
                "region_id": "CLI-ID",
                "region_name": "CLI地区",
                "channel_code": "CLI渠道",
                "request_time": "2026-09-01T12:00:00+08:00",
                "audience": "cli-audience",
                "customer_type": "cli-customer",
            },
        })
        self.assertFalse(summary["candidate_details_included"])
        self.assertEqual(summary["candidates"], [])
        self.assertNotIn(sensitive_marker, summary_text)
        log_text = (root / "processing_demo.log").read_text(encoding="utf-8")
        self.assertIn('"query": "CLI问题"', log_text)
        self.assertIn('"region_id": "CLI-ID"', log_text)
        self.assertNotIn(sensitive_marker, log_text)
        self.assertEqual(input_path.read_text(encoding="utf-8"), original)

    def test_mock_count_is_deterministic_and_configurable(self):
        first = make_top100_candidates(12)
        second = make_top100_candidates(12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(first[0]["demoScenario"], "region_except")
        self.assertEqual(first[9]["demoScenario"], "empty_rendered_content")
        small = self._run(DemoConfig(count=1))
        self.assertEqual(len(small.workspace.data["top3_candidates"]), 1)
        self.assertEqual(small.prompt_calls, [])


if __name__ == "__main__":
    unittest.main()
