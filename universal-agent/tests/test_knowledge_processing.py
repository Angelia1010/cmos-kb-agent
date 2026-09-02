"""知识级 Processing 独立链路测试。"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, "src")

from langchain_core.messages import AIMessage

from kbagent.processing.agent import KnowledgeProcessingOrchestrator
from kbagent.processing.rerank import rerank_candidates
from kbagent.processing.tools import PROCESSING_TOOLS, build_knowledge_processing_tools
from kbagent.scripted_model import ScriptedChatModel
from kbagent.shared.knowledge_processing.adapter import (
    normalize_knowledge_candidates,
    normalize_processing_context,
)
from kbagent.shared.knowledge_processing.analysis import analyze_candidates
from kbagent.shared.knowledge_processing.applicability import (
    evaluate_applicability,
    filter_candidates,
)
from kbagent.shared.knowledge_processing.atoms import (
    apply_except_override,
    except_rules_match,
    match_except_conditions,
    parse_except_rule,
    process_atoms,
)
from kbagent.shared.knowledge_processing.eligibility import is_rerank_eligible
from kbagent.shared.knowledge_processing.markdown import (
    build_candidate_markdown,
    build_knowledge_markdown,
)
from kbagent.shared.knowledge_processing.pipeline import process_knowledge_candidates
from kbagent.shared.knowledge_processing.models import (
    KnowledgeProcessingOptions,
    Applicability,
    KnowledgeAtom,
    KnowledgeCandidate,
    ProcessedKnowledge,
    ProcessingContext,
)
from kbagent.shared.knowledge_processing.richtext import (
    is_renderable_content,
    is_supported_content_type,
    render_richtext_with_warnings,
)
from kbagent.shared.workspace import RunWorkspace, set_workspace
from tests.processing_mock_data import make_top100_candidates


class _CapturingModel:
    def __init__(
        self,
        batch_response=None,
        global_response=None,
        raises=False,
        batch_delay=0.0,
        global_delay=0.0,
    ):
        self.batch_response = batch_response
        self.global_response = global_response
        self.raises = raises
        self.batch_delay = batch_delay
        self.global_delay = global_delay
        self.calls = []

    async def ainvoke(self, messages):
        text = "\n".join(str(message.content) for message in messages)
        self.calls.append(text)
        is_global = "[TASK:rerank_global]" in text
        delay = self.global_delay if is_global else self.batch_delay
        if delay:
            await asyncio.sleep(delay)
        if self.raises:
            raise RuntimeError("model unavailable")
        payload_text = text.split("RERANK_INPUT_BEGIN\n")[-1].split("\nRERANK_INPUT_END")[0]
        payload = json.loads(payload_text)
        ids = [item["evidence_id"] for item in payload["candidates"]]
        expected = payload["top_k"]
        configured = self.global_response if is_global else self.batch_response
        if configured is None:
            configured = json.dumps({"ranked_ids": ids[:expected]})
        return AIMessage(content=configured)


def _processed(count=8):
    return [
        ProcessedKnowledge(
            knowledge_id=f"SECRET-{index:03d}",
            name=f"标题 {index}",
            retrieval_rank=index,
            content_md=f"# 标题 {index}\n\n完整内容 {index}",
        )
        for index in range(1, count + 1)
    ]


class TestAdapter(unittest.TestCase):
    def test_candidate_applicability_normalization_is_shared_for_dict_and_object(self):
        raw_candidates = normalize_knowledge_candidates([
            {"knowledge_id": "offline", "knowledge_name": "下架", "content": "A",
             "applicability": {"status": "offline"}},
            {"knowledge_id": "future", "knowledge_name": "未生效", "content": "B",
             "applicability": {"effective_start": "2026-09-01"}},
            {"knowledge_id": "region", "knowledge_name": "地区", "content": "C",
             "applicability": {"region_ids": ["0755"]}},
        ]).candidates
        object_candidates = [
            KnowledgeCandidate(
                knowledge_id="expired", name="过期", content="D", end_at="2026-08-27",
            ),
            KnowledgeCandidate(
                knowledge_id="channel", name="渠道", content="E", channel_codes=["10086"],
            ),
            KnowledgeCandidate(
                knowledge_id="nested-wins", name="嵌套优先", content="F", status="offline",
                applicability=Applicability(status="active", region_ids=["010"]),
            ),
        ]
        normalized_objects = normalize_knowledge_candidates(object_candidates).candidates
        context = ProcessingContext(
            region_id="010", channel_code="web", request_time="2026-08-28T12:00:00+08:00"
        )
        filtered, decisions, _ = filter_candidates(
            [*raw_candidates, *normalized_objects], context
        )
        self.assertEqual([candidate.knowledge_id for candidate in filtered], ["nested-wins"])
        reasons = {decision.knowledge_id: decision.reasons for decision in decisions}
        self.assertIn("inactive_status", reasons["offline"])
        self.assertIn("not_started", reasons["future"])
        self.assertIn("expired", reasons["expired"])
        self.assertIn("region_not_applicable", reasons["region"])
        self.assertIn("channel_not_applicable", reasons["channel"])
        nested = normalized_objects[-1]
        self.assertEqual(nested.status, "active")
        self.assertEqual(nested.region_ids, ["010"])
        self.assertEqual(nested.applicability.region_ids, ["010"])

    def test_unknown_fields_are_traced_but_ignored_and_input_is_unchanged(self):
        raw = [{
            "knowledge_id": "snake-id",
            "knowledge_name": "测试",
            "status": "offline",
            "legacy_candidate_field": {"status": "offline"},
            "atoms": [{
                "param_name": "正文",
                "content": "A",
                "except_rules": {"region": "北京"},
                "channel_codes": ["blocked"],
                "legacy_atom_field": {"channel_codes": ["blocked"]},
            }],
        }]
        original = copy.deepcopy(raw)
        result = normalize_knowledge_candidates(raw)
        self.assertEqual(result.candidates[0].knowledge_id, "snake-id")
        self.assertEqual(result.candidates[0].atoms[0].except_rules, {"region": "北京"})
        self.assertEqual(
            result.candidates[0].metadata["legacy_candidate_field"], {"status": "offline"}
        )
        self.assertEqual(result.candidates[0].metadata["status"], "offline")
        self.assertIsNone(result.candidates[0].applicability.status)
        self.assertEqual(
            result.candidates[0].atoms[0].metadata["legacy_atom_field"],
            {"channel_codes": ["blocked"]},
        )
        self.assertEqual(
            result.candidates[0].atoms[0].metadata["channel_codes"], ["blocked"]
        )
        self.assertEqual(result.candidates[0].atoms[0].applicability.channel_codes, [])
        self.assertNotIn("field_conflict", [warning.code for warning in result.warnings])
        self.assertEqual(raw, original)
        self.assertEqual(result.candidates[0].raw, raw[0])

    def test_removed_legacy_aliases_are_not_interpreted(self):
        result = normalize_knowledge_candidates([{
            "knowledgeId": "legacy-id",
            "knowledgeName": "旧标题",
            "knowledgeAtoms": [{"atomId": "legacy-atom", "content": "旧正文"}],
            "content": "标准正文仍可处理",
        }])
        candidate = result.candidates[0]
        self.assertIsNone(candidate.knowledge_id)
        self.assertEqual(candidate.name, "未命名知识-001")
        self.assertEqual(candidate.atoms, [])
        self.assertEqual(candidate.metadata["knowledgeId"], "legacy-id")
        self.assertEqual(candidate.metadata["knowledgeName"], "旧标题")
        self.assertIn("missing_knowledge_id", [warning.code for warning in result.warnings])

    def test_missing_values_invalid_atoms_and_single_error_isolation(self):
        class Bad:
            @property
            def __dict__(self):
                raise RuntimeError("bad object")

        result = normalize_knowledge_candidates([
            {"content": "A", "atoms": "invalid"}, Bad(),
            {"knowledge_id": "K3", "knowledge_name": "C"},
        ])
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].name, "未命名知识-001")
        self.assertEqual(result.candidates[0].retrieval_rank, 1)
        self.assertEqual(result.candidates[0].atoms, [])
        codes = {warning.code for warning in result.warnings}
        self.assertTrue({"missing_name", "missing_knowledge_id", "invalid_atoms", "candidate_conversion_error"} <= codes)

    def test_collection_and_scalar_type_errors_keep_existing_defaults(self):
        wrapped = normalize_knowledge_candidates({"candidates": [{"knowledge_id": "K1"}]})
        self.assertEqual(wrapped.candidates, [])
        self.assertEqual([warning.code for warning in wrapped.warnings], ["invalid_candidates"])

        raw = [{
            "knowledge_id": "K1",
            "knowledge_name": None,
            "retrieval_rank": "bad-rank",
            "retrieval_score": "bad-score",
            "applicability": ["bad-shape"],
            "atoms": [
                {"atom_id": "A1", "param_name": None, "content": None,
                 "arrange_seq_number": "bad-order"},
                123,
            ],
        }]
        original = copy.deepcopy(raw)
        result = normalize_knowledge_candidates(raw)
        candidate = result.candidates[0]
        self.assertEqual(candidate.name, "未命名知识-001")
        self.assertEqual(candidate.retrieval_rank, 1)
        self.assertIsNone(candidate.retrieval_score)
        self.assertEqual(len(candidate.atoms), 1)
        self.assertEqual(candidate.atoms[0].param_name, "未命名字段")
        self.assertEqual(candidate.atoms[0].content, "")
        self.assertIsNone(candidate.atoms[0].arrange_seq_number)
        self.assertEqual(raw, original)
        codes = {warning.code for warning in result.warnings}
        self.assertTrue({
            "missing_name", "invalid_retrieval_score",
            "invalid_applicability", "missing_param_name", "invalid_arrange_seq_number",
            "atom_conversion_error",
        } <= codes)

    def test_standard_context_ignores_unknown_fields_but_preserves_raw(self):
        raw = {
            "region_id": "200",
            "region_name": "广东",
            "channel_code": "1",
            "request_time": "2026-09-01T14:00:00+08:00",
            "audience": "agent",
            "customer_type": "个人客户",
            "regionId": "legacy-region",
            "arbitrary_condition": "不参与业务判断",
        }
        original = copy.deepcopy(raw)
        context = normalize_processing_context(raw)
        self.assertEqual((context.region_id, context.region_name, context.channel_code), (
            "200", "广东", "1",
        ))
        self.assertEqual(context.customer_type, "个人客户")
        self.assertEqual(context.attributes, {})
        self.assertEqual(context.raw, original)
        self.assertEqual(raw, original)

    def test_candidate_applicability_uses_only_nested_standard_fields(self):
        result = normalize_knowledge_candidates([
            {
                "knowledge_id": "current",
                "knowledge_name": "当前适用性",
                "content": "A",
                "applicability": {
                    "effective_start": "2026-01-01T00:00:00+08:00",
                    "effective_end": "2026-12-31T23:59:59+08:00",
                    "region_ids": ["0755"],
                    "channel_codes": ["10086"],
                },
            },
            {
                "knowledge_id": "nested",
                "knowledge_name": "嵌套适用性",
                "content": "B",
                "applicability": {
                    "effective_start": "2025-01-01",
                    "effective_end": "2025-12-31",
                    "region_ids": ["0755"],
                    "channel_codes": ["10086"],
                },
            },
        ])
        current, nested = result.candidates
        self.assertEqual(current.applicability.effective_start, "2026-01-01T00:00:00+08:00")
        self.assertEqual(current.applicability.effective_end, "2026-12-31T23:59:59+08:00")
        self.assertEqual(current.applicability.region_ids, ["0755"])
        self.assertEqual(current.applicability.channel_codes, ["10086"])
        self.assertEqual(nested.applicability.region_ids, ["0755"])
        self.assertNotIn("effective_start", nested.applicability.conditions)
        filtered, _, _ = filter_candidates(result.candidates, ProcessingContext(
            region_id="0755", channel_code="10086",
            request_time="2026-08-28T10:00:00+08:00",
        ))
        self.assertEqual([candidate.knowledge_id for candidate in filtered], ["current"])

    def test_processing_context_contract_fields_and_zoned_request_time(self):
        context = normalize_processing_context({
            "region_id": "0755",
            "region_name": "深圳",
            "channel_code": "10086",
            "request_time": "2026-08-27T16:30:00Z",
            "audience": "customer",
        })
        self.assertEqual(context.region_id, "0755")
        self.assertEqual(context.region_name, "深圳")
        self.assertEqual(context.region, "深圳")
        self.assertEqual(context.channel_code, "10086")
        self.assertEqual(context.channel, "10086")
        self.assertEqual(context.request_time, "2026-08-27T16:30:00+00:00")
        self.assertEqual(context.as_of, context.request_time)
        self.assertEqual(context.audience, "customer")
        for recognized in ("region_id", "region_name", "channel_code", "request_time", "audience"):
            self.assertNotIn(recognized, context.attributes)
        applicability = Applicability(
            effective_start="2026-08-28T00:00:00+08:00",
            region_ids=["0755"], channel_codes=["10086"],
        )
        self.assertEqual(evaluate_applicability(applicability, context), [])

    def test_standard_candidate_and_atom_fields_map_to_dedicated_model_fields(self):
        result = normalize_knowledge_candidates([{
            "knowledge_id": "K",
            "knowledge_name": "套餐",
            "retrieval_rank": 3,
            "retrieval_score": 0.91,
            "matched_atom_ids": ["A1"],
            "source_routes": ["keyword"],
            "knowledge_type": "faq",
            "template_id": "T1",
            "atoms": [
                {
                    "atom_id": "A2", "param_name": "流量", "param_type": "number",
                    "group_id": "G1", "arrange_seq_number": 2, "content": 30, "wkuntt": "GB",
                },
                {
                    "atom_id": "A1", "param_name": "月费", "param_type": "number",
                    "group_id": "G1", "arrange_seq_number": 1, "content": 0, "wkuntt": "元",
                },
            ],
        }])
        candidate = result.candidates[0]
        self.assertEqual(candidate.retrieval_score, 0.91)
        self.assertEqual(candidate.matched_atom_ids, ["A1"])
        self.assertEqual(candidate.source_routes, ["keyword"])
        self.assertEqual((candidate.knowledge_type, candidate.template_id), ("faq", "T1"))
        first, second = candidate.atoms
        self.assertEqual(
            (first.atom_id, first.param_name, first.param_type, first.group_id,
             first.arrange_seq_number, first.wkuntt),
            ("A2", "流量", "number", "G1", 2, "GB"),
        )
        self.assertEqual(
            (second.atom_id, second.param_name, second.param_type, second.group_id,
             second.arrange_seq_number, second.wkuntt),
            ("A1", "月费", "number", "G1", 1, "元"),
        )
        markdown = build_candidate_markdown(candidate).content_md
        self.assertLess(markdown.index("### 月费"), markdown.index("### 流量"))
        self.assertIn("0 元", markdown)
        self.assertIn("30 GB", markdown)
        for recognized in ("retrieval_score", "matched_atom_ids", "source_routes", "knowledge_type", "template_id"):
            self.assertNotIn(recognized, candidate.metadata)

    def test_object_candidate_atoms_use_the_same_normalization_as_dict_atoms(self):
        object_candidate = KnowledgeCandidate(
            knowledge_id="object", name="对象原子", atoms=[
                KnowledgeAtom(atom_id="A2", title="流量", content=30, group="G", order=2, unit="GB"),
                KnowledgeAtom(atom_id="A1", title="月费", content=10, group="G", order=1, unit="元"),
            ],
        )
        dict_candidate = {
            "knowledge_id": "dict", "knowledge_name": "对象原子", "atoms": [
                {"atom_id": "A2", "param_name": "流量", "content": 30,
                 "group_id": "G", "arrange_seq_number": 2, "wkuntt": "GB"},
                {"atom_id": "A1", "param_name": "月费", "content": 10,
                 "group_id": "G", "arrange_seq_number": 1, "wkuntt": "元"},
            ],
        }
        object_normalized, dict_normalized = normalize_knowledge_candidates([
            object_candidate, dict_candidate,
        ]).candidates
        for candidate in (object_normalized, dict_normalized):
            self.assertEqual([
                (atom.param_name, atom.group_id, atom.arrange_seq_number, atom.wkuntt)
                for atom in candidate.atoms
            ], [("流量", "G", 2, "GB"), ("月费", "G", 1, "元")])
        object_md = build_candidate_markdown(object_normalized).content_md
        dict_md = build_candidate_markdown(dict_normalized).content_md
        self.assertEqual(object_md, dict_md)
        self.assertLess(object_md.index("### 月费"), object_md.index("### 流量"))

    def test_effective_end_date_is_inclusive_but_timestamp_is_strict(self):
        candidates = normalize_knowledge_candidates([
            {
                "knowledge_id": "date", "knowledge_name": "纯日期", "content": "A",
                "applicability": {"effective_end": "2026-08-28"},
            },
            {
                "knowledge_id": "time", "knowledge_name": "精确时间", "content": "B",
                "applicability": {"effective_end": "2026-08-28T12:00:00+08:00"},
            },
        ]).candidates
        date_rule, timed_rule = (candidate.applicability for candidate in candidates)
        self.assertEqual(date_rule.effective_end, "2026-08-28")
        self.assertNotIn("expired", evaluate_applicability(
            date_rule, ProcessingContext(request_time="2026-08-28T23:59:59.999999+08:00")
        ))
        self.assertIn("expired", evaluate_applicability(
            date_rule, ProcessingContext(request_time="2026-08-29T00:00:00+08:00")
        ))
        self.assertNotIn("expired", evaluate_applicability(
            timed_rule, ProcessingContext(request_time="2026-08-28T11:59:59+08:00")
        ))
        self.assertIn("expired", evaluate_applicability(
            timed_rule, ProcessingContext(request_time="2026-08-28T12:00:00.000001+08:00")
        ))


class TestCoreProcessing(unittest.TestCase):
    def test_annotation_visibility_is_enforced_before_markdown_and_rerank(self):
        candidate = normalize_knowledge_candidates([{
            "knowledge_id": "annotation-safe",
            "knowledge_name": "注解安全",
            "atoms": [
                {"atom_id": "public", "param_name": "公开", "content": "正文-public",
                 "annotation": {"visibility": "public", "content": "公开说明"}},
                {"atom_id": "customer", "param_name": "客户", "content": "正文-customer",
                 "annotation": {"visibleTo": ["customer"], "text": "客户说明"}},
                {"atom_id": "agent", "param_name": "坐席", "content": "正文-agent",
                 "annotation": {"visible_to": ["agent"], "content": "坐席秘密"}},
                {"atom_id": "internal", "param_name": "内部", "content": "正文-internal",
                 "annotation": {"visibility": "internal", "content": "内部秘密"}},
                {"atom_id": "private", "param_name": "个人", "content": "正文-private",
                 "annotation": {"personal": True, "content": "个人秘密"}},
                {"atom_id": "hidden", "param_name": "隐藏", "content": "正文-hidden",
                 "annotation": {"visible": False, "content": "不可见秘密"}},
                {"atom_id": "plain", "param_name": "未标记", "content": "正文-plain",
                 "annotation": "未标记秘密"},
                {"atom_id": "except", "param_name": "地区覆盖", "content": "默认正文",
                 "except_rules": json.dumps({
                     "region_id": "0755", "value": "深圳正文",
                     "annotation": {"visibility": "agent", "content": "覆盖秘密"},
                 }, ensure_ascii=False)},
            ],
        }]).candidates[0]

        customer = build_candidate_markdown(
            candidate, ProcessingContext(region_id="0755", audience="customer")
        )
        self.assertIn("公开说明", customer.content_md)
        self.assertIn("客户说明", customer.content_md)
        for secret in ("坐席秘密", "内部秘密", "个人秘密", "不可见秘密", "未标记秘密", "覆盖秘密"):
            self.assertNotIn(secret, customer.content_md)
            self.assertNotIn(secret, json.dumps(customer.to_dict(), ensure_ascii=False))
        for body in (
            "正文-public", "正文-customer", "正文-agent", "正文-internal",
            "正文-private", "正文-hidden", "正文-plain", "深圳正文",
        ):
            self.assertIn(body, customer.content_md)
        self.assertIn("annotation_filtered", [
            warning.code for warning in customer.processing_warnings
        ])
        customer_twice = build_candidate_markdown(
            customer, ProcessingContext(region_id="0755", audience="customer")
        )
        self.assertEqual(customer.content_md, customer_twice.content_md)

        agent = build_candidate_markdown(candidate, ProcessingContext(region_id="0755", audience="agent"))
        self.assertIn("坐席秘密", agent.content_md)
        self.assertIn("内部秘密", agent.content_md)
        self.assertIn("覆盖秘密", agent.content_md)
        self.assertNotIn("个人秘密", agent.content_md)
        self.assertNotIn("不可见秘密", agent.content_md)

        defaulted = build_candidate_markdown(candidate, ProcessingContext(region_id="0755", audience=""))
        self.assertIn("坐席秘密", defaulted.content_md)
        self.assertIn("annotation_audience_defaulted", [
            warning.code for warning in defaulted.processing_warnings
        ])
        normalized_default = build_candidate_markdown(
            candidate, normalize_processing_context({"region_id": "0755"})
        )
        self.assertIn("annotation_audience_defaulted", [
            warning.code for warning in normalized_default.processing_warnings
        ])

        capture = _CapturingModel()
        asyncio.run(rerank_candidates(
            capture, "q", ProcessingContext(audience="customer"), None,
            [customer, *_processed(3)], KnowledgeProcessingOptions(),
        ))
        prompt = "\n".join(capture.calls)
        for secret in ("坐席秘密", "内部秘密", "个人秘密", "不可见秘密", "未标记秘密", "覆盖秘密"):
            self.assertNotIn(secret, prompt)

    def test_indeterminate_applicability_is_kept_with_structured_warnings(self):
        normalized = normalize_knowledge_candidates([
            {"knowledge_id": "normal", "knowledge_name": "正常", "content": "A",
             "applicability": {
                 "status": "active", "effective_start": "2026-01-01",
                 "effective_end": "2026-12-31",
             }},
            {"knowledge_id": "status", "knowledge_name": "未知状态", "content": "B",
             "applicability": {"status": "mystery"}},
            {"knowledge_id": "start", "knowledge_name": "非法开始", "content": "C",
             "applicability": {"effective_start": "not-a-start"}},
            {"knowledge_id": "end", "knowledge_name": "非法结束", "content": "D",
             "applicability": {"effective_end": "not-an-end"}},
            {"knowledge_id": "multiple", "knowledge_name": "多异常", "content": "E",
             "applicability": {
                 "status": "unknown", "effective_start": "bad-start",
                 "effective_end": "bad-end", "region_ids": {"unexpected": "shape"},
             }},
        ])
        filtered, decisions, filter_warnings = filter_candidates(
            normalized.candidates,
            ProcessingContext(request_time="2026-08-28T12:00:00+08:00"),
        )
        self.assertEqual([candidate.knowledge_id for candidate in filtered], [
            "normal", "status", "start", "end", "multiple",
        ])
        self.assertTrue(all(decision.accepted for decision in decisions))
        all_warnings = [*normalized.warnings, *filter_warnings]
        by_id = {}
        for warning in all_warnings:
            by_id.setdefault(warning.knowledge_id, []).append(warning)
        self.assertNotIn("normal", by_id)
        self.assertIn("unknown_status", [warning.code for warning in by_id["status"]])
        self.assertIn("invalid_start_time", [warning.code for warning in by_id["start"]])
        self.assertIn("invalid_end_time", [warning.code for warning in by_id["end"]])
        multiple_codes = {warning.code for warning in by_id["multiple"]}
        self.assertTrue({
            "unknown_status", "invalid_start_time", "invalid_end_time",
            "invalid_applicability_field",
        } <= multiple_codes)
        for warning in all_warnings:
            if warning.code in {
                "unknown_status", "invalid_start_time", "invalid_end_time",
                "invalid_applicability_field",
            }:
                self.assertTrue(warning.knowledge_id)
                self.assertTrue(warning.field)
                self.assertEqual(warning.details["warning_code"], warning.code)
                self.assertIn("raw_value", warning.details)
                self.assertNotIn("正文", json.dumps(warning.to_dict(), ensure_ascii=False))

    def test_knowledge_and_atom_applicability(self):
        normalized = normalize_knowledge_candidates([
            {"knowledge_id": "expired", "knowledge_name": "A", "content": "x",
             "applicability": {"effective_end": "2025-01-01"}},
            {"knowledge_id": "valid", "knowledge_name": "B",
             "applicability": {"regions": ["河南"]}, "atoms": [
                {"param_name": "北京专用", "content": "x",
                 "applicability": {"regions": ["北京"]}},
                {"param_name": "河南可用", "content": "y",
                 "applicability": {"regions": ["河南"]}},
            ]},
        ]).candidates
        filtered, decisions, warnings = filter_candidates(
            normalized, ProcessingContext(region="河南", as_of="2026-08-27")
        )
        self.assertEqual([item.knowledge_id for item in filtered], ["valid"])
        self.assertEqual([atom.title for atom in filtered[0].atoms], ["河南可用"])
        self.assertFalse(decisions[0].accepted)
        self.assertIn("atom_not_applicable", [warning.code for warning in warnings])

    def test_richtext_html_json_structured_table_and_image_metadata(self):
        value = {
            "type": "blocks",
            "children": [
                {"type": "paragraph", "children": [{"text": "业务说明", "id": "internal"}]},
                {"type": "table", "rows": [["名称", "值"], ["月费", "59元"]]},
                {"type": "img", "src": "https://example.invalid/secret.png", "alt": "资费图"},
                {"type": "unknown", "nodes": [{"value": "递归文本", "url": "https://bad"}]},
            ],
        }
        text, warnings = render_richtext_with_warnings(json.dumps(value, ensure_ascii=False))
        self.assertIn("业务说明", text)
        self.assertIn("| 名称 | 值 |", text)
        self.assertIn("[图片：资费图]", text)
        self.assertIn("递归文本", text)
        self.assertNotIn("example.invalid", text)
        self.assertNotIn("internal", text)
        self.assertIn("unknown_richtext_node", [warning.code for warning in warnings])

    def test_markdown_is_idempotent_and_except_rule_uses_standard_field(self):
        candidate = normalize_knowledge_candidates([{
            "knowledge_id": "K1",
            "knowledge_name": "套餐说明",
            "content": "<h1>套餐说明</h1><p>主体内容</p>",
            "atoms": [{
                "param_name": "流量", "content": 30, "wkuntt": "GB",
                "except_rules": {"region": "北京"}, "annotation": "当月有效",
            }],
        }]).candidates[0]
        context = ProcessingContext(region="河南")
        once = build_candidate_markdown(candidate, context)
        twice = build_candidate_markdown(once, context)
        self.assertEqual(once.content_md, twice.content_md)
        self.assertEqual(once.content_md.count("# 套餐说明"), 1)
        self.assertIn("30 GB", once.content_md)
        self.assertIn("例外规则", once.content_md)
        self.assertIn("备注", once.content_md)

    def test_missing_id_is_processed_but_never_reranked(self):
        candidate = normalize_knowledge_candidates([{
            "knowledge_name": "无 ID 知识", "content": "仍应转换 Markdown",
        }]).candidates[0]
        filtered, decisions, _ = filter_candidates([candidate], ProcessingContext())
        self.assertTrue(decisions[0].accepted)
        processed = build_candidate_markdown(filtered[0])
        result = asyncio.run(rerank_candidates(
            _CapturingModel(), "q", ProcessingContext(), None,
            [processed, *_processed(3)], KnowledgeProcessingOptions(),
        ))
        self.assertNotIn(None, [item.knowledge_id for item in result.candidates])
        self.assertIn("rerank_missing_knowledge_id", [warning.code for warning in result.warnings])

    def test_except_rules_json_string_dict_list_and_malformed_warning(self):
        candidate = normalize_knowledge_candidates([{
            "knowledge_id": "K1", "knowledge_name": "例外规则", "atoms": [
                {"param_name": "JSON命中", "content": "A", "except_rules": '{"region_id":"0755","exclude":true}'},
                {"param_name": "dict不命中", "content": "B", "except_rules": {"region_id": "010"}},
                {"param_name": "list命中", "content": "C", "except_rules": [{"region_id": "0755", "exclude": True}]},
                {"param_name": "损坏JSON", "content": "D", "except_rules": "{bad-json"},
            ],
        }]).candidates[0]
        processed = build_candidate_markdown(candidate, ProcessingContext(region_id="0755"))
        self.assertNotIn("JSON命中", processed.content_md)
        self.assertNotIn("list命中", processed.content_md)
        self.assertIn("dict不命中", processed.content_md)
        self.assertIn("损坏JSON", processed.content_md)
        self.assertIn("{bad-json", processed.content_md)
        codes = [warning.code for warning in processed.processing_warnings]
        self.assertEqual(codes.count("atom_excepted"), 2)
        self.assertIn("invalid_except_rules_json", codes)

    def test_falsey_scalar_content_is_kept_and_only_true_empty_values_are_filtered(self):
        raw = [
            {"knowledge_id": "zero", "knowledge_name": "零资费", "content": 0},
            {"knowledge_id": "false", "knowledge_name": "布尔值", "content": False},
            {"knowledge_id": "blank", "knowledge_name": "空字符", "content": "  "},
            {"knowledge_id": "list", "knowledge_name": "空列表", "content": []},
            {"knowledge_id": "dict", "knowledge_name": "空对象", "content": {}},
            {"knowledge_id": "none", "knowledge_name": "空值", "content": None},
        ]
        candidates = normalize_knowledge_candidates(raw).candidates
        filtered, _, _ = filter_candidates(candidates, ProcessingContext())
        self.assertEqual([candidate.knowledge_id for candidate in filtered], ["zero", "false"])
        rendered = [build_candidate_markdown(candidate).content_md for candidate in filtered]
        self.assertTrue(rendered[0].endswith("0"))
        self.assertTrue(rendered[1].endswith("false"))

    def test_atom_dedupe_keeps_different_annotation_and_units(self):
        candidate = normalize_knowledge_candidates([{
            "knowledge_id": "K1", "knowledge_name": "去重", "atoms": [
                {"atom_id": "same", "group_id": "G", "param_name": "额度", "content": 10,
                 "annotation": "备注A", "wkuntt": "MB"},
                {"atom_id": "same", "group_id": "G", "param_name": "额度", "content": 10,
                 "annotation": "备注B", "wkuntt": "GB"},
                {"atom_id": "other", "group_id": "G", "param_name": "额度", "content": 10,
                 "annotation": "备注A", "wkuntt": "MB"},
            ],
        }]).candidates[0]
        atoms, warnings = process_atoms(candidate.atoms, ProcessingContext())
        self.assertEqual(len(atoms), 2)
        self.assertEqual({(atom.annotation, atom.wkuntt) for atom in atoms}, {
            ("备注A", "MB"), ("备注B", "GB"),
        })
        self.assertEqual([warning.code for warning in warnings].count("duplicate_atom"), 1)
        markdown = build_candidate_markdown(candidate).content_md
        self.assertIn("10 MB", markdown)
        self.assertIn("10 GB", markdown)
        self.assertIn("备注A", markdown)
        self.assertIn("备注B", markdown)

    def test_default_business_timezone_and_request_time_precedence(self):
        from kbagent.shared.knowledge_processing import applicability as applicability_module

        utc_previous_day = datetime(2026, 8, 27, 16, 30, tzinfo=timezone.utc)
        beijing_after_midnight = utc_previous_day.astimezone(applicability_module._BUSINESS_TIMEZONE)
        rule = Applicability(effective_start="2026-08-28")
        with patch.object(
            applicability_module, "_now_in_business_timezone", return_value=beijing_after_midnight
        ):
            self.assertEqual(evaluate_applicability(rule, ProcessingContext()), [])
        context = ProcessingContext(
            request_time="2026-08-27T16:30:00+00:00",
            as_of="2026-08-27T10:00:00+08:00",
        )
        self.assertEqual(evaluate_applicability(rule, context), [])

    def test_html_and_structured_nested_lists_keep_indentation(self):
        html_text, _ = render_richtext_with_warnings(
            "<ul><li>父项<ul><li>子项</li></ul></li></ul>"
        )
        self.assertIn("- 父项", html_text)
        self.assertIn("  - 子项", html_text)
        structured = {
            "type": "ul",
            "children": [{
                "type": "li",
                "children": [
                    {"text": "父项"},
                    {"type": "ol", "children": [{"type": "li", "text": "子项"}]},
                ],
            }],
        }
        structured_text, _ = render_richtext_with_warnings(structured)
        self.assertIn("- 父项", structured_text)
        self.assertIn("  1. 子项", structured_text)

    def test_region_id_and_name_are_compared_only_with_same_representation(self):
        id_rule = Applicability(region_ids=["0755"])
        name_rule = Applicability(regions=["深圳"])
        both_rule = Applicability(region_ids=["0755"], regions=["深圳"])

        self.assertEqual(evaluate_applicability(
            id_rule, ProcessingContext(region_id="0755", region="0755")
        ), [])
        self.assertIn("region_not_applicable", evaluate_applicability(
            id_rule, ProcessingContext(region_id="010", region="010")
        ))
        self.assertEqual(evaluate_applicability(
            name_rule, ProcessingContext(region_name="深圳", region="深圳")
        ), [])
        self.assertIn("region_not_applicable", evaluate_applicability(
            name_rule, ProcessingContext(region_name="广州", region="广州")
        ))
        self.assertEqual(evaluate_applicability(
            both_rule, ProcessingContext(region_id="0755", region_name="深圳", region="深圳")
        ), [])
        self.assertIn("region_not_applicable", evaluate_applicability(
            both_rule, ProcessingContext(region_id="0755", region_name="广州", region="广州")
        ))
        # 上下文和候选使用不同表示时保守保留。
        self.assertEqual(evaluate_applicability(
            name_rule, ProcessingContext(region_id="0755", region="0755")
        ), [])
        self.assertEqual(evaluate_applicability(
            id_rule, ProcessingContext(region_name="深圳", region="深圳")
        ), [])

    def test_all_html_like_strings_use_sanitizer(self):
        script_only, _ = render_richtext_with_warnings("<script>alert('secret')</script>")
        style_only, _ = render_richtext_with_warnings("<style>.secret{display:block}</style>")
        section, _ = render_richtext_with_warnings("<section>业务正文</section>")
        link, _ = render_richtext_with_warnings('<a href="https://invalid.example">链接文本</a>')
        plain, _ = render_richtext_with_warnings("普通文本 1 < 2，不是 HTML")
        nested, _ = render_richtext_with_warnings(
            "<ul><li>父项<ul><li>子项</li></ul></li></ul>"
        )
        self.assertEqual(script_only, "")
        self.assertEqual(style_only, "")
        self.assertEqual(section, "业务正文")
        self.assertEqual(link, "链接文本")
        self.assertNotIn("href", link)
        self.assertEqual(plain, "普通文本 1 < 2，不是 HTML")
        self.assertIn("  - 子项", nested)

    def test_compound_except_requires_every_declared_context_value(self):
        compound = {"region_id": "0755", "channel_code": "10086"}
        self.assertTrue(except_rules_match(
            compound, ProcessingContext(region_id="0755", channel_code="10086")
        ))
        self.assertFalse(except_rules_match(
            compound, ProcessingContext(region_id="0755", channel_code="web")
        ))
        self.assertFalse(except_rules_match(
            compound, ProcessingContext(channel_code="10086")
        ))
        self.assertFalse(except_rules_match(
            compound, ProcessingContext(region_id="0755")
        ))
        self.assertTrue(except_rules_match(
            {"region_id": "0755"}, ProcessingContext(region_id="0755")
        ))

    def test_except_conditions_overrides_and_explicit_exclusion_are_separate(self):
        rules = parse_except_rule(json.dumps([
            {
                "region_id": "0755", "channel_code": "10086",
                "value": "深圳地区特殊内容", "annotation": "深圳办理说明", "unit": "次",
            },
        ], ensure_ascii=False))
        self.assertTrue(match_except_conditions(
            rules[0], ProcessingContext(region_id="0755", channel_code="10086")
        ))
        self.assertFalse(match_except_conditions(
            rules[0], ProcessingContext(region_id="0755")
        ))
        atom = KnowledgeAtom(
            atom_id="A1", param_name="办理规则", title="办理规则", content="默认内容",
            annotation="默认备注", group_id="G1", group="G1", arrange_seq_number=2, order=2,
        )
        overridden = apply_except_override(atom, rules[0])
        self.assertIsNotNone(overridden)
        self.assertEqual((overridden.content, overridden.annotation, overridden.wkuntt), (
            "深圳地区特殊内容", "深圳办理说明", "次",
        ))
        self.assertEqual((overridden.atom_id, overridden.param_name, overridden.group_id,
                          overridden.arrange_seq_number), ("A1", "办理规则", "G1", 2))

        matched, warnings = process_atoms(
            [atom], ProcessingContext(region_id="0755", channel_code="10086")
        )
        # 默认原子本身没有 except_rules，不会被直接覆盖。
        self.assertEqual(matched[0].content, "默认内容")
        self.assertEqual(warnings, [])
        atom.except_rules = json.dumps(rules, ensure_ascii=False)
        matched, warnings = process_atoms(
            [atom], ProcessingContext(region_id="0755", channel_code="10086")
        )
        self.assertEqual((matched[0].content, matched[0].annotation), (
            "深圳地区特殊内容", "深圳办理说明",
        ))
        self.assertIn("atom_except_overridden", [warning.code for warning in warnings])
        unmatched, _ = process_atoms(
            [atom], ProcessingContext(region_id="010", channel_code="10086")
        )
        self.assertEqual(unmatched[0].content, "默认内容")

        excluded_atom = copy.deepcopy(atom)
        excluded_atom.except_rules = {
            "region_id": "0755", "exclude": True, "value": "不得作为覆盖正文"
        }
        excluded, warnings = process_atoms(
            [excluded_atom], ProcessingContext(region_id="0755", channel_code="10086")
        )
        self.assertEqual(excluded, [])
        self.assertIn("atom_excepted", [warning.code for warning in warnings])

    def test_supported_content_types_share_analysis_and_renderability_rules(self):
        candidates = normalize_knowledge_candidates([
            {"knowledge_id": "s", "knowledge_name": "字符串", "content": "正文"},
            {"knowledge_id": "d", "knowledge_name": "对象", "content": {"text": "正文"}},
            {"knowledge_id": "l", "knowledge_name": "列表", "content": ["正文"]},
            {"knowledge_id": "i", "knowledge_name": "整数", "content": 0},
            {"knowledge_id": "f", "knowledge_name": "浮点", "content": 1.5},
            {"knowledge_id": "b", "knowledge_name": "布尔", "content": False},
            {"knowledge_id": "e", "knowledge_name": "空HTML", "content": "<p></p>"},
        ]).candidates
        analysis = analyze_candidates(candidates)
        self.assertEqual(analysis["invalid_types"], {})
        self.assertEqual(analysis["missing_fields"].get("content"), 1)
        self.assertTrue(all(is_supported_content_type(item.content) for item in candidates))
        self.assertEqual(
            [is_renderable_content(item.content) for item in candidates],
            [True, True, True, True, True, True, False],
        )
        filtered, _, _ = filter_candidates(candidates, ProcessingContext())
        self.assertEqual([item.knowledge_id for item in filtered], ["s", "d", "l", "i", "f", "b"])

    def test_html_table_inline_tags_stay_inside_cells(self):
        html_tables = [
            "<table><tr><th><strong>名称</strong></th><th>值</th></tr><tr><td><em>月费</em></td><td><a href='x'>59元</a></td></tr></table>",
            "<table><tr><th>普通</th><th>表格</th></tr><tr><td>A</td><td>B</td></tr></table>",
            "<section><div><table><tr><th><strong><em>嵌套</em></strong></th></tr><tr><td>内容</td></tr></table></div></section>",
        ]
        rendered = [render_richtext_with_warnings(value)[0] for value in html_tables]
        self.assertTrue(rendered[0].splitlines()[0].startswith("| **名称** |"))
        self.assertIn("| *月费* | 59元 |", rendered[0])
        self.assertNotIn("href", rendered[0])
        self.assertIn("| 普通 | 表格 |", rendered[1])
        self.assertIn("| ***嵌套*** |", rendered[2])
        self.assertTrue(all(not text.startswith("****") for text in rendered))

    def test_html_table_pipes_are_escaped_exactly_once(self):
        rendered, _ = render_richtext_with_warnings(
            "<table>"
            "<tr><th>A|B</th><th>P|Q|R</th></tr>"
            "<tr><td><strong>X|Y</strong></td>"
            "<td><em>M|N</em> <a href='secret'>链接|文本</a></td></tr>"
            "<tr><td>普通</td><td>多列</td></tr>"
            "</table>"
        )
        self.assertIn("| A\\|B | P\\|Q\\|R |", rendered)
        self.assertIn("| **X\\|Y** | *M\\|N* 链接\\|文本 |", rendered)
        self.assertIn("| 普通 | 多列 |", rendered)
        self.assertNotIn("\\\\|", rendered)
        self.assertNotIn("secret", rendered)

    def test_atom_sort_uses_display_sequence_and_stays_stable(self):
        atoms = [
            KnowledgeAtom(atom_id="g1-2", group_id="G1", arrange_seq_number=2, content="A"),
            KnowledgeAtom(atom_id="g2-1", group_id="G2", arrange_seq_number=1, content="B"),
            KnowledgeAtom(atom_id="g3-2", group_id="G3", arrange_seq_number=2, content="C"),
            KnowledgeAtom(atom_id="g3-1a", group_id="G3", arrange_seq_number=1, content="D"),
            KnowledgeAtom(atom_id="g3-1b", group_id="G3", arrange_seq_number=1, content="E"),
            KnowledgeAtom(atom_id="missing-a", group_id="G4", arrange_seq_number=None, content="F"),
            KnowledgeAtom(atom_id="missing-b", group_id="G4", arrange_seq_number=None, content="G"),
        ]
        ordered, _ = process_atoms(atoms, ProcessingContext())
        self.assertEqual([atom.atom_id for atom in ordered], [
            "g2-1", "g3-1a", "g3-1b", "g3-2", "g1-2", "missing-a", "missing-b",
        ])

    def test_unrenderable_candidates_are_filtered_and_post_atom_empty_is_warned(self):
        candidates = normalize_knowledge_candidates([
            {"knowledge_id": "empty-html", "knowledge_name": "空HTML", "content": "<p></p>"},
            {"knowledge_id": "blank-html", "knowledge_name": "空白HTML", "content": "<p>   </p>"},
            {"knowledge_id": "empty-node", "knowledge_name": "空节点", "content": {"type": "paragraph", "children": []}},
            {"knowledge_id": "empty-atom", "knowledge_name": "空原子", "atoms": [{"param_name": "字段", "content": ""}]},
            {"knowledge_id": "script", "knowledge_name": "脚本", "content": "<script>x()</script>"},
            {"knowledge_id": "title", "knowledge_name": "仅标题", "content": "<h1>仅标题</h1>"},
            {"knowledge_id": "zero", "knowledge_name": "零", "content": 0},
            {"knowledge_id": "false", "knowledge_name": "布尔", "content": False},
        ]).candidates
        filtered, decisions, _ = filter_candidates(candidates, ProcessingContext())
        self.assertEqual([candidate.knowledge_id for candidate in filtered], ["zero", "false"])
        rejected = {
            decision.knowledge_id: decision.reasons for decision in decisions if not decision.accepted
        }
        for knowledge_id in ("empty-html", "blank-html", "empty-node", "empty-atom", "script", "title"):
            self.assertIn("empty_content", rejected[knowledge_id])

        excepted = normalize_knowledge_candidates([{
            "knowledge_id": "excepted", "knowledge_name": "全部例外", "atoms": [{
                "param_name": "字段", "content": "A",
                "except_rules": {"region_id": "0755", "exclude": True},
            }],
        }]).candidates
        pre_markdown, _, _ = filter_candidates(excepted, ProcessingContext(region_id="0755"))
        processed, warnings = build_knowledge_markdown(
            pre_markdown, ProcessingContext(region_id="0755")
        )
        self.assertEqual(processed, [])
        self.assertIn("empty_rendered_content", [warning.code for warning in warnings])

    def test_rerank_eligible_count_matches_shared_rerank_filter(self):
        pipeline = process_knowledge_candidates([
            {"knowledge_name": "缺ID", "content": "A"},
            {"knowledge_id": "  ", "knowledge_name": "空ID", "content": "B"},
            {"knowledge_id": "valid", "knowledge_name": "正常", "content": "C"},
        ])
        self.assertEqual(pipeline.meta.processed_count, 3)
        self.assertEqual(pipeline.meta.rerank_eligible_count, 1)
        self.assertEqual([is_rerank_eligible(item) for item in pipeline.processed], [False, False, True])
        reranked = asyncio.run(rerank_candidates(
            _CapturingModel(), "q", ProcessingContext(), None,
            pipeline.processed, KnowledgeProcessingOptions(),
        ))
        self.assertEqual(reranked.details["eligible_count"], 1)
        self.assertEqual([item.knowledge_id for item in reranked.candidates], ["valid"])


class TestRerank(unittest.TestCase):
    def test_rerank_finalization_metadata_is_consistent(self):
        model = asyncio.run(rerank_candidates(
            _CapturingModel(), "q", ProcessingContext(), None,
            _processed(8), KnowledgeProcessingOptions(),
        ))
        self.assertEqual(model.details["global"]["mode"], "model")
        self.assertFalse(model.degraded)
        self.assertEqual(model.details["global"]["fallback_count"], 0)

        partial = asyncio.run(rerank_candidates(
            _CapturingModel(global_response=json.dumps({"ranked_ids": ["E005"]})),
            "q", ProcessingContext(), None, _processed(8), KnowledgeProcessingOptions(),
        ))
        self.assertEqual(partial.details["global"]["mode"], "model_with_fallback")
        self.assertTrue(partial.degraded)
        self.assertEqual(partial.details["global"]["fallback_count"], 2)

        fallback = asyncio.run(rerank_candidates(
            _CapturingModel(global_response="bad"), "q", ProcessingContext(), None,
            _processed(8), KnowledgeProcessingOptions(),
        ))
        self.assertEqual(fallback.details["global"]["mode"], "fallback")
        self.assertTrue(fallback.degraded)
        self.assertEqual(fallback.details["global"]["fallback_count"], 3)

        pool_shortage = asyncio.run(rerank_candidates(
            _CapturingModel(), "q", ProcessingContext(), None, _processed(4),
            KnowledgeProcessingOptions(batch_size=2, batch_top_k=1),
        ))
        self.assertEqual(pool_shortage.details["global"]["mode"], "model_with_fallback")
        self.assertTrue(pool_shortage.degraded)
        self.assertEqual(pool_shortage.details["global"]["fallback_count"], 1)

        insufficient = asyncio.run(rerank_candidates(
            _CapturingModel(), "q", ProcessingContext(), None,
            _processed(2), KnowledgeProcessingOptions(),
        ))
        self.assertEqual(insufficient.details["global"]["mode"], "insufficient_candidates")
        self.assertTrue(insufficient.degraded)
        self.assertEqual(insufficient.details["global"]["fallback_count"], 0)

    def test_standard_contract_preserves_filter_markdown_and_top3_baseline(self):
        raw = make_top100_candidates()
        original = copy.deepcopy(raw)
        ws = RunWorkspace(query="5G流量套餐")
        ws.data.update({
            "processing_context": {
                "region_id": "0755",
                "region_name": "深圳",
                "channel_code": "10086",
                "request_time": "2026-08-28T10:00:00+08:00",
                "audience": "agent",
            },
            "retrieval_query": "5G 套餐 流量",
            "knowledge_candidates": raw,
        })
        set_workspace(ws)
        result = asyncio.run(KnowledgeProcessingOrchestrator(ScriptedChatModel()).run())
        self.assertEqual([item.knowledge_id for item in result], [
            "REAL-KNOWLEDGE-042", "REAL-KNOWLEDGE-073", "REAL-KNOWLEDGE-099",
        ])
        self.assertEqual(len(ws.data["filtered_knowledge_candidates"]), 95)
        self.assertEqual(len(ws.data["processed_knowledge_candidates"]), 95)
        first_markdown = ws.data["processed_knowledge_candidates"][0].content_md
        self.assertIn("深圳地区专享59元含100GB", first_markdown)
        self.assertNotIn("全国默认资费说明", first_markdown)
        self.assertEqual(len(ws.data["rerank_details"]["batches"]), 5)
        self.assertEqual(len(ws.data["rerank_details"]["global"]["pool_ids"]), 25)
        self.assertEqual([item.rerank_rank for item in result], [1, 2, 3])
        self.assertFalse(ws.data["processing_meta"].degraded)
        self.assertEqual(raw, original)

        capture = _CapturingModel()
        asyncio.run(rerank_candidates(
            capture, "查询", ProcessingContext(), None, _processed(), KnowledgeProcessingOptions()
        ))
        joined = "\n".join(capture.calls)
        for item in _processed():
            self.assertNotIn(item.knowledge_id, joined)

    def test_invalid_json_falls_back_to_retrieval_rank(self):
        model = _CapturingModel(batch_response="not json", global_response="not json")
        result = asyncio.run(rerank_candidates(
            model, "q", ProcessingContext(), None, _processed(30), KnowledgeProcessingOptions()
        ))
        self.assertEqual([item.retrieval_rank for item in result.candidates], [1, 2, 3])
        self.assertTrue(result.degraded)
        self.assertIn("rerank_invalid_json", [warning.code for warning in result.warnings])

    def test_partial_global_result_is_kept_then_supplemented(self):
        model = _CapturingModel(global_response=json.dumps({"ranked_ids": ["E005", "UNKNOWN"]}))
        result = asyncio.run(rerank_candidates(
            model, "q", ProcessingContext(), None, _processed(10), KnowledgeProcessingOptions()
        ))
        self.assertEqual([item.knowledge_id for item in result.candidates], ["SECRET-005", "SECRET-001", "SECRET-002"])
        self.assertTrue(result.degraded)
        self.assertIn("rerank_unknown_id", [warning.code for warning in result.warnings])

    def test_model_exception_and_insufficient_candidates_degrade(self):
        failed = asyncio.run(rerank_candidates(
            _CapturingModel(raises=True), "q", ProcessingContext(), None,
            _processed(6), KnowledgeProcessingOptions(),
        ))
        self.assertEqual([item.retrieval_rank for item in failed.candidates], [1, 2, 3])
        self.assertTrue(failed.degraded)
        insufficient = asyncio.run(rerank_candidates(
            _CapturingModel(), "q", ProcessingContext(), None,
            _processed(2), KnowledgeProcessingOptions(),
        ))
        self.assertEqual(len(insufficient.candidates), 2)
        self.assertTrue(insufficient.degraded)

    def test_batch_rerank_timeout_degrades_without_blocking(self):
        result = asyncio.run(rerank_candidates(
            _CapturingModel(batch_delay=0.05), "q", ProcessingContext(), None,
            _processed(6), KnowledgeProcessingOptions(rerank_timeout_seconds=0.001),
        ))
        timeout_warnings = [warning for warning in result.warnings if warning.code == "rerank_timeout"]
        self.assertTrue(any(warning.field == "batch_1" for warning in timeout_warnings))
        self.assertTrue(result.degraded)
        self.assertEqual([item.rerank_rank for item in result.candidates], [1, 2, 3])

    def test_global_rerank_timeout_uses_deterministic_top3(self):
        result = asyncio.run(rerank_candidates(
            _CapturingModel(global_delay=0.05), "q", ProcessingContext(), None,
            _processed(8), KnowledgeProcessingOptions(rerank_timeout_seconds=0.001),
        ))
        timeout_warnings = [warning for warning in result.warnings if warning.code == "rerank_timeout"]
        self.assertTrue(any(warning.field == "global" for warning in timeout_warnings))
        self.assertEqual([item.retrieval_rank for item in result.candidates], [1, 2, 3])
        self.assertEqual([item.rerank_rank for item in result.candidates], [1, 2, 3])
        self.assertTrue(result.degraded)

    def test_rerank_rank_is_continuous_for_model_and_fallback_serialization(self):
        model_result = asyncio.run(rerank_candidates(
            _CapturingModel(global_response=json.dumps({"ranked_ids": ["E005", "E003", "E001"]})),
            "q", ProcessingContext(), None, _processed(8), KnowledgeProcessingOptions(),
        ))
        self.assertEqual([item.knowledge_id for item in model_result.candidates], [
            "SECRET-005", "SECRET-003", "SECRET-001",
        ])
        serialized = [item.to_dict() for item in model_result.candidates]
        self.assertEqual([item["rerank_rank"] for item in serialized], [1, 2, 3])
        self.assertEqual(
            [item["knowledge_id"] for item in sorted(serialized, key=lambda item: item["rerank_rank"])],
            ["SECRET-005", "SECRET-003", "SECRET-001"],
        )
        fallback = asyncio.run(rerank_candidates(
            _CapturingModel(batch_response="bad", global_response="bad"),
            "q", ProcessingContext(), None, _processed(8), KnowledgeProcessingOptions(),
        ))
        self.assertEqual([item.rerank_rank for item in fallback.candidates], [1, 2, 3])

    def test_prompt_uses_field_whitelist_and_never_rewrites_business_text(self):
        candidates = []
        real_ids = ["1", "3", "ID", "E001", "K5", "K6"]
        for index, knowledge_id in enumerate(real_ids, 1):
            title = "100元套餐 ID" if index == 1 else f"业务标题 {index}"
            content_md = (
                "# 100元套餐 ID\n\n包含30GB，业务编号1和E001自然出现"
                if index == 1 else f"# 业务标题 {index}\n\n正文 {index}"
            )
            candidates.append(ProcessedKnowledge(
                knowledge_id=knowledge_id,
                name=title,
                content="工程侧正文不进入 Prompt",
                retrieval_rank=index,
                content_md=content_md,
                metadata={"knowledge_id": knowledge_id, "private": "不应发送"},
                raw={"knowledge_id": knowledge_id, "raw": "不应发送"},
            ))
        model = _CapturingModel()
        result = asyncio.run(rerank_candidates(
            model, "100元套餐有30GB吗？ID和E001是自然文本",
            ProcessingContext(
                region_id="0755", region_name="深圳", channel_code="10086",
                request_time="2026-08-28T10:00:00+08:00", audience="agent",
                attributes={"1": "工程映射", "E001": {"knowledge_id": "3"}},
            ),
            "检索100元和30GB", candidates, KnowledgeProcessingOptions(),
        ))
        self.assertFalse(result.degraded)
        self.assertEqual([item.knowledge_id for item in result.candidates], ["1", "3", "ID"])
        payloads = []
        for call in model.calls:
            serialized = call.split("RERANK_INPUT_BEGIN\n")[-1].split("\nRERANK_INPUT_END")[0]
            payloads.append(json.loads(serialized))
        self.assertEqual([payload["top_k"] for payload in payloads], [5, 3])
        for payload in payloads:
            self.assertEqual(payload["query"], "100元套餐有30GB吗？ID和E001是自然文本")
            self.assertEqual(payload["retrieval_query"], "检索100元和30GB")
            self.assertEqual(set(payload["context"]), {
                "region_id", "region_name", "channel_code", "request_time", "audience",
                "customer_type",
            })
            self.assertNotIn("attributes", payload["context"])
            evidence_ids = [item["evidence_id"] for item in payload["candidates"]]
            self.assertTrue(all(re.fullmatch(r"E\d{3}", evidence_id) for evidence_id in evidence_ids))
            self.assertTrue(all(set(item) == {"evidence_id", "title", "content_md"}
                                for item in payload["candidates"]))
            serialized_payload = json.dumps(payload, ensure_ascii=False)
            for forbidden_field in ('"knowledge_id"', '"evidence_map"', '"raw"', '"metadata"', '"attributes"'):
                self.assertNotIn(forbidden_field, serialized_payload)
        self.assertEqual(payloads[0]["candidates"][0]["evidence_id"], "E001")
        self.assertEqual(payloads[0]["candidates"][0]["title"], "100元套餐 ID")
        self.assertEqual(
            payloads[0]["candidates"][0]["content_md"],
            "# 100元套餐 ID\n\n包含30GB，业务编号1和E001自然出现",
        )


class TestToolsAndOrchestrator(unittest.TestCase):
    def test_all_warnings_survive_every_processing_stage_in_stable_order(self):
        ws = RunWorkspace(query="风险告警")
        ws.data.update({
            "processing_context": {
                "region_name": "河南", "audience": "customer",
                "request_time": "2026-08-28T12:00:00+08:00",
            },
            "retrieval_query": "风险告警",
            "knowledge_candidates": [
                {"knowledge_id": "status", "knowledge_name": "未知状态", "content": "A",
                 "applicability": {"status": "mystery"}},
                {"knowledge_id": "start", "knowledge_name": "非法开始", "content": "B",
                 "applicability": {"effective_start": "bad-start"}},
                {"knowledge_id": "end", "knowledge_name": "非法结束", "content": "C",
                 "applicability": {"effective_end": "bad-end"}},
                {"knowledge_id": "shape", "knowledge_name": "异常范围", "content": "D",
                 "applicability": {"region_ids": {"bad": "shape"}}},
                {"knowledge_id": "atom", "knowledge_name": "原子告警", "content": "主体",
                 "atoms": [
                     {"param_name": "不适用", "content": "X",
                      "applicability": {"regions": ["北京"]}},
                     {"param_name": "受限注解", "content": "Y",
                      "annotation": {"visibility": "agent", "content": "内部说明"}},
                 ]},
            ],
        })
        set_workspace(ws)
        asyncio.run(KnowledgeProcessingOrchestrator(_CapturingModel()).run())
        warnings = ws.data["processing_warnings"]
        codes = [warning.code for warning in warnings]
        for code in (
            "unknown_status", "invalid_start_time", "invalid_end_time",
            "invalid_applicability_field", "atom_not_applicable", "annotation_filtered",
        ):
            self.assertIn(code, codes)
        keys = [
            (warning.code, warning.message, warning.source_index, warning.knowledge_id,
             warning.field, json.dumps(warning.details, ensure_ascii=False, sort_keys=True, default=str))
            for warning in warnings
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(ws.data["processing_meta"].warning_count, len(warnings))

    def test_workspace_degradation_reasons_come_from_authoritative_rerank_details(self):
        async def run_case(model, options, count=8):
            ws = RunWorkspace(query="套餐")
            ws.data.update({
                "processing_context": {"audience": "agent"},
                "retrieval_query": "套餐",
                "knowledge_candidates": make_top100_candidates()[:count],
            })
            set_workspace(ws)
            await KnowledgeProcessingOrchestrator(model, options).run()
            return ws

        normal = asyncio.run(run_case(_CapturingModel(), KnowledgeProcessingOptions()))
        self.assertFalse(normal.data["processing_meta"].degraded)
        self.assertEqual(normal.data["processing_meta"].degradation_reasons, [])

        pool_shortage = asyncio.run(run_case(
            _CapturingModel(),
            KnowledgeProcessingOptions(global_pool_size=2, final_top_k=3),
        ))
        pool_reasons = pool_shortage.data["processing_meta"].degradation_reasons
        self.assertTrue(pool_shortage.data["processing_meta"].degraded)
        self.assertIn("retrieval_rank_supplement", pool_reasons)
        self.assertEqual(pool_reasons, pool_shortage.data["rerank_details"]["fallback_reasons"])

        timeout = asyncio.run(run_case(
            _CapturingModel(global_delay=0.05),
            KnowledgeProcessingOptions(rerank_timeout_seconds=0.001),
        ))
        timeout_reasons = timeout.data["processing_meta"].degradation_reasons
        self.assertIn("rerank_timeout", timeout_reasons)
        self.assertIn("retrieval_rank_supplement", timeout_reasons)

        invalid = asyncio.run(run_case(
            _CapturingModel(batch_response="bad", global_response="bad"),
            KnowledgeProcessingOptions(),
            count=25,
        ))
        invalid_reasons = invalid.data["processing_meta"].degradation_reasons
        self.assertIn("rerank_invalid_json", invalid_reasons)
        self.assertIn("retrieval_rank_supplement", invalid_reasons)
        self.assertIn("global_model_failed", invalid_reasons)
        self.assertEqual(len(invalid_reasons), len(set(invalid_reasons)))
        self.assertEqual(invalid_reasons, invalid.data["rerank_details"]["fallback_reasons"])

    def test_new_registry_is_independent_and_ordered(self):
        old_names = [tool.name for tool in PROCESSING_TOOLS]
        new_tools = build_knowledge_processing_tools(ScriptedChatModel(), KnowledgeProcessingOptions())
        self.assertEqual([tool.name for tool in new_tools], [
            "analyze_knowledge_candidates", "filter_knowledge_candidates",
            "build_knowledge_markdown", "rerank_knowledge_candidates",
        ])
        self.assertEqual(len(old_names), 7)
        self.assertNotIn("analyze_knowledge_candidates", old_names)

    def test_repeated_run_overwrites_outputs_and_keeps_input(self):
        raw = make_top100_candidates()[:8]
        original = copy.deepcopy(raw)
        ws = RunWorkspace(query="5G流量套餐")
        ws.data.update({
            "processing_context": {"region_name": "河南"},
            "retrieval_query": "5G 流量",
            "knowledge_candidates": raw,
        })
        set_workspace(ws)
        orchestrator = KnowledgeProcessingOrchestrator(ScriptedChatModel())
        first = asyncio.run(orchestrator.run())
        first_markdown = [item.content_md for item in ws.data["processed_knowledge_candidates"]]
        second = asyncio.run(orchestrator.run())
        self.assertEqual([item.knowledge_id for item in first], [item.knowledge_id for item in second])
        self.assertEqual(first_markdown, [item.content_md for item in ws.data["processed_knowledge_candidates"]])
        self.assertEqual(raw, original)
        self.assertEqual(ws.data["processing_meta"].stage_order, [
            "analyze", "filter", "build_markdown", "rerank",
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
