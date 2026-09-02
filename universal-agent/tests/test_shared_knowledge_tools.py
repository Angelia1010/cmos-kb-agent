"""公共知识 Tool 的真实注册、调用与 Processing 回归测试。"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import unittest
from typing import Any

sys.path.insert(0, "src")

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402

from kbagent.processing.agent import KnowledgeProcessingOrchestrator  # noqa: E402
from kbagent.processing.tools import (  # noqa: E402
    PROCESSING_TOOLS,
    build_knowledge_processing_tools,
)
from kbagent.scripted_model import ScriptedChatModel  # noqa: E402
from kbagent.shared.knowledge_processing.adapter import (  # noqa: E402
    normalize_knowledge_candidates,
    normalize_processing_context,
)
from kbagent.shared.knowledge_processing.analysis import analyze_candidates  # noqa: E402
from kbagent.shared.knowledge_processing.applicability import filter_candidates  # noqa: E402
from kbagent.shared.knowledge_processing.markdown import (  # noqa: E402
    build_knowledge_markdown as build_markdown_core,
)
from kbagent.shared.knowledge_processing.models import KnowledgeProcessingOptions  # noqa: E402
from kbagent.shared.tools import (  # noqa: E402
    SHARED_KNOWLEDGE_TOOLS,
    analyze_knowledge_candidates,
    build_knowledge_markdown,
    filter_knowledge_candidates,
)
from kbagent.shared.workspace import RunWorkspace, set_workspace  # noqa: E402
from tests.processing_mock_data import make_top100_candidates  # noqa: E402
from uniagent.agents.factory import create_agent  # noqa: E402
from uniagent.config.app_config import AppConfig  # noqa: E402
from uniagent.tools.registry import get_available_tools  # noqa: E402


def _minimal_candidates() -> list[dict]:
    return [
        {
            "knowledge_id": "K001",
            "knowledge_name": "流量查询说明",
            "content": "",
            "retrieval_rank": 1,
            "retrieval_score": 0.9,
            "applicability": {"status": "1", "channel_codes": ["1"]},
            "atoms": [{
                "atom_id": "A001",
                "group_id": "G001",
                "param_name": "查询方式",
                "param_type": "rich_text",
                "content": "<p>可通过指定渠道查询剩余流量。</p>",
                "except_rules": [],
                "annotation": "仅供坐席参考",
                "arrange_seq_number": 1,
                "wkuntt": None,
                "applicability": {},
            }],
        },
        {
            "knowledge_id": "K002",
            "knowledge_name": "已下架说明",
            "content": "不应保留",
            "retrieval_rank": 2,
            "retrieval_score": 0.8,
            "applicability": {"status": "offline"},
            "atoms": [],
        },
    ]


def _chain_candidates() -> list[dict]:
    return [
        {
            "knowledge_id": "CHAIN-KEEP",
            "knowledge_name": "串联保真测试",
            "content": "",
            "retrieval_rank": 1,
            "retrieval_score": 0.99,
            "matched_atom_ids": ["ATOM-OVERRIDE", "ATOM-MIDDLE"],
            "source_routes": ["keyword", "vector"],
            "knowledge_type": "synthetic",
            "template_id": "TPL-SYNTHETIC",
            "applicability": {
                "status": "1",
                "region_ids": ["200"],
                "channel_codes": ["1"],
                "conditions": {"customer_type": ["个人客户"]},
            },
            "metadata": {"trace": "CANDIDATE_METADATA_MARKER"},
            "raw": {"trace": "CANDIDATE_RAW_MARKER"},
            "atoms": [
                {
                    "atom_id": "ATOM-LATE",
                    "group_id": "GROUP-B",
                    "param_name": "后置内容",
                    "param_type": "text",
                    "content": "最后展示",
                    "except_rules": [],
                    "annotation": {"visibility": "public", "content": "公开备注"},
                    "arrange_seq_number": 30,
                    "wkuntt": "次",
                    "applicability": {"status": "1"},
                    "metadata": {"trace": "LATE_METADATA_MARKER"},
                    "raw": {"trace": "LATE_RAW_MARKER"},
                },
                {
                    "atom_id": "ATOM-OVERRIDE",
                    "group_id": "GROUP-A",
                    "param_name": "例外覆盖",
                    "param_type": "text",
                    "content": "默认值",
                    "except_rules": {
                        "region_id": "200",
                        "channel_code": "1",
                        "content": "命中覆盖值",
                        "wkuntt": "GB",
                        "annotation": {
                            "visibility": "agent",
                            "content": "例外坐席备注",
                        },
                    },
                    "annotation": {"visibility": "customer", "content": "原备注"},
                    "arrange_seq_number": 10,
                    "wkuntt": "MB",
                    "applicability": {"status": "1", "channel_codes": ["1"]},
                    "metadata": {"trace": "OVERRIDE_METADATA_MARKER"},
                    "raw": {"trace": "OVERRIDE_RAW_MARKER"},
                },
                {
                    "atom_id": "ATOM-MIDDLE",
                    "group_id": "GROUP-A",
                    "param_name": "中间内容",
                    "param_type": "text",
                    "content": "中间展示",
                    "except_rules": [],
                    "annotation": {"visibility": "agent", "content": "坐席可见备注"},
                    "arrange_seq_number": 20,
                    "wkuntt": "项",
                    "applicability": {},
                    "metadata": {"trace": "MIDDLE_METADATA_MARKER"},
                    "raw": {"trace": "MIDDLE_RAW_MARKER"},
                },
                {
                    "atom_id": "ATOM-FILTERED",
                    "group_id": "GROUP-Z",
                    "param_name": "应过滤内容",
                    "content": "不应进入 Markdown",
                    "except_rules": [],
                    "annotation": None,
                    "arrange_seq_number": 1,
                    "wkuntt": None,
                    "applicability": {"channel_codes": ["2"]},
                    "metadata": {"trace": "FILTERED_METADATA_MARKER"},
                    "raw": {"trace": "FILTERED_RAW_MARKER"},
                },
            ],
        },
        {
            "knowledge_id": "CHAIN-REJECT",
            "knowledge_name": "应过滤候选",
            "content": "不应进入下一阶段",
            "retrieval_rank": 2,
            "retrieval_score": 0.5,
            "matched_atom_ids": [],
            "source_routes": ["keyword"],
            "applicability": {"status": "offline"},
            "metadata": {"trace": "REJECT_METADATA_MARKER"},
            "raw": {"trace": "REJECT_RAW_MARKER"},
            "atoms": [],
        },
    ]


def _contains_marker(value: Any, marker: str) -> bool:
    return marker in json.dumps(value, ensure_ascii=False, default=str)


class _ToolReturnCaptureModel(BaseChatModel):
    """只发起一次公共 ToolCall，并捕获下一轮模型收到的 ToolMessage。"""

    tool_args: dict[str, Any]
    call_count: int = 0
    captured_content: str = ""
    captured_artifact: Any = None

    @property
    def _llm_type(self) -> str:
        return "shared-tool-return-capture"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.call_count += 1
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            response = AIMessage(
                content="调用公共过滤 Tool",
                tool_calls=[{
                    "name": filter_knowledge_candidates.name,
                    "args": copy.deepcopy(self.tool_args),
                    "id": "synthetic-public-tool-call",
                    "type": "tool_call",
                }],
            )
        else:
            latest = tool_messages[-1]
            self.captured_content = str(latest.content)
            self.captured_artifact = copy.deepcopy(latest.artifact)
            response = AIMessage(content="完成")
        return ChatResult(generations=[ChatGeneration(message=response)])


class TestSharedKnowledgeTools(unittest.TestCase):
    def test_tools_use_real_registry_with_unique_names_and_explicit_schemas(self):
        self.assertTrue(all(isinstance(item, BaseTool) for item in SHARED_KNOWLEDGE_TOOLS))
        self.assertEqual([item.name for item in SHARED_KNOWLEDGE_TOOLS], [
            "shared_analyze_knowledge_candidates",
            "shared_filter_knowledge_candidates",
            "shared_build_knowledge_markdown",
        ])
        registered = get_available_tools(
            AppConfig(),
            extra_tools=[
                *SHARED_KNOWLEDGE_TOOLS,
                *PROCESSING_TOOLS,
                *build_knowledge_processing_tools(
                    ScriptedChatModel(), KnowledgeProcessingOptions()
                ),
            ],
        )
        names = [item.name for item in registered]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 14)
        self.assertIn("candidates", analyze_knowledge_candidates.args)
        self.assertIn("processing_context", filter_knowledge_candidates.args)
        self.assertIn("options", build_knowledge_markdown.args)

    def test_tools_invoke_independently_without_mutation_or_workspace_pollution(self):
        raw = _minimal_candidates()
        original = copy.deepcopy(raw)
        ws = RunWorkspace(query="不应被读取", data={"sentinel": {"kept": True}})
        set_workspace(ws)
        workspace_before = copy.deepcopy(ws.data)

        analysis = analyze_knowledge_candidates.invoke({"candidates": raw})
        self.assertEqual(analysis["analysis"]["candidate_count"], 2)
        self.assertEqual(analysis["analysis"]["atom_count"], 1)

        rejected = filter_knowledge_candidates.invoke({
            "candidates": raw,
            "processing_context": {"channel_code": "OTHER", "audience": "agent"},
        })
        accepted = filter_knowledge_candidates.invoke({
            "candidates": raw,
            "processing_context": {"channel_code": "1", "audience": "agent"},
        })
        self.assertEqual(rejected["candidates"], [])
        self.assertEqual(
            [item["knowledge_id"] for item in accepted["candidates"]], ["K001"]
        )

        markdown = build_knowledge_markdown.invoke({
            "candidates": accepted["candidates"],
            "processing_context": {"channel_code": "1", "audience": "agent"},
        })
        self.assertEqual(len(markdown["candidates"]), 1)
        self.assertIn("# 流量查询说明", markdown["candidates"][0]["content_md"])
        self.assertIn("查询方式", markdown["candidates"][0]["content_md"])

        self.assertEqual(raw, original)
        self.assertEqual(ws.data, workspace_before)

    def test_analyze_filter_markdown_chain_preserves_fields_and_business_behavior(self):
        raw = _chain_candidates()
        original = copy.deepcopy(raw)
        context = {
            "region_id": "200",
            "region_name": "广东",
            "channel_code": "1",
            "request_time": "2026-09-01T14:00:00+08:00",
            "audience": "agent",
            "customer_type": "个人客户",
        }

        analyzed = analyze_knowledge_candidates.invoke({"candidates": raw})
        filtered = filter_knowledge_candidates.invoke({
            "candidates": analyzed["candidates"],
            "processing_context": context,
        })
        markdown = build_knowledge_markdown.invoke({
            "candidates": filtered["candidates"],
            "processing_context": context,
        })

        analyzed_keep = analyzed["candidates"][0]
        self.assertEqual(analyzed_keep["matched_atom_ids"], ["ATOM-OVERRIDE", "ATOM-MIDDLE"])
        self.assertEqual(analyzed_keep["source_routes"], ["keyword", "vector"])
        self.assertEqual(analyzed_keep["knowledge_type"], "synthetic")
        self.assertEqual(analyzed_keep["template_id"], "TPL-SYNTHETIC")
        self.assertEqual(analyzed_keep["applicability"]["region_ids"], ["200"])
        self.assertEqual(analyzed_keep["applicability"]["conditions"], {
            "customer_type": ["个人客户"],
        })
        analyzed_atoms = {item["atom_id"]: item for item in analyzed_keep["atoms"]}
        self.assertEqual(analyzed_atoms["ATOM-OVERRIDE"]["group_id"], "GROUP-A")
        self.assertEqual(analyzed_atoms["ATOM-OVERRIDE"]["arrange_seq_number"], 10)
        self.assertEqual(analyzed_atoms["ATOM-OVERRIDE"]["wkuntt"], "MB")
        self.assertEqual(
            analyzed_atoms["ATOM-OVERRIDE"]["applicability"]["channel_codes"], ["1"]
        )

        self.assertEqual([item["knowledge_id"] for item in filtered["candidates"]], [
            "CHAIN-KEEP",
        ])
        filtered_keep = filtered["candidates"][0]
        self.assertEqual(filtered_keep["matched_atom_ids"], analyzed_keep["matched_atom_ids"])
        self.assertEqual(filtered_keep["source_routes"], analyzed_keep["source_routes"])
        self.assertEqual([item["atom_id"] for item in filtered_keep["atoms"]], [
            "ATOM-LATE", "ATOM-OVERRIDE", "ATOM-MIDDLE",
        ])
        self.assertTrue(_contains_marker(filtered_keep, "CANDIDATE_METADATA_MARKER"))
        self.assertTrue(_contains_marker(filtered_keep, "CANDIDATE_RAW_MARKER"))
        self.assertTrue(_contains_marker(filtered_keep, "OVERRIDE_METADATA_MARKER"))
        self.assertTrue(_contains_marker(filtered_keep, "OVERRIDE_RAW_MARKER"))

        processed = markdown["candidates"][0]
        content_md = processed["content_md"]
        self.assertNotIn("默认值", content_md)
        self.assertNotIn("不应进入 Markdown", content_md)
        self.assertIn("命中覆盖值 GB", content_md)
        self.assertIn("例外坐席备注", content_md)
        self.assertIn("坐席可见备注", content_md)
        self.assertIn("中间展示 项", content_md)
        self.assertIn("最后展示 次", content_md)
        self.assertLess(content_md.index("例外覆盖"), content_md.index("中间内容"))
        self.assertLess(content_md.index("中间内容"), content_md.index("后置内容"))
        self.assertEqual(processed["matched_atom_ids"], ["ATOM-OVERRIDE", "ATOM-MIDDLE"])
        self.assertEqual(processed["source_routes"], ["keyword", "vector"])
        self.assertTrue(_contains_marker(processed, "CANDIDATE_METADATA_MARKER"))
        self.assertTrue(_contains_marker(processed, "CANDIDATE_RAW_MARKER"))
        self.assertTrue(_contains_marker(processed, "OVERRIDE_METADATA_MARKER"))
        self.assertTrue(_contains_marker(processed, "OVERRIDE_RAW_MARKER"))

        self.assertEqual(analyzed["warnings"], [])
        self.assertEqual(
            [item["code"] for item in filtered["warnings"]], ["atom_not_applicable"]
        )
        self.assertEqual(
            [item["code"] for item in markdown["warnings"]], ["atom_except_overridden"]
        )

        normalized = normalize_knowledge_candidates(raw)
        core_analysis = analyze_candidates(normalized.candidates)
        core_filtered, core_decisions, core_filter_warnings = filter_candidates(
            normalized.candidates,
            normalize_processing_context(context),
        )
        core_processed, core_markdown_warnings = build_markdown_core(
            core_filtered,
            normalize_processing_context(context),
        )
        self.assertEqual(analyzed["analysis"], core_analysis)
        self.assertEqual(
            [item["knowledge_id"] for item in filtered["candidates"]],
            [item.knowledge_id for item in core_filtered],
        )
        self.assertEqual(filtered["decisions"], [item.to_dict() for item in core_decisions])
        self.assertEqual(
            [item["code"] for item in filtered["warnings"]],
            [item.code for item in core_filter_warnings],
        )
        self.assertEqual(
            [item["content_md"] for item in markdown["candidates"]],
            [item.content_md for item in core_processed],
        )
        self.assertEqual(
            [item["code"] for item in markdown["warnings"]],
            [item.code for item in core_markdown_warnings],
        )
        self.assertEqual(raw, original)

    def test_agent_sees_safe_tool_content_while_full_result_stays_in_artifact(self):
        marker = "SYNTHETIC_TOOL_RETURN_PRIVATE_MARKER"
        tool_args = {
            "candidates": [{
                "knowledge_id": "VISIBILITY-TEST",
                "knowledge_name": "模型可见性测试",
                "content": "合成正文",
                "retrieval_rank": 1,
                "retrieval_score": 1.0,
                "applicability": {"status": {"synthetic_marker": marker}},
                "atoms": [],
                "metadata": {"synthetic_marker": marker},
                "raw": {"synthetic_marker": marker},
            }],
            "processing_context": {"audience": "agent"},
        }

        direct = filter_knowledge_candidates.invoke(copy.deepcopy(tool_args))
        self.assertIsInstance(direct, dict)
        self.assertTrue(_contains_marker(direct, marker))
        self.assertIn("candidates", direct)
        self.assertIn("decisions", direct)
        self.assertIn("warnings", direct)

        model = _ToolReturnCaptureModel(tool_args=copy.deepcopy(tool_args))
        agent = create_agent(
            model=model,
            tools=[filter_knowledge_candidates],
            middleware=[],
            name="shared_tool_visibility_test",
        )
        result = agent.invoke({"messages": [HumanMessage(content="执行合成工具测试")]})
        tool_messages = [
            message for message in result["messages"] if isinstance(message, ToolMessage)
        ]

        self.assertEqual(model.call_count, 2)
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(model.captured_content, tool_messages[0].content)
        self.assertNotIn(marker, model.captured_content)
        self.assertNotIn("metadata", model.captured_content)
        self.assertNotIn("raw", model.captured_content)
        self.assertNotIn("decisions", model.captured_content)
        self.assertNotIn("details", model.captured_content)
        content_payload = json.loads(model.captured_content)
        self.assertEqual(content_payload["stage"], "filter")
        self.assertEqual(content_payload["warning_codes"], {"unknown_status": 1})

        self.assertEqual(model.captured_artifact, tool_messages[0].artifact)
        self.assertTrue(_contains_marker(model.captured_artifact, marker))
        self.assertIn("decisions", model.captured_artifact)
        self.assertIn("warnings", model.captured_artifact)

        provider_message = convert_to_openai_messages(tool_messages[0])
        self.assertNotIn(marker, json.dumps(provider_message, ensure_ascii=False))
        self.assertNotIn("artifact", provider_message)

    def test_all_public_tool_messages_separate_safe_content_and_full_artifact(self):
        marker = "SYNTHETIC_ALL_PUBLIC_TOOLS_MARKER"
        candidates = [{
            "knowledge_id": "SAFE-MESSAGE",
            "knowledge_name": "合成返回值测试",
            "content": "可渲染合成正文",
            "retrieval_rank": 1,
            "retrieval_score": 1.0,
            "applicability": {"status": "1"},
            "atoms": [],
            "metadata": {"synthetic_marker": marker},
            "raw": {"synthetic_marker": marker},
        }]
        calls = [
            (analyze_knowledge_candidates, {"candidates": candidates}),
            (
                filter_knowledge_candidates,
                {"candidates": candidates, "processing_context": {"audience": "agent"}},
            ),
            (
                build_knowledge_markdown,
                {"candidates": candidates, "processing_context": {"audience": "agent"}},
            ),
        ]

        for index, (public_tool, args) in enumerate(calls):
            with self.subTest(tool=public_tool.name):
                message = public_tool.invoke({
                    "name": public_tool.name,
                    "args": copy.deepcopy(args),
                    "id": f"synthetic-call-{index}",
                    "type": "tool_call",
                })
                self.assertIsInstance(message, ToolMessage)
                self.assertNotIn(marker, str(message.content))
                self.assertNotIn("metadata", str(message.content))
                self.assertNotIn("raw", str(message.content))
                self.assertTrue(_contains_marker(message.artifact, marker))

    def test_public_tools_match_fixed_processing_orchestrator_artifacts(self):
        raw = make_top100_candidates()
        original = copy.deepcopy(raw)
        context = {
            "region_id": "0755",
            "region_name": "深圳",
            "channel_code": "10086",
            "request_time": "2026-09-01T14:00:00+08:00",
            "audience": "agent",
        }
        filtered = filter_knowledge_candidates.invoke({
            "candidates": raw,
            "processing_context": context,
        })
        markdown = build_knowledge_markdown.invoke({
            "candidates": filtered["candidates"],
            "processing_context": context,
        })

        ws = RunWorkspace(query="5G流量套餐")
        ws.data.update({
            "processing_context": copy.deepcopy(context),
            "retrieval_query": "5G 流量套餐",
            "knowledge_candidates": raw,
        })
        set_workspace(ws)
        top3 = asyncio.run(KnowledgeProcessingOrchestrator(ScriptedChatModel()).run())

        self.assertEqual(
            [item["knowledge_id"] for item in filtered["candidates"]],
            [item.knowledge_id for item in ws.data["filtered_knowledge_candidates"]],
        )
        self.assertEqual(
            {
                item["knowledge_id"]: item["content_md"]
                for item in markdown["candidates"]
            },
            {
                item.knowledge_id: item.content_md
                for item in ws.data["processed_knowledge_candidates"]
            },
        )
        self.assertEqual(
            [item.knowledge_id for item in top3],
            ["REAL-KNOWLEDGE-042", "REAL-KNOWLEDGE-073", "REAL-KNOWLEDGE-099"],
        )
        self.assertFalse(ws.data["processing_meta"].degraded)
        self.assertEqual(ws.data["processing_meta"].stage_order, [
            "analyze", "filter", "build_markdown", "rerank",
        ])
        self.assertEqual(raw, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
