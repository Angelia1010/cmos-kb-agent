#!/usr/bin/env python
"""Processing 第一版固定流水线的本地、离线终端演示。"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Sequence

# 支持从 universal-agent 目录执行 ``python scripts/run_processing_demo.py``。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from langchain_core.messages import AIMessage  # noqa: E402

from kbagent.processing.agent import KnowledgeProcessingOrchestrator  # noqa: E402
from kbagent.scripted_model import ScriptedChatModel  # noqa: E402
from kbagent.shared.knowledge_processing.models import (  # noqa: E402
    KnowledgeProcessingOptions,
    ProcessedKnowledge,
)
from kbagent.shared.workspace import RunWorkspace, set_workspace  # noqa: E402
from tests.processing_mock_data import make_top100_candidates  # noqa: E402


LOGGER_NAME = "processing_demo"
PROMPT_BEGIN = "RERANK_INPUT_BEGIN\n"
PROMPT_END = "\nRERANK_INPUT_END"
PROMPT_TOP_LEVEL_FIELDS = {"query", "context", "retrieval_query", "top_k", "candidates"}
PROMPT_CONTEXT_FIELDS = {
    "region_id", "region_name", "channel_code", "request_time", "audience", "customer_type",
}
PROMPT_CANDIDATE_FIELDS = {"evidence_id", "title", "content_md"}
FORBIDDEN_PROMPT_FIELDS = {"knowledge_id", "evidence_map", "raw", "metadata", "attributes"}
OUTPUT_FILENAMES = {
    "01_mock_input.json", "02_normalized_candidates.json", "03_filtered_candidates.json",
    "04_processed_candidates.json", "05_sample_content.md", "06_rerank_prompt.json",
    "07_top3_result.json", "processing_demo.log",
}
DEFAULT_QUERY = "5G流量套餐，59元100GB"
DEFAULT_RETRIEVAL_QUERY = "5G 套餐 59元 100GB"
DEFAULT_PROCESSING_CONTEXT = {
    "region_id": "0755",
    "region_name": "深圳",
    "channel_code": "10086",
    "request_time": "2026-08-28T10:00:00+08:00",
    "audience": "agent",
}
PROCESSING_CONTEXT_FIELDS = (
    "region_id", "region_name", "channel_code", "request_time", "audience", "customer_type",
)


@dataclass
class DemoConfig:
    count: int = 100
    input_json: Path | None = None
    model_mode: str = "scripted"
    simulate: str = "normal"
    output_dir: Path | None = None
    verbose: bool = False
    show_markdown: bool = False
    query: str | None = None
    retrieval_query: str | None = None
    region_id: str | None = None
    region_name: str | None = None
    channel_code: str | None = None
    request_time: str | None = None
    audience: str | None = None
    customer_type: str | None = None


@dataclass
class DemoRunResult:
    workspace: RunWorkspace
    prompt_calls: list[dict[str, Any]]
    input_unchanged: bool
    elapsed_ms: int
    output_dir: Path | None


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def _configure_logging(output_dir: Path | None, verbose: bool) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            output_dir / "processing_demo.log", mode="w", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"--input-json 中的 {field_name} 必须是字符串或 null")
    return value


def _load_candidates(config: DemoConfig) -> tuple[list[Any], str, dict[str, Any]]:
    if config.input_json is None:
        return make_top100_candidates(config.count), f"mock:{config.count}", {}
    input_path = config.input_json.expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    runtime_input: dict[str, Any] = {}
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        candidates = payload["candidates"]
        for field_name in ("query", "retrieval_query"):
            if field_name in payload:
                runtime_input[field_name] = _optional_text(payload[field_name], field_name)
        raw_context = payload.get("processing_context")
        if raw_context is not None and not isinstance(raw_context, dict):
            raise ValueError("--input-json 中的 processing_context 必须是 JSON 对象或 null")
        if isinstance(raw_context, dict):
            runtime_input["processing_context"] = {
                field_name: _optional_text(
                    raw_context[field_name], f"processing_context.{field_name}"
                )
                for field_name in PROCESSING_CONTEXT_FIELDS
                if field_name in raw_context
            }
    else:
        raise ValueError("--input-json 必须是候选列表，或包含 candidates 列表的 JSON 对象")
    return copy.deepcopy(candidates), f"input_json:{input_path}", copy.deepcopy(runtime_input)


def _resolve_runtime_input(
    config: DemoConfig,
    json_runtime_input: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    query = config.query
    if query is None:
        json_query = json_runtime_input.get("query")
        query = json_query if json_query is not None else DEFAULT_QUERY
    retrieval_query = config.retrieval_query
    if retrieval_query is None:
        json_retrieval_query = json_runtime_input.get("retrieval_query")
        retrieval_query = (
            json_retrieval_query
            if json_retrieval_query is not None
            else DEFAULT_RETRIEVAL_QUERY
        )

    context = copy.deepcopy(DEFAULT_PROCESSING_CONTEXT)
    json_context = json_runtime_input.get("processing_context") or {}
    for field_name in PROCESSING_CONTEXT_FIELDS:
        json_value = json_context.get(field_name)
        if json_value is not None:
            context[field_name] = json_value
        cli_value = getattr(config, field_name)
        if cli_value is not None:
            context[field_name] = cli_value
    return query, retrieval_query, context


def _extract_prompt_payload(messages: Sequence[Any]) -> dict[str, Any]:
    user_text = str(getattr(messages[-1], "content", "")) if messages else ""
    if PROMPT_BEGIN not in user_text or PROMPT_END not in user_text:
        raise ValueError("无法从重排消息中提取 RERANK_INPUT")
    serialized = user_text.split(PROMPT_BEGIN, 1)[1].split(PROMPT_END, 1)[0]
    payload = json.loads(serialized)
    if not isinstance(payload, dict):
        raise ValueError("重排输入不是 JSON 对象")
    return payload


def _message_stage(messages: Sequence[Any]) -> str:
    text = "\n".join(str(getattr(message, "content", "")) for message in messages)
    return "global" if "[TASK:rerank_global]" in text else "batch"


class DemoScriptedModel:
    """捕获实际 Prompt，并在 Demo 层非侵入式模拟模型故障。"""

    def __init__(self, simulate: str, timeout_delay: float = 1.0) -> None:
        self.delegate = ScriptedChatModel()
        self.simulate = simulate
        self.timeout_delay = timeout_delay
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> AIMessage:
        payload = _extract_prompt_payload(messages)
        stage = _message_stage(messages)
        record = {
            "call_index": len(self.calls) + 1,
            "stage": stage,
            "payload": copy.deepcopy(payload),
            "elapsed_ms": 0,
        }
        self.calls.append(record)
        started = time.perf_counter()
        try:
            if self.simulate == "timeout":
                await asyncio.sleep(self.timeout_delay)
                return AIMessage(content='{"ranked_ids": []}')
            if self.simulate == "invalid_json":
                return AIMessage(content="这不是合法 JSON")
            if self.simulate == "insufficient_results":
                candidates = payload.get("candidates") or []
                expected = max(0, int(payload.get("top_k") or 0))
                keep = max(0, expected - 1)
                ids = [str(item.get("evidence_id")) for item in candidates[:keep]]
                return AIMessage(content=json.dumps({"ranked_ids": ids}, ensure_ascii=False))
            return await self.delegate.ainvoke(messages, **kwargs)
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)


def _build_model(config: DemoConfig, timeout_seconds: float) -> DemoScriptedModel:
    if config.model_mode != "scripted":
        # 后续 real 模式应在此接入项目模型工厂，不在 Demo 中读取配置或 Key。
        raise ValueError("第一版 Demo 仅支持 --model-mode scripted")
    return DemoScriptedModel(
        simulate=config.simulate,
        timeout_delay=max(0.2, timeout_seconds * 10),
    )


def _assert_prompt_whitelist(calls: Sequence[dict[str, Any]]) -> None:
    if not calls:
        return  # 有效候选不超过 final_top_k 时，真实重排入口不会调用模型。

    def walk_keys(value: Any) -> set[str]:
        keys: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(str(key))
                keys.update(walk_keys(item))
        elif isinstance(value, list):
            for item in value:
                keys.update(walk_keys(item))
        return keys

    for call in calls:
        payload = call["payload"]
        if set(payload) != PROMPT_TOP_LEVEL_FIELDS:
            raise RuntimeError(f"重排 Prompt 顶层字段不符合白名单: {sorted(payload)}")
        if set(payload.get("context") or {}) != PROMPT_CONTEXT_FIELDS:
            raise RuntimeError("重排 Prompt context 字段不符合白名单")
        candidates = payload.get("candidates") or []
        if any(set(candidate) != PROMPT_CANDIDATE_FIELDS for candidate in candidates):
            raise RuntimeError("重排 Prompt candidate 字段不符合白名单")
        leaked_keys = walk_keys(payload) & FORBIDDEN_PROMPT_FIELDS
        if leaked_keys:
            raise RuntimeError(f"重排 Prompt 包含禁止的结构化字段: {sorted(leaked_keys)}")


def _raw_summary(candidate: Any, include_content_preview: bool = True) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        summary = {"type": type(candidate).__name__}
        if include_content_preview:
            summary["summary"] = str(candidate)[:160]
        return summary
    atoms = candidate.get("knowledgeAtoms", candidate.get("atoms", []))
    content = candidate.get("content", candidate.get("text", ""))
    summary = {
        "knowledge_id": candidate.get("knowledgeId", candidate.get("knowledge_id")),
        "knowledge_name": candidate.get("knowledgeName", candidate.get("knowledge_name")),
        "retrieval_rank": candidate.get("retrievalRank", candidate.get("retrieval_rank")),
        "status": candidate.get("status"),
        "atom_count": len(atoms) if isinstance(atoms, list) else None,
        "content_type": type(content).__name__,
    }
    if include_content_preview:
        summary["content_preview"] = str(content)[:120]
    return summary


def _normalized_summary(candidate: Any) -> dict[str, Any]:
    return {
        "knowledge_id": candidate.knowledge_id,
        "name": candidate.name,
        "retrieval_rank": candidate.retrieval_rank,
        "retrieval_score": candidate.retrieval_score,
        "status": candidate.applicability.status,
        "regions": candidate.applicability.regions,
        "region_ids": candidate.applicability.region_ids,
        "channel_codes": candidate.applicability.channel_codes,
        "atom_count": len(candidate.atoms),
    }


def _find_except_snapshot(ws: RunWorkspace) -> dict[str, Any] | None:
    processed_by_id = {
        item.knowledge_id: item for item in ws.data.get("processed_knowledge_candidates", [])
    }
    for candidate in ws.data.get("filtered_knowledge_candidates", []):
        processed = processed_by_id.get(candidate.knowledge_id)
        if processed is None:
            continue
        processed_atoms = {atom.atom_id: atom for atom in processed.atoms}
        for atom in candidate.atoms:
            if not atom.except_rules:
                continue
            final_atom = processed_atoms.get(atom.atom_id)
            if final_atom is not None and final_atom.content != atom.content:
                context = ws.data.get("processing_context", {})
                return {
                    "knowledge_id": candidate.knowledge_id,
                    "atom_id": atom.atom_id,
                    "current_region_id": context.get("region_id"),
                    "default_content": _jsonable(atom.content),
                    "overridden_content": _jsonable(final_atom.content),
                    "overridden_annotation": _jsonable(final_atom.annotation),
                }
    return None


def _reason_counts(decisions: Sequence[Any]) -> Counter[str]:
    return Counter(
        reason
        for decision in decisions
        if not decision.accepted
        for reason in decision.reasons
    )


def _warning_counts(warnings: Sequence[Any]) -> Counter[str]:
    return Counter(str(warning.code) for warning in warnings)


def _completion_times(ws: RunWorkspace) -> dict[str, int]:
    return {
        event.event: event.ts_ms
        for event in ws.tracer.events
        if event.stage == "processing.knowledge"
        and event.event in {"analyze", "filter", "build_markdown", "rerank"}
    }


def _top3_rows(top3: Sequence[ProcessedKnowledge]) -> list[dict[str, Any]]:
    return [{
        "rerank_rank": item.rerank_rank,
        "knowledge_id": item.knowledge_id,
        "knowledge_name": item.name,
        "retrieval_rank": item.retrieval_rank,
        "retrieval_score": item.retrieval_score,
        "rerank_score": getattr(item, "rerank_score", None),
        "rerank_reason": getattr(item, "rerank_reason", None) or "当前接口未提供候选级原因",
        "content_md_length": len(item.content_md),
    } for item in top3]


def _log_run_report(
    logger: logging.Logger,
    ws: RunWorkspace,
    model: DemoScriptedModel,
    raw_candidates: Sequence[Any],
    input_elapsed_ms: int,
    show_markdown: bool,
    redact_candidate_content: bool = False,
) -> tuple[str, int]:
    normalized = ws.data.get("normalized_knowledge_candidates", [])
    filtered = ws.data.get("filtered_knowledge_candidates", [])
    processed = ws.data.get("processed_knowledge_candidates", [])
    warnings = ws.data.get("processing_warnings", [])
    decisions = ws.data.get("knowledge_filter_reasons", [])
    analysis = ws.data.get("knowledge_candidate_analysis", {})
    details = ws.data.get("rerank_details", {})
    global_detail = details.get("global", {})
    completion = _completion_times(ws)
    analyze_ms = max(0, completion.get("analyze", ws.tracer.started_ms) - ws.tracer.started_ms)
    filter_ms = max(0, completion.get("filter", 0) - completion.get("analyze", 0))
    markdown_ms = max(0, completion.get("build_markdown", 0) - completion.get("filter", 0))
    rerank_ms = max(0, completion.get("rerank", 0) - completion.get("build_markdown", 0))
    filter_warning_codes = {
        "unknown_status", "invalid_start_time", "invalid_end_time", "invalid_applicability_field",
    }
    filter_warning_count = sum(warning.code in filter_warning_codes for warning in warnings)
    pre_rerank_warning_count = sum(not warning.code.startswith("rerank_") for warning in warnings)

    logger.info(
        "[1/7] 初始化输入数据和RunWorkspace | 输入=%d | 耗时=%dms",
        len(raw_candidates), input_elapsed_ms,
    )
    if raw_candidates:
        logger.info(
            "原始候选摘要 | %s",
            json.dumps(
                _raw_summary(raw_candidates[0], not redact_candidate_content),
                ensure_ascii=False,
            ),
        )
        if not redact_candidate_content:
            logger.debug("原始候选代表样例 | %s", json.dumps(_jsonable(raw_candidates[0]), ensure_ascii=False))

    logger.info(
        "[2/7] 规范化候选 | 输入=%d | 输出=%d | 变化/告警=%d | 耗时=%dms（与分析由同一真实Tool完成）",
        len(raw_candidates), len(normalized), len(ws.data.get("adapter_warnings", [])), analyze_ms,
    )
    if normalized:
        logger.info("规范化候选摘要 | %s", json.dumps(_normalized_summary(normalized[0]), ensure_ascii=False))
        if not redact_candidate_content:
            logger.debug("规范化候选代表样例 | %s", json.dumps(_jsonable(normalized[0]), ensure_ascii=False))

    logger.info(
        "[3/7] 分析候选 | 候选=%s | 原子=%s | HTML=%s | 表格=%s | Adapter告警=%d | 耗时包含于上一步",
        analysis.get("candidate_count"), analysis.get("atom_count"),
        analysis.get("html_candidate_count"), analysis.get("table_candidate_count"),
        len(ws.data.get("adapter_warnings", [])),
    )
    logger.debug("完整分析结果 | %s", json.dumps(_jsonable(analysis), ensure_ascii=False))

    reasons = _reason_counts(decisions)
    except_snapshot = _find_except_snapshot(ws)
    logger.info(
        "[4/7] 适用性过滤和地区例外处理 | 输入=%d | 过滤后=%d | 过滤=%d | 告警=%d | 原因=%s | 耗时=%dms",
        len(normalized), len(filtered), len(normalized) - len(filtered),
        filter_warning_count, dict(reasons), filter_ms,
    )
    if except_snapshot:
        logged_snapshot = except_snapshot
        if redact_candidate_content:
            logged_snapshot = {
                key: except_snapshot[key]
                for key in ("knowledge_id", "atom_id", "current_region_id")
            }
        logger.info("地区except命中 | %s", json.dumps(logged_snapshot, ensure_ascii=False))
    else:
        logger.info("地区except命中 | 当前输入和上下文未产生正文覆盖")

    sample = next(
        (item for item in processed if except_snapshot and item.knowledge_id == except_snapshot["knowledge_id"]),
        processed[0] if processed else None,
    )
    sample_markdown = sample.content_md if sample is not None else ""
    logger.info(
        "[5/7] 生成Markdown | 输入=%d | 输出=%d | 空内容剔除=%d | 告警=%d | 耗时=%dms",
        len(filtered), len(processed), len(filtered) - len(processed),
        pre_rerank_warning_count, markdown_ms,
    )
    if sample is not None:
        if redact_candidate_content:
            logger.info("Markdown样例 | 标题=%s | 长度=%d | 正文已隐藏", sample.name, len(sample_markdown))
        elif show_markdown:
            logger.info("完整content_md (%s, %d字符)\n%s", sample.name, len(sample_markdown), sample_markdown)
        else:
            logger.info(
                "Markdown样例 | 标题=%s | 长度=%d | 摘要=%s",
                sample.name, len(sample_markdown), sample_markdown.replace("\n", " ")[:160],
            )

    prompt_sample = model.calls[0]["payload"] if model.calls else {}
    prompt_candidates = prompt_sample.get("candidates") or []
    prompt_summary = {
        "stage": model.calls[0]["stage"] if model.calls else None,
        "top_level_fields": list(prompt_sample),
        "context_fields": list((prompt_sample.get("context") or {}).keys()),
        "candidate_fields": list(prompt_candidates[0]) if prompt_candidates else [],
        "top_k": prompt_sample.get("top_k"),
        "candidate_count": len(prompt_candidates),
        "first_evidence_id": prompt_candidates[0].get("evidence_id") if prompt_candidates else None,
        "query": prompt_sample.get("query"),
        "sample_title": prompt_candidates[0].get("title") if prompt_candidates else None,
    }
    if not redact_candidate_content:
        prompt_summary["sample_content_md"] = (
            str(prompt_candidates[0].get("content_md"))[:200] if prompt_candidates else None
        )
    if model.calls:
        logger.info("ScriptedModel实际Prompt摘要 | %s", json.dumps(prompt_summary, ensure_ascii=False))
    else:
        logger.info("ScriptedModel实际Prompt摘要 | 有效候选不足以触发模型调用")
    for batch in details.get("batches", []):
        logger.info(
            "批内重排 #%d | 输入=%d | 模型=%s | 最终选出=%s | 降级原因=%s",
            batch["batch_index"], len(batch["input_ids"]), batch["model_ids"],
            batch["selected_ids"], batch["fallback_reasons"],
        )
    logger.info(
        "全局复排 | 池大小=%d | 模型编号=%s | 最终编号=%s",
        len(global_detail.get("pool_ids", [])), global_detail.get("model_ids", []),
        global_detail.get("selected_ids", []),
    )
    mapping = ws.data.get("rerank_evidence_map", {})
    selected_mapping = {
        evidence_id: mapping.get(evidence_id)
        for evidence_id in global_detail.get("selected_ids", [])
    }
    logger.info("E编号映射回真实knowledge_id | %s", json.dumps(selected_mapping, ensure_ascii=False))
    logger.info(
        "[6/7] 分批重排和全局重排 | 批次=%d | mode=%s | degraded=%s | fallback_count=%s | 原因=%s | 耗时=%dms",
        len(details.get("batches", [])), global_detail.get("mode"),
        global_detail.get("degraded"), global_detail.get("fallback_count"),
        details.get("fallback_reasons", []), rerank_ms,
    )

    warning_counts = _warning_counts(warnings)
    if warning_counts:
        logger.warning("处理告警汇总（不含正文） | %s", dict(warning_counts))
    top3 = ws.data.get("top3_candidates", [])
    logger.info(
        "[7/7] 输出Top3并写回RunWorkspace | Top3=%d | Workspace字段=%s",
        len(top3), sorted(ws.data),
    )
    for row in _top3_rows(top3):
        logger.info("Top3 | %s", json.dumps(row, ensure_ascii=False))
    return sample_markdown, rerank_ms


def _write_outputs(
    output_dir: Path,
    source: str,
    raw_candidates: Sequence[Any],
    ws: RunWorkspace,
    model: DemoScriptedModel,
    sample_markdown: str,
    rerank_elapsed_ms: int,
    config: DemoConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "01_mock_input.json", {
        "source": source,
        "count": len(raw_candidates),
        "candidates": raw_candidates,
    })
    _write_json(output_dir / "02_normalized_candidates.json", ws.data.get("normalized_knowledge_candidates", []))
    _write_json(output_dir / "03_filtered_candidates.json", {
        "candidates": ws.data.get("filtered_knowledge_candidates", []),
        "decisions": ws.data.get("knowledge_filter_reasons", []),
        "warnings": ws.data.get("processing_warnings", []),
    })
    _write_json(output_dir / "04_processed_candidates.json", ws.data.get("processed_knowledge_candidates", []))
    (output_dir / "05_sample_content.md").write_text(
        sample_markdown or "<!-- 当前输入没有可输出的 Markdown 候选 -->\n", encoding="utf-8"
    )
    _write_json(output_dir / "06_rerank_prompt.json", {
        "model_mode": config.model_mode,
        "simulate": config.simulate,
        "calls": model.calls,
    })
    details = ws.data.get("rerank_details", {})
    global_detail = details.get("global", {})
    include_candidate_details = config.input_json is None
    _write_json(output_dir / "07_top3_result.json", {
        "runtime_input": {
            "query": ws.query,
            "retrieval_query": ws.data.get("retrieval_query"),
            "processing_context": ws.data.get("processing_context"),
        },
        "top3": _top3_rows(ws.data.get("top3_candidates", [])),
        "candidates": ws.data.get("top3_candidates", []) if include_candidate_details else [],
        "candidate_details_included": include_candidate_details,
        "rerank_metadata": {
            "mode": global_detail.get("mode"),
            "degraded": global_detail.get("degraded"),
            "fallback_count": global_detail.get("fallback_count"),
            "degradation_reasons": details.get("fallback_reasons", []),
            "rerank_elapsed_ms": rerank_elapsed_ms,
        },
        "processing_meta": ws.data.get("processing_meta"),
        "workspace_keys": sorted(ws.data),
    })


async def run_demo(config: DemoConfig) -> DemoRunResult:
    """调用一次真实 KnowledgeProcessingOrchestrator.run() 并观察其产物。"""
    output_dir = config.output_dir.expanduser().resolve() if config.output_dir else None
    input_path = config.input_json.expanduser().resolve() if config.input_json else None
    if (
        input_path is not None
        and output_dir is not None
        and input_path.parent == output_dir
        and input_path.name in OUTPUT_FILENAMES
    ):
        raise ValueError("--output-dir 会覆盖 --input-json，请使用不同目录")
    input_file_snapshot = input_path.read_bytes() if input_path is not None else None
    logger = _configure_logging(output_dir, config.verbose)
    started = time.perf_counter()
    raw_candidates, source, json_runtime_input = _load_candidates(config)
    input_snapshot = copy.deepcopy(raw_candidates)
    input_elapsed_ms = round((time.perf_counter() - started) * 1000)

    query, retrieval_query, context = _resolve_runtime_input(config, json_runtime_input)
    ws = RunWorkspace(query=query)
    ws.data.update({
        "processing_context": context,
        "retrieval_query": retrieval_query,
        "knowledge_candidates": raw_candidates,
    })
    set_workspace(ws)
    timeout_seconds = 0.05 if config.simulate == "timeout" else 2.0
    options = KnowledgeProcessingOptions(rerank_timeout_seconds=timeout_seconds)
    model = _build_model(config, timeout_seconds)

    logger.info("Demo数据源=%s | model_mode=%s | simulate=%s", source, config.model_mode, config.simulate)
    logger.info(
        "最终请求参数 | %s",
        json.dumps({
            "query": query,
            "retrieval_query": retrieval_query,
            "processing_context": context,
        }, ensure_ascii=False),
    )
    pipeline_started = time.perf_counter()
    top3 = await KnowledgeProcessingOrchestrator(model=model, options=options).run()
    elapsed_ms = round((time.perf_counter() - pipeline_started) * 1000)
    input_unchanged = raw_candidates == input_snapshot
    if not input_unchanged:
        raise RuntimeError("真实 Processing 入口修改了原始输入候选")
    if ws.data.get("top3_candidates") != top3:
        raise RuntimeError("Top3 未正确写回 RunWorkspace")
    _assert_prompt_whitelist(model.calls)

    sample_markdown, rerank_elapsed_ms = _log_run_report(
        logger, ws, model, raw_candidates, input_elapsed_ms, config.show_markdown,
        redact_candidate_content=input_path is not None,
    )
    if output_dir is not None:
        _write_outputs(
            output_dir, source, raw_candidates, ws, model,
            sample_markdown, rerank_elapsed_ms, config,
        )
        logger.info("输出文件已写入 %s", output_dir)
    if input_path is not None and input_path.read_bytes() != input_file_snapshot:
        raise RuntimeError("--input-json 文件内容发生变化")
    logger.info("Demo完成 | 总耗时=%dms | 输入未原地修改=%s", elapsed_ms, input_unchanged)
    for handler in logger.handlers:
        handler.flush()
    return DemoRunResult(ws, model.calls, input_unchanged, elapsed_ms, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线运行 Processing 第一版固定流水线 Demo")
    parser.add_argument("--count", type=int, default=100, help="Mock 候选数量，默认 100")
    parser.add_argument("--input-json", type=Path, help="候选列表或包含 candidates 的 UTF-8 JSON")
    parser.add_argument("--model-mode", choices=("scripted",), default="scripted")
    parser.add_argument(
        "--simulate",
        choices=("normal", "timeout", "invalid_json", "insufficient_results"),
        default="normal",
    )
    parser.add_argument("--output-dir", type=Path, help="写出阶段快照和 processing_demo.log")
    parser.add_argument("--verbose", action="store_true", help="显示代表性 DEBUG 信息")
    parser.add_argument("--show-markdown", action="store_true", help="在终端显示一条完整 Markdown")
    parser.add_argument("--query", help="用户原始问题；优先于 input JSON")
    parser.add_argument("--retrieval-query", help="检索问题；优先于 input JSON")
    parser.add_argument("--region-id", help="请求地区 ID；优先于 input JSON")
    parser.add_argument("--region-name", help="请求地区名称；优先于 input JSON")
    parser.add_argument("--channel-code", help="请求渠道编码；优先于 input JSON")
    parser.add_argument("--request-time", help="请求时间；优先于 input JSON")
    parser.add_argument("--audience", help="目标受众；优先于 input JSON")
    parser.add_argument("--customer-type", help="客户类型；优先于 input JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_json is None and args.count < 1:
        parser.error("--count 必须大于 0")
    config = DemoConfig(
        count=args.count,
        input_json=args.input_json,
        model_mode=args.model_mode,
        simulate=args.simulate,
        output_dir=args.output_dir,
        verbose=args.verbose,
        show_markdown=args.show_markdown,
        query=args.query,
        retrieval_query=args.retrieval_query,
        region_id=args.region_id,
        region_name=args.region_name,
        channel_code=args.channel_code,
        request_time=args.request_time,
        audience=args.audience,
        customer_type=args.customer_type,
    )
    try:
        asyncio.run(run_demo(config))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        logging.getLogger(LOGGER_NAME).error("Demo无法继续执行: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
