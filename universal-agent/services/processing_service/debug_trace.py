"""Processing 调试请求的内存 Trace 采集、脱敏和大小限制。"""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence


MAX_TRACE_RESPONSE_BYTES = 4_000_000
# 全局重排 Prompt 生产预算为 30,000 字符；单字段上限略高于该值，
# 以便正常请求可查看完整实际 Prompt，异常大输入仍会被截断。
MAX_TRACE_FIELD_CHARS = 40_000
MAX_TRACE_LIST_ITEMS = 100
MAX_TRACE_DEPTH = 12
MAX_SNAPSHOT_TEXT_CHARS = 350_000
MAX_EXPORT_TEXT_CHARS = 1_500_000

_SENSITIVE_KEY_RE = re.compile(
    r"(?:authorization|api[_-]?key|access[_-]?token|debug[_-]?token|password|"
    r"secret|private[_-]?key|credential|endpoint|base[_-]?url|model[_-]?url|"
    r"environment|env(?:iron)?|local[_-]?path)$",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+")
_API_KEY_VALUE_RE = re.compile(r"(?i)\b(?:sk|ak|api)[-_][A-Za-z0-9_-]{8,}\b")
_URL_RE = re.compile(r"(?i)\bhttps?://[^\s\"'<>]+")
_WINDOWS_PATH_RE = re.compile(r"(?i)\b[A-Z]:\\[^\r\n\"'<>]+")
_UNIX_PATH_RE = re.compile(r"/(?:home|Users|tmp|var|etc|opt|workspace)/[^\s\"'<>]+")
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
_ID_RE = re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3}[0-9Xx])(?!\d)")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _redact_text(value: str) -> str:
    text = _BEARER_RE.sub("[REDACTED_AUTH]", value)
    text = _API_KEY_VALUE_RE.sub("[REDACTED_KEY]", text)
    text = _URL_RE.sub("[REDACTED_URL]", text)
    text = _WINDOWS_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _UNIX_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _PHONE_RE.sub(lambda match: f"{match.group(1)}****{match.group(3)}", text)
    text = _ID_RE.sub(lambda match: f"{match.group(1)[:3]}***********{match.group(3)[-4:]}", text)
    return text


class _Sanitizer:
    def __init__(self, text_budget: int) -> None:
        self.remaining_text = max(0, text_budget)
        self.truncated = False

    def clean(self, value: Any, *, depth: int = 0) -> Any:
        if depth > MAX_TRACE_DEPTH:
            self.truncated = True
            return "[TRACE_DEPTH_LIMIT]"
        value = _jsonable(value)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            text = _redact_text(value)
            allowed = min(MAX_TRACE_FIELD_CHARS, self.remaining_text)
            if len(text) > allowed:
                self.truncated = True
                if allowed <= 0:
                    return "[TRACE_TEXT_BUDGET_EXHAUSTED]"
                suffix = f"…[trace字段已截断, 原始字符数={len(text)}]"
                text = text[:max(0, allowed - len(suffix))] + suffix
            self.remaining_text = max(0, self.remaining_text - len(text))
            return text
        if isinstance(value, list):
            rows = value[:MAX_TRACE_LIST_ITEMS]
            result = [self.clean(item, depth=depth + 1) for item in rows]
            if len(value) > len(rows):
                self.truncated = True
                result.append({
                    "_trace_omitted_items": len(value) - len(rows),
                    "_trace_limit": MAX_TRACE_LIST_ITEMS,
                })
            return result
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if _SENSITIVE_KEY_RE.search(key_text):
                    result[key_text] = "[REDACTED]"
                else:
                    result[key_text] = self.clean(item, depth=depth + 1)
            return result
        return self.clean(str(value), depth=depth + 1)


def _safe_snapshot(value: Any, *, text_budget: int = MAX_SNAPSHOT_TEXT_CHARS) -> Any:
    sanitizer = _Sanitizer(text_budget)
    result = sanitizer.clean(value)
    if isinstance(result, dict) and sanitizer.truncated:
        result.setdefault("_trace_truncated", True)
    return result


def _warnings(data: dict[str, Any]) -> Any:
    # 内部 warning.message 可能拼入模型异常文本；调试 Trace 只展示可定位字段，
    # 避免把网关地址、凭证片段或完整异常通过诊断页面带出。
    rows = []
    for warning in data.get("processing_warnings", []):
        item = _jsonable(warning)
        if not isinstance(item, dict):
            continue
        rows.append({
            "code": item.get("code", "unknown_warning"),
            "message": "Processing 产生告警，请根据 code 和 field 排查",
            "source_index": item.get("source_index"),
            "knowledge_id": item.get("knowledge_id"),
            "field": item.get("field"),
        })
    return rows


class _TracingModel:
    """透明代理模型；一次真实调用对应一次记录，不重试也不补调。"""

    def __init__(self, delegate: Any, collector: "ProcessingTraceCollector") -> None:
        self._delegate = delegate
        self._collector = collector

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = await self._delegate.ainvoke(messages, **kwargs)
        except Exception as exc:
            self._collector.record_model_call(
                messages,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error_type=type(exc).__name__,
            )
            raise
        self._collector.record_model_call(
            messages,
            response=response,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        return response


class ProcessingTraceCollector:
    """仅供 `/process/ui/run` 使用的单请求内存采集器。"""

    def __init__(self, request_payload: Any) -> None:
        self.started = time.perf_counter()
        self.request = _safe_snapshot(request_payload)
        self.stages: dict[str, dict[str, Any]] = {}
        self.model_calls: list[dict[str, Any]] = []
        self.final: dict[str, Any] = {}
        self.collection_errors: list[str] = []
        self._stage_started: dict[str, float] = {}

    def wrap_model(self, model: Any) -> Any:
        return _TracingModel(model, self)

    @staticmethod
    def _stage_name(tool_name: str) -> str:
        return {
            "analyze_knowledge_candidates": "analyze",
            "filter_knowledge_candidates": "filter",
            "build_knowledge_markdown": "build_markdown",
            "rerank_knowledge_candidates": "rerank",
        }.get(tool_name, tool_name)

    def before_stage(self, tool_name: str, workspace: Any) -> None:
        stage = self._stage_name(tool_name)
        self._stage_started[stage] = time.perf_counter()
        data = workspace.data
        if stage == "analyze":
            stage_input = {
                "chunks": data.get("chunks", []),
                "knowledge_candidates": data.get("knowledge_candidates", []),
            }
        elif stage == "filter":
            stage_input = {
                "processing_context": data.get("processing_context", {}),
                "normalized_candidates": data.get("normalized_knowledge_candidates", []),
            }
        elif stage == "build_markdown":
            stage_input = {
                "filtered_candidates": data.get("filtered_knowledge_candidates", []),
            }
        else:
            stage_input = {
                "processed_candidates": data.get("processed_knowledge_candidates", []),
            }
        self.stages[stage] = {"input": _safe_snapshot(stage_input)}

    def after_stage(
        self,
        tool_name: str,
        workspace: Any,
        *,
        error_type: str | None = None,
    ) -> None:
        stage = self._stage_name(tool_name)
        started = self._stage_started.pop(stage, time.perf_counter())
        record = self.stages.setdefault(stage, {})
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if error_type:
            record["error"] = {"type": error_type, "message": "Processing阶段执行失败"}
            return
        data = workspace.data
        if stage == "analyze":
            output = {
                "normalized_candidates": data.get("normalized_knowledge_candidates", []),
                "analysis": data.get("knowledge_candidate_analysis", {}),
                "warnings": _warnings(data),
                "processing_meta": data.get("processing_meta"),
            }
        elif stage == "filter":
            decisions = data.get("knowledge_filter_reasons", [])
            output = {
                "kept_candidates": data.get("filtered_knowledge_candidates", []),
                "kept_decisions": [
                    item for item in decisions if getattr(item, "accepted", False)
                ],
                "filtered_decisions": [
                    item for item in decisions if not getattr(item, "accepted", False)
                ],
                "warnings": _warnings(data),
                "processing_meta": data.get("processing_meta"),
            }
        elif stage == "build_markdown":
            output = {
                "processed_candidates": data.get("processed_knowledge_candidates", []),
                "content_comparison": self._markdown_comparison(data),
                "warnings": _warnings(data),
                "processing_meta": data.get("processing_meta"),
            }
        else:
            output = {
                "evidence_map": data.get("rerank_evidence_map", {}),
                "rerank_details": data.get("rerank_details", {}),
                "top3_candidates": data.get("top3_candidates", []),
                "warnings": _warnings(data),
                "processing_meta": data.get("processing_meta"),
            }
        record["output"] = _safe_snapshot(output)

    @staticmethod
    def _markdown_comparison(data: dict[str, Any]) -> list[dict[str, Any]]:
        filtered = {
            (
                str(getattr(item, "knowledge_id", "")),
                str(getattr(item, "chunk_id", "")),
            ): item
            for item in data.get("filtered_knowledge_candidates", [])
        }
        rows = []
        for item in data.get("processed_knowledge_candidates", []):
            knowledge_id = str(getattr(item, "knowledge_id", ""))
            source = filtered.get((knowledge_id, str(getattr(item, "chunk_id", ""))))
            rows.append({
                "knowledge_id": knowledge_id,
                "chunk_id": getattr(item, "chunk_id", ""),
                "knowledge_name": getattr(item, "name", ""),
                "source_content": getattr(source, "content", "") if source else "",
                "source_atoms": getattr(source, "atoms", []) if source else [],
                "content_md": getattr(item, "content_md", ""),
            })
        return rows

    def record_model_call(
        self,
        messages: Sequence[Any],
        *,
        response: Any | None = None,
        elapsed_ms: float,
        error_type: str | None = None,
    ) -> None:
        try:
            contents = [str(getattr(message, "content", "")) for message in messages]
            joined = "\n".join(contents)
            stage = "global" if "[TASK:rerank_global]" in joined else "batch"
            user_prompt = contents[-1] if contents else ""
            call = {
                "call_index": len(self.model_calls) + 1,
                "stage": stage,
                "elapsed_ms": round(elapsed_ms, 3),
                "prompt": {
                    "system": contents[0] if contents else "",
                    "user": user_prompt,
                    "system_chars": len(contents[0]) if contents else 0,
                    "user_chars": len(user_prompt),
                    "total_chars": sum(len(item) for item in contents),
                },
                "raw_output": str(getattr(response, "content", response)) if response is not None else None,
            }
            payload_match = re.search(
                r"RERANK_INPUT_BEGIN\s*(\{.*\})\s*RERANK_INPUT_END",
                user_prompt,
                re.DOTALL,
            )
            if payload_match:
                try:
                    call["input_payload"] = json.loads(payload_match.group(1))
                except (json.JSONDecodeError, TypeError, ValueError):
                    call["input_payload"] = {"parse_error": "invalid_prompt_json"}
            if error_type:
                call["error"] = {"type": error_type, "message": "模型调用失败"}
            self.model_calls.append(_safe_snapshot(call, text_budget=100_000))
        except Exception as exc:  # noqa: BLE001 - Trace 失败不能影响真实模型结果
            self.collection_errors.append(type(exc).__name__)

    def finish(self, workspace: Any) -> None:
        data = workspace.data
        self.final = _safe_snapshot({
            "top3_candidates": data.get("top3_candidates", []),
            "processed_chunks": data.get("processed_chunks", []),
            "processing_meta": data.get("processing_meta"),
            "warnings": _warnings(data),
            "total_elapsed_ms": workspace.tracer.elapsed_ms(),
            "events": workspace.tracer.events,
        })

    def export(self, trace_id: str) -> dict[str, Any]:
        stages = copy.deepcopy(self.stages)
        rerank = stages.setdefault("rerank", {})
        model_calls = copy.deepcopy(self.model_calls)
        rerank_output = rerank.get("output", {})
        rerank_details = (
            rerank_output.get("rerank_details", {})
            if isinstance(rerank_output, dict) else {}
        )
        batch_details = [
            item for item in rerank_details.get("batches", [])
            if isinstance(item, dict)
            and isinstance(item.get("prompt"), dict)
            and item["prompt"].get("within_budget")
        ] if isinstance(rerank_details, dict) else []
        batch_index = 0
        global_details = (
            rerank_details.get("global", {})
            if isinstance(rerank_details, dict) else {}
        )
        for call in model_calls:
            if not isinstance(call, dict):
                continue
            if call.get("stage") == "batch":
                if batch_index < len(batch_details):
                    call["parsed_result"] = batch_details[batch_index]
                    batch_index += 1
            elif isinstance(global_details, dict):
                call["parsed_result"] = global_details
        rerank["model_calls"] = model_calls
        payload = {
            "schema_version": 1,
            "trace_id": trace_id,
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3),
            "request": self.request,
            "stages": stages,
            "final": copy.deepcopy(self.final),
            "collection_errors": list(self.collection_errors),
            "limits": {
                "max_response_bytes": MAX_TRACE_RESPONSE_BYTES,
                "max_field_chars": MAX_TRACE_FIELD_CHARS,
                "max_list_items": MAX_TRACE_LIST_ITEMS,
            },
        }
        sanitizer = _Sanitizer(MAX_EXPORT_TEXT_CHARS)
        safe = sanitizer.clean(payload)
        if isinstance(safe, dict) and sanitizer.truncated:
            safe["trace_truncated"] = True
        encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) <= MAX_TRACE_RESPONSE_BYTES:
            return safe
        return {
            "schema_version": 1,
            "trace_id": trace_id,
            "elapsed_ms": safe.get("elapsed_ms", 0) if isinstance(safe, dict) else 0,
            "trace_truncated": True,
            "truncation_reason": "trace_response_size_limit",
            "stage_summary": {
                name: {
                    "elapsed_ms": record.get("elapsed_ms", 0),
                    "has_input": "input" in record,
                    "has_output": "output" in record,
                }
                for name, record in self.stages.items()
            },
            "limits": {"max_response_bytes": MAX_TRACE_RESPONSE_BYTES},
        }
