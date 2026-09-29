"""Processing生产摘要日志；不记录Query、正文、Prompt或模型原始输出。"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Sequence


ENV_LOG_LEVEL = "KB_PROCESSING_LOG_LEVEL"
_PROMPT_PAYLOAD_RE = re.compile(
    r"RERANK_INPUT_BEGIN\s*(\{.*\})\s*RERANK_INPUT_END",
    re.DOTALL,
)
_STAGE_NAMES = {
    "analyze_knowledge_candidates": "analyze",
    "filter_knowledge_candidates": "filter",
    "build_knowledge_markdown": "build_markdown",
    "rerank_knowledge_candidates": "rerank",
}


def _configured_level() -> int:
    value = os.environ.get(ENV_LOG_LEVEL, "INFO").strip().upper()
    return getattr(logging, value, logging.INFO)


# 作为uvicorn.error的子logger，复用Uvicorn已经配置好的容器stdout/stderr handler。
logger = logging.getLogger("uvicorn.error.processing_service")
logger.setLevel(_configured_level())


def log_event(level: int, event: str, **fields: Any) -> None:
    """输出单行JSON摘要；调用失败不得影响Processing主链路。"""
    try:
        payload = {"event": event, **fields}
        logger.log(
            level,
            "processing_event=%s",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
        )
    except Exception:  # noqa: BLE001 - 日志绝不能影响业务请求
        return


def _message_contents(messages: Sequence[Any]) -> list[str]:
    return [str(getattr(message, "content", "")) for message in messages]


def _model_stage(joined: str) -> str:
    if "[STAGE:rerank_semifinal]" in joined:
        return "semifinal"
    if "[TASK:rerank_global]" in joined:
        return "global"
    if "[TASK:rerank_batch]" in joined:
        return "batch"
    return "unknown"


def _prompt_summary(contents: Sequence[str]) -> dict[str, Any]:
    joined = "\n".join(contents)
    summary: dict[str, Any] = {
        "rerank_stage": _model_stage(joined),
        "prompt_chars": sum(len(item) for item in contents),
        "candidate_count": 0,
        "top_k": 0,
        "candidate_ids": [],
    }
    user_prompt = contents[-1] if contents else ""
    matched = _PROMPT_PAYLOAD_RE.search(user_prompt)
    if not matched:
        return summary
    try:
        payload = json.loads(matched.group(1))
    except (json.JSONDecodeError, TypeError, ValueError):
        return summary
    if not isinstance(payload, dict):
        return summary
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        candidate_ids = [
            str(item.get("evidence_id"))
            for item in candidates
            if isinstance(item, dict) and item.get("evidence_id") is not None
        ]
        summary["candidate_count"] = len(candidates)
        summary["candidate_ids"] = candidate_ids
    top_k = payload.get("top_k")
    if isinstance(top_k, int):
        summary["top_k"] = top_k
    return summary


def _output_summary(response: Any, candidate_ids: Sequence[str], top_k: int) -> dict[str, Any]:
    content = str(getattr(response, "content", response))
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "output_status": "invalid_json",
            "ranked_id_count": 0,
            "valid_id_count": 0,
            "unknown_id_count": 0,
            "duplicate_id_count": 0,
            "exact_count": False,
        }
    if not isinstance(payload, dict) or set(payload) != {"ranked_ids"}:
        return {
            "output_status": "invalid_schema",
            "ranked_id_count": 0,
            "valid_id_count": 0,
            "unknown_id_count": 0,
            "duplicate_id_count": 0,
            "exact_count": False,
        }
    ranked_ids = payload.get("ranked_ids")
    if not isinstance(ranked_ids, list):
        return {
            "output_status": "invalid_schema",
            "ranked_id_count": 0,
            "valid_id_count": 0,
            "unknown_id_count": 0,
            "duplicate_id_count": 0,
            "exact_count": False,
        }
    allowed = set(candidate_ids)
    seen: set[str] = set()
    valid_count = 0
    unknown_count = 0
    duplicate_count = 0
    for item in ranked_ids:
        if not isinstance(item, str) or item not in allowed:
            unknown_count += 1
            continue
        if item in seen:
            duplicate_count += 1
            continue
        seen.add(item)
        valid_count += 1
    exact = (
        len(ranked_ids) == top_k
        and valid_count == top_k
        and not unknown_count
        and not duplicate_count
    )
    return {
        "output_status": "complete" if exact else "incomplete",
        "ranked_id_count": len(ranked_ids),
        "valid_id_count": valid_count,
        "unknown_id_count": unknown_count,
        "duplicate_id_count": duplicate_count,
        "exact_count": exact,
    }


class _LoggingModel:
    """只观测一次真实ainvoke，不重试、不补调、不读取敏感配置。"""

    def __init__(self, delegate: Any, observer: "ProcessingLogObserver") -> None:
        self._delegate = delegate
        self._observer = observer

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        contents = _message_contents(messages)
        prompt = _prompt_summary(contents)
        stage = str(prompt.pop("rerank_stage"))
        candidate_ids = list(prompt.pop("candidate_ids"))
        stage_call_index = self._observer.next_model_call_index(stage)
        started = time.perf_counter()
        try:
            response = await self._delegate.ainvoke(messages, **kwargs)
        except asyncio.CancelledError:
            log_event(
                logging.WARNING,
                "model_call_completed",
                request_id=self._observer.request_id,
                trace_id=self._observer.trace_id,
                rerank_stage=stage,
                stage_call_index=stage_call_index,
                status="cancelled",
                error_type="CancelledError",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                **prompt,
            )
            raise
        except Exception as exc:
            log_event(
                logging.WARNING,
                "model_call_completed",
                request_id=self._observer.request_id,
                trace_id=self._observer.trace_id,
                rerank_stage=stage,
                stage_call_index=stage_call_index,
                status="error",
                error_type=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                **prompt,
            )
            raise
        output = _output_summary(response, candidate_ids, int(prompt["top_k"]))
        log_event(
            logging.INFO if output["exact_count"] else logging.WARNING,
            "model_call_completed",
            request_id=self._observer.request_id,
            trace_id=self._observer.trace_id,
            rerank_stage=stage,
            stage_call_index=stage_call_index,
            status="success",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            **prompt,
            **output,
        )
        return response


class ProcessingLogObserver:
    """组合生产日志与可选调试Trace，不改变模型调用和流水线结果。"""

    def __init__(self, request_id: str, delegate: Any | None = None) -> None:
        self.request_id = request_id
        self.delegate = delegate
        self.trace_id = ""
        self._stage_started: dict[str, float] = {}
        self._model_call_counts: dict[str, int] = defaultdict(int)

    def next_model_call_index(self, stage: str) -> int:
        self._model_call_counts[stage] += 1
        return self._model_call_counts[stage]

    def wrap_model(self, model: Any) -> Any:
        wrapped = model
        callback = getattr(self.delegate, "wrap_model", None)
        if callable(callback):
            try:
                wrapped = callback(model)
            except Exception:  # noqa: BLE001 - 可选Trace失败不影响日志和主链路
                wrapped = model
        return _LoggingModel(wrapped, self)

    def before_stage(self, tool_name: str, workspace: Any) -> None:
        self.trace_id = str(getattr(getattr(workspace, "tracer", None), "trace_id", ""))
        stage = _STAGE_NAMES.get(tool_name, tool_name)
        self._stage_started[tool_name] = time.perf_counter()
        log_event(
            logging.INFO,
            "stage_started",
            request_id=self.request_id,
            trace_id=self.trace_id,
            stage=stage,
        )
        callback = getattr(self.delegate, "before_stage", None)
        if callable(callback):
            try:
                callback(tool_name, workspace)
            except Exception:  # noqa: BLE001
                pass

    def after_stage(
        self,
        tool_name: str,
        workspace: Any,
        *,
        error_type: str | None = None,
    ) -> None:
        callback = getattr(self.delegate, "after_stage", None)
        if callable(callback):
            try:
                callback(tool_name, workspace, error_type=error_type)
            except Exception:  # noqa: BLE001
                pass
        stage = _STAGE_NAMES.get(tool_name, tool_name)
        started = self._stage_started.pop(tool_name, time.perf_counter())
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if error_type:
            log_event(
                logging.ERROR,
                "stage_completed",
                request_id=self.request_id,
                trace_id=self.trace_id,
                stage=stage,
                status="error",
                error_type=error_type,
                elapsed_ms=elapsed_ms,
            )
            return
        data = getattr(workspace, "data", {})
        meta = data.get("processing_meta")
        fields: dict[str, Any] = {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "stage": stage,
            "status": "success",
            "elapsed_ms": elapsed_ms,
            "warning_count": int(getattr(meta, "warning_count", 0) or 0),
        }
        if stage == "analyze":
            fields.update(
                input_count=int(getattr(meta, "input_count", 0) or 0),
                output_count=int(getattr(meta, "normalized_count", 0) or 0),
            )
        elif stage == "filter":
            fields.update(
                input_count=int(getattr(meta, "normalized_count", 0) or 0),
                output_count=int(getattr(meta, "filtered_count", 0) or 0),
            )
        elif stage == "build_markdown":
            fields.update(
                input_count=int(getattr(meta, "filtered_count", 0) or 0),
                output_count=int(getattr(meta, "processed_count", 0) or 0),
                rerank_eligible_count=int(
                    getattr(meta, "rerank_eligible_count", 0) or 0
                ),
            )
        elif stage == "rerank":
            details = data.get("rerank_details")
            details = details if isinstance(details, dict) else {}
            batches = details.get("batches")
            batches = batches if isinstance(batches, list) else []
            semifinal = details.get("semifinal")
            semifinal = semifinal if isinstance(semifinal, dict) else {}
            global_detail = details.get("global")
            global_detail = global_detail if isinstance(global_detail, dict) else {}
            batch_prompt_chars = [
                int(item.get("prompt", {}).get("sent_chars", 0) or 0)
                for item in batches
                if isinstance(item, dict) and isinstance(item.get("prompt"), dict)
            ]
            semifinal_prompt = semifinal.get("prompt")
            semifinal_prompt = (
                semifinal_prompt if isinstance(semifinal_prompt, dict) else {}
            )
            global_prompt = global_detail.get("prompt")
            global_prompt = global_prompt if isinstance(global_prompt, dict) else {}
            warning_codes = list(dict.fromkeys(
                str(getattr(item, "code", "unknown_warning"))
                for item in data.get("processing_warnings", [])
            ))
            fields.update(
                input_count=int(getattr(meta, "processed_count", 0) or 0),
                eligible_count=int(getattr(meta, "rerank_eligible_count", 0) or 0),
                batch_count=len(batches),
                incomplete_batch_count=sum(
                    not bool(item.get("complete"))
                    for item in batches
                    if isinstance(item, dict)
                ),
                batch_prompt_chars_total=sum(batch_prompt_chars),
                batch_prompt_chars_max=max(batch_prompt_chars, default=0),
                batch_prompt_budget=(
                    int(batches[0].get("prompt", {}).get("budget_chars", 0) or 0)
                    if batches and isinstance(batches[0], dict) else 0
                ),
                semifinal_used=bool(semifinal),
                semifinal_input_count=len(semifinal.get("input_ids", [])),
                semifinal_output_count=len(semifinal.get("selected_ids", [])),
                semifinal_complete=(
                    bool(semifinal.get("complete")) if semifinal else None
                ),
                semifinal_prompt_chars=int(
                    semifinal_prompt.get("sent_chars", 0) or 0
                ),
                semifinal_prompt_budget=int(
                    semifinal_prompt.get("budget_chars", 0) or 0
                ),
                global_pool_count=len(global_detail.get("pool_ids", [])),
                global_complete=bool(global_detail.get("complete")),
                global_mode=global_detail.get("mode"),
                global_prompt_chars=int(global_prompt.get("sent_chars", 0) or 0),
                global_prompt_budget=int(global_prompt.get("budget_chars", 0) or 0),
                output_count=int(getattr(meta, "top_count", 0) or 0),
                degraded=bool(getattr(meta, "degraded", False)),
                degradation_reasons=list(
                    getattr(meta, "degradation_reasons", []) or []
                ),
                warning_codes=warning_codes,
            )
        log_event(logging.INFO, "stage_completed", **fields)

    def finish(self, workspace: Any) -> None:
        callback = getattr(self.delegate, "finish", None)
        if callable(callback):
            try:
                callback(workspace)
            except Exception:  # noqa: BLE001
                pass
