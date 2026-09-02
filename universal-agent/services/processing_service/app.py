"""独立 Processing FastAPI 服务，只调用固定知识处理编排器。"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kbagent.scripted_model import ScriptedChatModel

from .models import (
    RTN_BAD_REQUEST,
    RTN_INTERNAL,
    RTN_TIMEOUT,
    ProcessingErrorResponse,
    ProcessingRequest,
    ProcessingResponse,
)
from .runner import run_processing_request


logger = logging.getLogger("processing_service")

ENV_BASE_PATH = "PROCESSING_SERVICE_BASE_PATH"
DEFAULT_BASE_PATH = "/api/processing-service/prod"
DEFAULT_TIMEOUT_S = 60.0
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _resolve_base_path(base_path: str | None) -> str:
    value = (base_path or os.environ.get(ENV_BASE_PATH) or DEFAULT_BASE_PATH).strip()
    if not value.startswith("/"):
        value = "/" + value
    return value.rstrip("/")


def _request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID", "").strip()
    if _REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return f"processing-{uuid.uuid4().hex}"


def _error(code: str, message: str, request_id: str, status_code: int) -> JSONResponse:
    payload = ProcessingErrorResponse(
        rtnCode=code,
        rtnMsg=message,
        request_id=request_id,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def create_app(
    *,
    model: Any | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    base_path: str | None = None,
) -> FastAPI:
    """创建首轮 Scripted Processing 服务；model 参数仅用于离线故障测试。"""
    base = _resolve_base_path(base_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model if model is not None else ScriptedChatModel()
        app.state.timeout_s = timeout_s
        logger.info("Processing 服务就绪 base=%s model_mode=scripted", base)
        yield

    app = FastAPI(title="processing-service", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        request_id = _request_id(request)
        fields = [".".join(str(part) for part in item.get("loc", ())) for item in exc.errors()[:5]]
        detail = ", ".join(field for field in fields if field) or "request body"
        logger.warning("request_id=%s 请求格式错误 fields=%s", request_id, detail)
        return _error(RTN_BAD_REQUEST, f"请求格式错误: {detail}", request_id, 422)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model_mode": "scripted"}

    @app.post(f"{base}/process", response_model=ProcessingResponse)
    async def process(payload: ProcessingRequest, request: Request):
        request_id = _request_id(request)
        try:
            result = await asyncio.wait_for(
                run_processing_request(
                    payload,
                    model=request.app.state.model,
                    request_id=request_id,
                ),
                timeout=request.app.state.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error("request_id=%s Processing 请求超时", request_id)
            return _error(RTN_TIMEOUT, "Processing 处理超时", request_id, 504)
        except Exception as exc:  # noqa: BLE001 - HTTP 边界不回显堆栈或异常值
            logger.error(
                "request_id=%s Processing 内部异常 error_type=%s",
                request_id,
                type(exc).__name__,
            )
            return _error(RTN_INTERNAL, "Processing 服务内部错误", request_id, 500)

        logger.info(
            "request_id=%s trace_id=%s outcome=%s degraded=%s input=%d top=%d elapsed_ms=%d",
            request_id,
            result.trace_id,
            result.outcome,
            result.degraded,
            result.processing_meta.input_count,
            len(result.top3_candidates),
            result.elapsed_ms,
        )
        return ProcessingResponse(object=result)

    return app


app = create_app()
