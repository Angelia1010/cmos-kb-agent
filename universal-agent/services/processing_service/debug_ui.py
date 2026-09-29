"""挂载在 Processing 主服务内的受保护调试页面与单次执行接口。"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .debug_trace import ProcessingTraceCollector
from .models import ProcessingRequest
from .runner import run_processing_request


STATIC_ROOT = Path(__file__).resolve().parent / "static" / "processing_debug"
MAX_DEBUG_RESPONSE_BYTES = 50_000_000
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def create_debug_router(processing_path: str) -> APIRouter:
    """创建继承正式 Processing 路径前缀的调试 Router。"""
    normalized_path = "/" + processing_path.strip("/")
    ui_path = f"{normalized_path}/ui"
    router = APIRouter(prefix=normalized_path, include_in_schema=False)

    @router.get("/ui")
    async def debug_index() -> HTMLResponse:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        response = HTMLResponse(html.replace("__PROCESSING_UI_BASE__", ui_path))
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @router.post("/ui/run")
    async def debug_run(
        payload: ProcessingRequest,
        request: Request,
    ) -> JSONResponse:
        if len(payload.chunks) > 300:
            raise HTTPException(status_code=413, detail="调试页面最多支持300条候选")

        request_id = request.headers.get("X-Request-ID", "").strip()
        if not _REQUEST_ID_PATTERN.fullmatch(request_id):
            request_id = f"processing-debug-{uuid.uuid4().hex}"
        collector = ProcessingTraceCollector(payload.model_dump())
        try:
            result = await asyncio.wait_for(
                run_processing_request(
                    payload,
                    model=request.app.state.model,
                    request_id=request_id,
                    trace_collector=collector,
                ),
                timeout=request.app.state.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Processing调试请求超时") from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - 不回显异常值或堆栈
            raise HTTPException(status_code=500, detail="Processing调试请求执行失败") from exc

        body: dict[str, Any] = {
            "trace_id": result.trace_id,
            "final_result": result.model_dump(),
            "trace": collector.export(result.trace_id),
        }
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_DEBUG_RESPONSE_BYTES:
            body["trace"] = collector.summary(
                result.trace_id,
                reason="debug_response_size_limit",
            )
            encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_DEBUG_RESPONSE_BYTES:
            raise HTTPException(status_code=413, detail="调试响应超过大小上限")
        response = JSONResponse(content=body)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return router
