"""独立 Retrieval FastAPI 服务,只调用固定检索流水线(intergrate_all)。

后端选择(对照 processing_service 的 model 注入模式):
- 生产:ProduceESClient(ngkm 槽位提取 → 知识主索引 → 原子表拼接);
- 离线/测试:MockESClient(内置坐席知识库样例,无网络依赖)。
默认按环境变量 RETRIEVAL_SERVICE_BACKEND 切换(produce/mock),缺省 produce。
也可在 create_app 时直接注入 es 实例(用于单测)。
"""
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

from kbagent.shared.search import ESClient, MockESClient, ProduceESClient

from .models import (
    RTN_BAD_REQUEST,
    RTN_INTERNAL,
    RTN_TIMEOUT,
    RetrievalErrorResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from .runner import run_retrieval_request


logger = logging.getLogger("retrieval_service")

ENV_BASE_PATH = "RETRIEVAL_SERVICE_BASE_PATH"
ENV_BACKEND = "RETRIEVAL_SERVICE_BACKEND"      # produce | mock
ENV_REGION = "RETRIEVAL_SERVICE_REGION"         # 默认区号,如 000 / 591
ENV_TIMEOUT = "RETRIEVAL_SERVICE_TIMEOUT"       # ngkm HTTP 超时(秒)
DEFAULT_BASE_PATH = "/api/retrieval-service/prod"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_NGKM_TIMEOUT_S = 30
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
    return f"retrieval-{uuid.uuid4().hex}"


def _error(code: str, message: str, request_id: str, status_code: int) -> JSONResponse:
    payload = RetrievalErrorResponse(
        rtnCode=code,
        rtnMsg=message,
        request_id=request_id,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _default_es() -> ESClient:
    """按环境变量构建检索后端。

    - RETRIEVAL_SERVICE_BACKEND=mock → MockESClient(离线演示/自测);
    - 缺省/produce → ProduceESClient(生产 ngkm,需网络可达
      restapi.ly4.tyyt.cmos / restapi.ngkmsearch.cs.glb.cmos)。

    ProduceESClient 构造本身不发网络请求,故不做异常回退;
    真实失败发生在 full_recall 调用时,由 runner/app 的异常路径处理。
    """
    backend = os.environ.get(ENV_BACKEND, "produce").strip().lower()
    if backend == "mock":
        logger.info("Retrieval 后端=MockESClient(离线)")
        return MockESClient()

    region = os.environ.get(ENV_REGION, "000").strip() or "000"
    try:
        ngkm_timeout = int(os.environ.get(ENV_TIMEOUT, str(DEFAULT_NGKM_TIMEOUT_S)))
    except ValueError:
        ngkm_timeout = DEFAULT_NGKM_TIMEOUT_S
    logger.info("Retrieval 后端=ProduceESClient region=%s timeout=%ss",
                region, ngkm_timeout)
    return ProduceESClient(region_code=region, timeout=ngkm_timeout)


def create_app(
    *,
    es: ESClient | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    base_path: str | None = None,
) -> FastAPI:
    """创建 Retrieval 服务;es 缺省按环境变量构建(生产 ProduceESClient)。"""
    base = _resolve_base_path(base_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.es = es if es is not None else _default_es()
        app.state.timeout_s = timeout_s
        logger.info("Retrieval 服务就绪 base=%s backend=%s",
                    base, type(app.state.es).__name__)
        yield

    app = FastAPI(title="retrieval-service", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        request_id = _request_id(request)
        fields = [".".join(str(part) for part in item.get("loc", ())) for item in exc.errors()[:5]]
        detail = ", ".join(field for field in fields if field) or "request body"
        logger.warning("request_id=%s 请求格式错误 fields=%s", request_id, detail)
        return _error(RTN_BAD_REQUEST, f"请求格式错误: {detail}", request_id, 422)

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        return {"status": "ok",
                "backend": type(request.app.state.es).__name__}

    @app.post(f"{base}/retrieve", response_model=RetrievalResponse)
    async def retrieve(payload: RetrievalRequest, request: Request):
        request_id = _request_id(request)
        try:
            result = await asyncio.wait_for(
                run_retrieval_request(
                    payload,
                    es=request.app.state.es,
                    request_id=request_id,
                ),
                timeout=request.app.state.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error("request_id=%s Retrieval 请求超时", request_id)
            return _error(RTN_TIMEOUT, "Retrieval 处理超时", request_id, 504)
        except Exception as exc:  # noqa: BLE001 - HTTP 边界不回显堆栈或异常值
            logger.error(
                "request_id=%s Retrieval 内部异常 error_type=%s",
                request_id,
                type(exc).__name__,
            )
            return _error(RTN_INTERNAL, "Retrieval 服务内部错误", request_id, 500)

        logger.info(
            "request_id=%s trace_id=%s outcome=%s degraded=%s region=%s "
            "recalled=%d elapsed_ms=%d",
            request_id,
            result.trace_id,
            result.outcome,
            result.degraded,
            result.region_code,
            result.recalled_count,
            result.elapsed_ms,
        )
        return RetrievalResponse(object=result)

    return app


app = create_app()
