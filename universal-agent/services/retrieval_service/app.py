# -*- coding: utf-8 -*-
"""retrieval 服务层 — 纯检索结果的 FastAPI 封装(对照 kbagent_service)。

启动(在 knowbase-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    Windows:  set PYTHONPATH=src;services && python -m uvicorn retrieval_service.app:app --host 0.0.0.0 --port 8000
    Linux:    PYTHONPATH=src:services python -m uvicorn retrieval_service.app:app --host 0.0.0.0 --port 8000

服务定位:
    只调用固定检索流水线(intergrate_all,全程零 LLM),不拼接对话历史、
    不做答案生成;需要完整问答(检索 + 加工 + 生成)时走 kbagent_service。

并发模型:
    - es 全局共享(ES client 需线程安全);
    - 每请求新建 RunWorkspace / RetrievalSubAgent / Tracer,请求间零共享;
    - 检索链路为纯同步 IO(HTTP 调 ngkm),经 asyncio.wait_for 包端到端超时。

后端选择:
    - 缺省 produce → ProduceESClient(生产 ngkm 一体化流水线);
    - 仅 RETRIEVAL_SERVICE_BACKEND=mock 时回退离线 MockESClient(内置样例);
    - 也可 create_app(es=...) 显式注入覆盖。
"""
from __future__ import annotations

# ── 直接脚本运行引导(python app.py)──────────────────────────
# 顶层 import 执行前先补好运行环境:
# 1. 把项目根(universal-agent/)、src/、services/ 加入 sys.path;
# 2. 设置 __package__ → 直接运行时相对导入(from .models/.runner)可正确解析。
# 以 python -m uvicorn retrieval_service.app:app 方式启动时 __package__ 已有值,跳过。
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]   # .../universal-agent
    for _p in (str(_root), str(_root / "src"), str(_root / "services")):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    __package__ = "retrieval_service"

import asyncio
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kbagent.shared.search import ESClient, MockESClient, ProduceESClient

from .models import (
    RTN_BAD_REQUEST,
    RTN_INTERNAL,
    RTN_OK,
    RTN_TIMEOUT,
    RetrievalRequest,
    RetrievalResponse,
    error_body,
)
from .runner import run_retrieval_request

logger = logging.getLogger("retrieval_service")


def _setup_logging() -> None:
    """配置根日志器,保证每请求 INFO 行可见。

    经 ``python -m uvicorn`` 启动时根日志器默认无 handler,INFO 会被丢弃、
    只有 WARNING+ 经 lastResort 落到 stderr。级别可用环境变量
    RETRIEVAL_SERVICE_LOG_LEVEL 覆盖。
    """
    level = getattr(logging,
                    os.environ.get("RETRIEVAL_SERVICE_LOG_LEVEL", "INFO").upper(),
                    logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    else:
        root.setLevel(level)


_setup_logging()

# 端到端超时(秒):纯检索链路(无 LLM 多轮),60s 覆盖内网 ngkm 慢查询;
# 超时返回 50002
DEFAULT_TIMEOUT_S = 60.0
# ngkm 单次 HTTP 超时(秒),下传 ProduceESClient
DEFAULT_NGKM_TIMEOUT_S = 30
# 检索后端选择:缺省 produce=生产 ngkm 一体化流水线(ProduceESClient);
# 仅当显式设 RETRIEVAL_SERVICE_BACKEND=mock 时才用离线 MockESClient(本地演示)
ENV_BACKEND = "RETRIEVAL_SERVICE_BACKEND"
# ProduceESClient 缺省区域(请求未携带 region_code 时兜底);支持省份名或区号
ENV_REGION = "RETRIEVAL_SERVICE_REGION"
# ngkm HTTP 超时(秒)环境变量
ENV_TIMEOUT = "RETRIEVAL_SERVICE_TIMEOUT"
# 请求ID白名单:允许字母数字与 ._:-,防止头注入与超长日志
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

# ── 路由前缀 ────────────────────────────────────────────────────────────────
# 形如 /api/{服务名}/{环境},业务端点挂在其下(如 {base}/retrieve)。
# 可经环境变量 RETRIEVAL_SERVICE_BASE_PATH 覆盖(如 /api/retrieval-service/test-prod)。
ENV_BASE_PATH = "RETRIEVAL_SERVICE_BASE_PATH"
DEFAULT_BASE_PATH = "/api/retrieval-service/prod"


def _resolve_base_path(base_path: Optional[str]) -> str:
    """规范化路由前缀:确保以 / 开头、无尾部 /。"""
    bp = (base_path or os.environ.get(ENV_BASE_PATH) or DEFAULT_BASE_PATH).strip()
    if not bp.startswith("/"):
        bp = "/" + bp
    return bp.rstrip("/")


def _request_id(request: Request) -> str:
    """请求标识:优先取 X-Request-ID 头(合法时),缺省服务端生成。"""
    supplied = request.headers.get("X-Request-ID", "").strip()
    if _REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return f"retrieval-{uuid.uuid4().hex}"


def _default_es() -> ESClient:
    """按环境变量选择检索后端;**缺省即生产后端**。

    默认/RETRIEVAL_SERVICE_BACKEND=produce → ProduceESClient(生产 ngkm
    一体化流水线:槽位提取 → 知识主索引召回 → 原子表拼接,full_recall 必然可用);
    仅 RETRIEVAL_SERVICE_BACKEND=mock → 离线 MockESClient(内置样例,本地演示)。
    ProduceESClient 构造本身不发网络请求,故不做异常回退;
    真实失败发生在 full_recall 调用时,由 runner/app 的异常路径处理。
    """
    backend = os.environ.get(ENV_BACKEND, "").strip().lower()
    if backend == "mock":
        logger.warning("RETRIEVAL_SERVICE_BACKEND=mock: 使用离线 MockESClient(仅内置样例)")
        return MockESClient()

    region = os.environ.get(ENV_REGION, "000").strip() or "000"
    try:
        ngkm_timeout = int(os.environ.get(ENV_TIMEOUT, str(DEFAULT_NGKM_TIMEOUT_S)))
    except ValueError:
        ngkm_timeout = DEFAULT_NGKM_TIMEOUT_S
    logger.info("检索后端: 生产 ngkm ProduceESClient region=%s timeout=%ss",
                region, ngkm_timeout)
    return ProduceESClient(region_code=region, timeout=ngkm_timeout)


def create_app(es: Any = None,
               timeout_s: float = DEFAULT_TIMEOUT_S,
               base_path: Optional[str] = None) -> FastAPI:
    """创建服务应用。

    es 可显式注入(测试或生产接真实依赖),缺省按环境变量构建;
    base_path 为业务路由前缀,缺省取环境变量 RETRIEVAL_SERVICE_BASE_PATH,
    再缺省为 DEFAULT_BASE_PATH(/api/retrieval-service/prod)。
    """
    base = _resolve_base_path(base_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.es = es or _default_es()
        app.state.timeout_s = timeout_s
        logger.info("retrieval 服务就绪 base=%s es=%s",
                    base, type(app.state.es).__name__)
        yield

    app = FastAPI(title="retrieval-service", version="1.0.0", lifespan=lifespan)
    _register_routes(app, base)
    return app


def _register_routes(app: FastAPI, base: str) -> None:

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        request_id = _request_id(request)
        detail = "; ".join(
            f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg', '')}"
            for e in exc.errors()[:5]
        )
        logger.warning("request_id=%s 参数错误: %s", request_id, detail)
        return JSONResponse(error_body(RTN_BAD_REQUEST, f"参数错误: {detail}"))

    # 健康检查固定在根路径,供网关/负载均衡探活;backend 便于确认实际后端
    @app.get("/health")
    async def health(request: Request) -> dict:
        return {"status": "ok",
                "backend": type(request.app.state.es).__name__}

    @app.post(f"{base}/retrieve", response_model=RetrievalResponse)
    async def retrieve(payload: RetrievalRequest, request: Request):
        request_id = _request_id(request)
        try:
            result = await asyncio.wait_for(
                run_retrieval_request(payload,
                                      es=request.app.state.es,
                                      request_id=request_id),
                timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("request_id=%s 检索端到端超时", request_id)
            return JSONResponse(error_body(RTN_TIMEOUT, "服务处理超时"))
        except Exception:  # noqa: BLE001
            logger.exception("request_id=%s 检索未预期异常", request_id)
            return JSONResponse(error_body(RTN_INTERNAL, "服务内部错误"))

        logger.info("request_id=%s traceId=%s outcome=%s degraded=%s "
                    "region=%s recalled=%d elapsedMs=%d",
                    request_id, result.trace_id, result.outcome,
                    result.degraded, result.region_code,
                    result.recalled_count, result.elapsed_ms)
        return RetrievalResponse(rtnCode=RTN_OK, rtnMsg="success", object=result)


# 默认应用实例:python -m uvicorn retrieval_service.app:app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
