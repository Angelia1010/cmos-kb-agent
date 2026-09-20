# -*- coding: utf-8 -*-
"""retrieval 服务层 — GoalLoop 三轮形态检索的 FastAPI 封装。

启动(在 knowbase-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    Windows:  set PYTHONPATH=src;services && python -m uvicorn retrieval_service.app:app --host 0.0.0.0 --port 8000
    Linux:    PYTHONPATH=src:services python -m uvicorn retrieval_service.app:app --host 0.0.0.0 --port 8000

服务定位:
    调用 RetrievalSubAgent(GoalLoop 三轮形态)执行检索:
    intergrate_all(原始 query) → query_rewrite(LLM 重写) → intergrate_all(改写 query)
    需要注入 model 供 LLM 推理与 query_rewrite/关键词提取使用。

并发模型:
    - es/model 全局共享(需线程安全);
    - 每请求新建 RunWorkspace / RetrievalSubAgent / Tracer,请求间零共享;
    - 检索链路内 GoalLoop 调 LLM + HTTP 调 ngkm,经 asyncio.wait_for 包端到端超时。

后端选择:
    - 缺省 produce → ProduceESClient(生产 ngkm 一体化流水线);
    - 仅 RETRIEVAL_SERVICE_BACKEND=mock 时回退离线 MockESClient(内置样例);
    - 也可 create_app(es=..., model=...) 显式注入覆盖。
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
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from kbagent import ScriptedChatModel
from kbagent.shared.search import ESClient, ProduceESClient

# from .models import (
#     RTN_BAD_REQUEST,
#     RTN_INTERNAL,
#     RTN_OK,
#     RTN_TIMEOUT,
#     RetrievalRequest,
#     RetrievalResponse,
#     error_body,
# )
from .models import (
    RTN_BAD_REQUEST,
    RTN_INTERNAL,
    RTN_OK,
    RTN_TIMEOUT,
    BatchRetrievalItem,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    BatchRetrievalResponseObject,
    KeywordRequest,
    RetrievalRequest,
    RetrievalResponse,
    VectorRequest,
    error_body,
)
from .runner import run_keyword_request, run_retrieval_request, run_vector_request

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

# 端到端超时(秒):GoalLoop 三轮(含 LLM 调用),60s 覆盖内网 ngkm + LLM 慢查询;
# 超时返回 50002
# DEFAULT_TIMEOUT_S = 60.0
# 观测完整请求耗时期间不设端到端超时:wait_for(timeout=None) 等价于无限等待,
# 保证慢请求不被 50002 中断(排障结束后按需恢复)
DEFAULT_TIMEOUT_S: Optional[float] = None
# ngkm 单次 HTTP 超时(秒),下传 ProduceESClient
DEFAULT_NGKM_TIMEOUT_S = 30
ENV_BACKEND = "RETRIEVAL_SERVICE_BACKEND"
ENV_REGION = "RETRIEVAL_SERVICE_REGION"
ENV_TIMEOUT = "RETRIEVAL_SERVICE_TIMEOUT"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

ENV_BASE_PATH = "RETRIEVAL_SERVICE_BASE_PATH"
DEFAULT_BASE_PATH = "/api/retrieval-service/prod"

# 测试集路径:环境变量 RETRIEVAL_SERVICE_TESTSET 配置;
# 缺省为 services/retrieval_service/测试集.json(相对项目根 universal-agent/);
# 兜底再退到本文件同目录的 测试集.json,保证任意 cwd 下均可定位
ENV_TESTSET = "RETRIEVAL_SERVICE_TESTSET"
DEFAULT_TESTSET_REL = "services/retrieval_service/测试集.json"


def _resolve_testset_path() -> str:
    """解析测试集文件路径。

    优先级:
      1. 环境变量 RETRIEVAL_SERVICE_TESTSET(原样使用,允许绝对或相对路径);
      2. 项目根(universal-agent/)下的 services/retrieval_service/测试集.json;
      3. 本文件(app.py)同目录的 测试集.json(兼容 python app.py 在不同 cwd 启动)。
    """
    env_path = os.environ.get(ENV_TESTSET, "").strip()
    if env_path:
        return env_path
    # 项目根:app.py 上溯两级 services/retrieval_service/ → universal-agent/
    project_root = Path(__file__).resolve().parents[2]
    rel_to_root = project_root / DEFAULT_TESTSET_REL
    if rel_to_root.exists():
        return str(rel_to_root)
    # 兜底:本文件同目录
    return str(Path(__file__).resolve().parent / "测试集.json")


def _load_testset() -> list[dict]:
    """加载测试集;失败返回空列表并由调用方处理错误。"""
    path = _resolve_testset_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("测试集格式非法(非数组): %s", path)
            return []
        logger.info("测试集加载成功 path=%s 条目数=%d", path, len(data))
        return data
    except FileNotFoundError:
        logger.warning("测试集文件不存在: %s", path)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取测试集失败 path=%s err=%r", path, exc)
        return []


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


def _default_model() -> Any:
    """优先按 config.yaml 的 models[].use 构建;失败回退离线模型。

    本仓库裁剪版 uniagent 未内置独立 ModelFactory,故直接经
    ``resolve_class`` 解析 ``use`` 指向的类,并按 ModelConfig 的
    model/temperature/kwargs 构建(与参考实现的工厂语义一致)。
    """
    try:
        from uniagent.config.app_config import get_app_config
        from uniagent.imports.resolvers import resolve_class
        cfg = get_app_config()
        if cfg.models:
            mc = next((m for m in cfg.models if m.name == "default"),
                      cfg.models[0])
            model_cls = resolve_class(mc.use)
            return model_cls(model=mc.model, temperature=mc.temperature,
                             **mc.kwargs)
        logger.warning("config.yaml 未配置 models,使用离线 ScriptedChatModel")
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载模型配置失败(%r),使用离线 ScriptedChatModel", exc)
    return ScriptedChatModel()


def _default_es(model: Any = None) -> ESClient:
    """按环境变量构建生产检索后端 ProduceESClient。

    ProduceESClient 构造本身不发网络请求,故不做异常回退;
    真实失败发生在 keyword_search/vector_search 调用时,由 runner/app 的
    异常路径处理。model 注入供 _extract_keywords 使用。
    """
    region = os.environ.get(ENV_REGION, "000").strip() or "000"
    try:
        ngkm_timeout = int(os.environ.get(ENV_TIMEOUT, str(DEFAULT_NGKM_TIMEOUT_S)))
    except ValueError:
        ngkm_timeout = DEFAULT_NGKM_TIMEOUT_S
    logger.info("检索后端: 生产 ngkm ProduceESClient region=%s timeout=%ss",
                region, ngkm_timeout)
    return ProduceESClient(region_code=region, model=model, timeout=ngkm_timeout)

def create_app(es: Any = None, model: Any = None,
               # timeout_s: float = DEFAULT_TIMEOUT_S,
               # 观测完整请求耗时期间 DEFAULT_TIMEOUT_S=None(wait_for 不设超时)
               timeout_s: Optional[float] = DEFAULT_TIMEOUT_S,
               base_path: Optional[str] = None) -> FastAPI:
    """创建服务应用。

    es/model 可显式注入(测试或生产接真实依赖),缺省按环境变量构建;
    base_path 为业务路由前缀,缺省取环境变量 RETRIEVAL_SERVICE_BASE_PATH,
    再缺省为 DEFAULT_BASE_PATH(/api/retrieval-service/prod)。
    """
    base = _resolve_base_path(base_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or _default_model()
        app.state.es = es or _default_es(app.state.model)
        app.state.timeout_s = timeout_s
        logger.info("retrieval 服务就绪 base=%s model=%s es=%s",
                    base, type(app.state.model).__name__,
                    type(app.state.es).__name__)
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

    @app.get("/health")
    async def health(request: Request) -> dict:
        return {"status": "ok",
                "backend": type(request.app.state.es).__name__}

    @app.get("/debug", response_class=HTMLResponse)
    async def debug_page() -> HTMLResponse:
        """检索调试台:表单构造 /retrieve 请求并展示返回。启动后访问
        http://localhost:8000/debug(端口随 uvicorn --port)。"""
        html_path = Path(__file__).parent / "frontend" / "index.html"
        html = html_path.read_text(encoding="utf-8")
        return html.replace("__BASE_PATH__", base)

    @app.get("/test", response_class=HTMLResponse)
    async def test_page() -> HTMLResponse:
        """召回率测试台(前端页面,后端 /test-recall/stream 暂不可用)。"""
        html_path = Path(__file__).parent / "frontend" / "test.html"
        html = html_path.read_text(encoding="utf-8")
        return html.replace("__BASE_PATH__", base)

    @app.post(f"{base}/retrieve", response_model=RetrievalResponse)
    async def retrieve(payload: RetrievalRequest, request: Request):
        """检索端点:调用 RetrievalSubAgent(GoalLoop 三轮形态)。

        - mode=integrate(缺省):intergrate_all → query_rewrite → intergrate_all
          需要 model 供 LLM 推理与 query_rewrite/关键词提取使用。
        """
        request_id = _request_id(request)
        mode = payload.mode or "integrate"
        vector_mode = payload.vector_mode
        try:
            result = await asyncio.wait_for(
                run_retrieval_request(payload,
                                      es=request.app.state.es,
                                      model=request.app.state.model,
                                      request_id=request_id,
                                      mode=mode,
                                      vector_mode=vector_mode),
                timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("request_id=%s 检索端到端超时 mode=%s", request_id, mode)
            return JSONResponse(error_body(RTN_TIMEOUT, "服务处理超时"))
        except ValueError as exc:
            logger.warning("request_id=%s 非法 mode=%s err=%s", request_id, mode, exc)
            return JSONResponse(error_body(RTN_BAD_REQUEST, str(exc)))
        except Exception:  # noqa: BLE001
            logger.exception("request_id=%s 检索未预期异常 mode=%s", request_id, mode)
            return JSONResponse(error_body(RTN_INTERNAL, "服务内部错误"))

        logger.info("request_id=%s traceId=%s mode=%s outcome=%s degraded=%s "
                    "region=%s recalled=%d elapsedMs=%d",
                    request_id, result.trace_id, mode, result.outcome,
                    result.degraded, result.region_code,
                    result.recalled_count, result.elapsed_ms)
        return RetrievalResponse(rtnCode=RTN_OK, rtnMsg="success", object=result)

    @app.post(f"{base}/retrieve/batch", response_model=BatchRetrievalResponse)
    async def retrieve_batch(payload: BatchRetrievalRequest, request: Request):
        """批量检索端点:从测试集选取前 count 条逐条执行检索并统计命中。

        - count=0 表示选取测试集全部条目;
        - 每条用 RetrievalRequest(query=user_query, region_code=province)走 GoalLoop 三轮;
        - 命中定义:recalled_kids 至少包含测试集标注的一个 knowledge_id;
        - 顺序串行执行(每条独立 Workspace,但共享 es/model,避免并发压力)。
        """
        request_id = _request_id(request)
        testset = _load_testset()
        if not testset:
            return JSONResponse(error_body(RTN_BAD_REQUEST, "测试集为空或读取失败"))

        count = payload.count if payload.count > 0 else len(testset)
        items = testset[:count]
        if not items:
            return JSONResponse(error_body(RTN_BAD_REQUEST, "count=0 且测试集为空"))

        es = request.app.state.es
        model = request.app.state.model
        timeout_s = request.app.state.timeout_s

        t0 = time.perf_counter()
        results: list[BatchRetrievalItem] = []
        hit_count = 0

        for i, item in enumerate(items):
            user_query = str(item.get("user_query") or "").strip()
            province = str(item.get("province") or "000").strip() or "000"
            expected_kids = [str(k) for k in (item.get("knowledge_ids") or [])]

            if not user_query:
                # 跳过空 query(避免 RetrievalRequest 校验失败)
                results.append(BatchRetrievalItem(
                    index=i, user_query="", region_code=province,
                    expected_kids=expected_kids, recalled_kids=[],
                    hit=False, hit_kids=[],
                    outcome="error", elapsed_ms=0, recalled_count=0))
                continue

            sub_req = RetrievalRequest(
                query=user_query,
                region_code=province,
                mode="integrate",
                vector_mode="both",
            )
            sub_req_id = f"{request_id}#{i}"
            try:
                r = await asyncio.wait_for(
                    run_retrieval_request(
                        sub_req,
                        es=es,
                        model=model,
                        request_id=sub_req_id,
                        mode="integrate",
                        vector_mode="both",
                    ),
                    timeout=timeout_s,
                )
                recalled_kids = list(r.kids or [])
                hit_kids = [k for k in expected_kids if k in recalled_kids]
                hit = bool(hit_kids)
                if hit:
                    hit_count += 1
                results.append(BatchRetrievalItem(
                    index=i,
                    user_query=user_query,
                    region_code=province,
                    expected_kids=expected_kids,
                    recalled_kids=recalled_kids,
                    hit=hit,
                    hit_kids=hit_kids,
                    keywords_history=list(r.keywords_history or []),
                    ranked_kids_history=list(r.ranked_kids_history or []),
                    outcome=r.outcome,
                    elapsed_ms=r.elapsed_ms,
                    recalled_count=r.recalled_count,
                ))
                logger.info("batch %s#%d outcome=%s recalled=%d hit=%s",
                            request_id, i, r.outcome, r.recalled_count, hit)
            except asyncio.TimeoutError:
                logger.error("batch %s#%d 检索超时", request_id, i)
                results.append(BatchRetrievalItem(
                    index=i, user_query=user_query, region_code=province,
                    expected_kids=expected_kids, recalled_kids=[],
                    hit=False, hit_kids=[],
                    outcome="timeout", elapsed_ms=0, recalled_count=0))
            except Exception as exc:  # noqa: BLE001
                logger.exception("batch %s#%d 检索异常 err=%r", request_id, i, exc)
                results.append(BatchRetrievalItem(
                    index=i, user_query=user_query, region_code=province,
                    expected_kids=expected_kids, recalled_kids=[],
                    hit=False, hit_kids=[],
                    outcome="error", elapsed_ms=0, recalled_count=0))

        total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
        obj = BatchRetrievalResponseObject(
            total=len(results),
            hit_count=hit_count,
            total_elapsed_ms=total_elapsed_ms,
            results=results,
        )
        logger.info("batch request_id=%s total=%d hit=%d elapsedMs=%d",
                    request_id, obj.total, hit_count, total_elapsed_ms)
        return BatchRetrievalResponse(
            rtnCode=RTN_OK, rtnMsg="success", object=obj)

    # ── 2026-09-18 恢复启用 /retrieve/batch/stream 流式端点 ──
    @app.post(f"{base}/retrieve/batch/stream")
    async def retrieve_batch_stream(payload: BatchRetrievalRequest, request: Request):
        """流式批量检索端点(NDJSON):每完成一条立即 yield,避免网关 504。

        - 同 /retrieve/batch 逻辑,但响应改为 NDJSON 流;
        - 媒体类型: application/x-ndjson; charset=utf-8;
        - 每行一个 JSON,三种类型:
            {"type":"start","total":N,"request_id":"..."}                 // 头部,告知总数
            {"type":"item","data":{...BatchRetrievalItem}}                 // 每条结果
            {"type":"summary","data":{"total":N,"hit_count":K,"total_elapsed_ms":M}}
        - 顺序串行执行(每条独立 Workspace,共享 es/model)。
        """
        request_id = _request_id(request)
        testset = _load_testset()
        if not testset:
            return JSONResponse(error_body(RTN_BAD_REQUEST, "测试集为空或读取失败"))

        count = payload.count if payload.count > 0 else len(testset)
        items = testset[:count]
        if not items:
            return JSONResponse(error_body(RTN_BAD_REQUEST, "count=0 且测试集为空"))

        es = request.app.state.es
        model = request.app.state.model
        timeout_s = request.app.state.timeout_s

        async def stream():
            t0 = time.perf_counter()
            # 先发 start 头,前端立刻知道总数与 request_id
            yield json.dumps(
                {"type": "start", "total": len(items), "request_id": request_id},
                ensure_ascii=False) + "\n"
            hit_count = 0
            for i, item in enumerate(items):
                user_query = str(item.get("user_query") or "").strip()
                province = str(item.get("province") or "000").strip() or "000"
                expected_kids = [str(k) for k in (item.get("knowledge_ids") or [])]

                if not user_query:
                    item_obj = BatchRetrievalItem(
                        index=i, user_query="", region_code=province,
                        expected_kids=expected_kids, recalled_kids=[],
                        hit=False, hit_kids=[],
                        outcome="error", elapsed_ms=0, recalled_count=0)
                    yield json.dumps(
                        {"type": "item", "data": item_obj.model_dump()},
                        ensure_ascii=False) + "\n"
                    continue

                sub_req = RetrievalRequest(
                    query=user_query,
                    region_code=province,
                    mode="integrate",
                    vector_mode="both",
                )
                sub_req_id = f"{request_id}#{i}"
                try:
                    r = await asyncio.wait_for(
                        run_retrieval_request(
                            sub_req,
                            es=es,
                            model=model,
                            request_id=sub_req_id,
                            mode="integrate",
                            vector_mode="both",
                        ),
                        timeout=timeout_s,
                    )
                    recalled_kids = list(r.kids or [])
                    hit_kids = [k for k in expected_kids if k in recalled_kids]
                    hit = bool(hit_kids)
                    if hit:
                        hit_count += 1
                    item_obj = BatchRetrievalItem(
                        index=i,
                        user_query=user_query,
                        region_code=province,
                        expected_kids=expected_kids,
                        recalled_kids=recalled_kids,
                        hit=hit,
                        hit_kids=hit_kids,
                        keywords_history=list(r.keywords_history or []),
                        ranked_kids_history=list(r.ranked_kids_history or []),
                        outcome=r.outcome,
                        elapsed_ms=r.elapsed_ms,
                        recalled_count=r.recalled_count,
                    )
                    logger.info("batch-stream %s#%d outcome=%s recalled=%d hit=%s",
                                request_id, i, r.outcome, r.recalled_count, hit)
                except asyncio.TimeoutError:
                    logger.error("batch-stream %s#%d 检索超时", request_id, i)
                    item_obj = BatchRetrievalItem(
                        index=i, user_query=user_query, region_code=province,
                        expected_kids=expected_kids, recalled_kids=[],
                        hit=False, hit_kids=[],
                        outcome="timeout", elapsed_ms=0, recalled_count=0)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("batch-stream %s#%d 检索异常 err=%r",
                                     request_id, i, exc)
                    item_obj = BatchRetrievalItem(
                        index=i, user_query=user_query, region_code=province,
                        expected_kids=expected_kids, recalled_kids=[],
                        hit=False, hit_kids=[],
                        outcome="error", elapsed_ms=0, recalled_count=0)

                yield json.dumps(
                    {"type": "item", "data": item_obj.model_dump()},
                    ensure_ascii=False) + "\n"

            total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
            summary = BatchRetrievalResponseObject(
                total=len(items),
                hit_count=hit_count,
                total_elapsed_ms=total_elapsed_ms,
                results=[],
            )
            logger.info("batch-stream request_id=%s total=%d hit=%d elapsedMs=%d",
                        request_id, summary.total, hit_count, total_elapsed_ms)
            yield json.dumps(
                {"type": "summary", "data": summary.model_dump()},
                ensure_ascii=False) + "\n"

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                # 禁用 nginx/uvicorn 的流缓冲,保证每条立即推送
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    @app.post(f"{base}/keyword", response_model=RetrievalResponse)
    # 原端点共用 RetrievalRequest(含无关的 vector_mode),关键词路径不走向量召回,
    # 改用不含 vector_mode 的 KeywordRequest
    # async def keyword(payload: RetrievalRequest, request: Request):
    async def keyword(payload: KeywordRequest, request: Request):
        """关键词单路召回:直调 keyword_recall(LLM 关键词提取 → 知识主索引 → 原子表拼接)。

        不走 GoalLoop,单次调用,需要 model 供 LLM 关键词提取使用。
        """
        request_id = _request_id(request)
        try:
            result = await asyncio.wait_for(
                run_keyword_request(payload,
                                    es=request.app.state.es,
                                    model=request.app.state.model,
                                    request_id=request_id),
                timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("request_id=%s keyword 召回超时", request_id)
            return JSONResponse(error_body(RTN_TIMEOUT, "服务处理超时"))
        except Exception:  # noqa: BLE001
            logger.exception("request_id=%s keyword 召回异常", request_id)
            return JSONResponse(error_body(RTN_INTERNAL, "服务内部错误"))

        logger.info("request_id=%s traceId=%s outcome=%s region=%s recalled=%d",
                    request_id, result.trace_id, result.outcome,
                    result.region_code, result.recalled_count)
        return RetrievalResponse(rtnCode=RTN_OK, rtnMsg="success", object=result)

    @app.post(f"{base}/vector", response_model=RetrievalResponse)
    # 原端点共用 RetrievalRequest(含无意义的 mode),向量路径固定走向量召回不做 mode 分发,
    # 改用不含 mode 的 VectorRequest
    # async def vector(payload: RetrievalRequest, request: Request):
    async def vector(payload: VectorRequest, request: Request):
        """向量单路召回:直调 vector_recall(在线 embedding 向量检索)。

        不走 GoalLoop,单次调用,不经关键词/槽位提取。
        """
        request_id = _request_id(request)
        vector_mode = payload.vector_mode
        try:
            result = await asyncio.wait_for(
                run_vector_request(payload,
                                   es=request.app.state.es,
                                   model=request.app.state.model,
                                   request_id=request_id,
                                   vector_mode=vector_mode),
                timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("request_id=%s vector 召回超时", request_id)
            return JSONResponse(error_body(RTN_TIMEOUT, "服务处理超时"))
        except Exception:  # noqa: BLE001
            logger.exception("request_id=%s vector 召回异常", request_id)
            return JSONResponse(error_body(RTN_INTERNAL, "服务内部错误"))

        logger.info("request_id=%s traceId=%s outcome=%s region=%s recalled=%d",
                    request_id, result.trace_id, result.outcome,
                    result.region_code, result.recalled_count)
        return RetrievalResponse(rtnCode=RTN_OK, rtnMsg="success", object=result)


# 默认应用实例:python -m uvicorn retrieval_service.app:app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
