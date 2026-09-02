# -*- coding: utf-8 -*-
"""kbagent 服务层 — 灵犀契约的 FastAPI 封装。

启动(在 knowbase-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    Windows:  set PYTHONPATH=src;services && python -m uvicorn kbagent_service.app:app --host 0.0.0.0 --port 8000
    Linux:    PYTHONPATH=src:services python -m uvicorn kbagent_service.app:app --host 0.0.0.0 --port 8000

并发模型:
    - model / es 全局共享(ModelFactory 自带实例缓存;ES client 需线程安全)
    - MainAgent 持有实例状态(tracer),**每请求新建实例**;
      工作区为 ContextVar,asyncio 每个请求任务天然隔离
    - 在事件循环内必须用 arun(),不能用 run()(内部 asyncio.run 会与现有循环冲突)

多轮策略:
    kbagent 为单轮接口,取 conversations 中最后一条用户消息作为 query;
    不拼接历史,避免稀释检索关键词。

生产依赖:
    - model: config.yaml 配置了 models 时经 models[].use 解析构建;否则回退离线 ScriptedChatModel
    - es:    默认 MockESClient(离线);生产请 create_app(es=...) 注入真实 ESClient 实现
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kbagent import MainAgent, MockESClient, ScriptedChatModel
from kbagent.models import FinalAnswer

from .models import (
    RTN_BAD_REQUEST,
    RTN_INTERNAL,
    RTN_OK,
    RTN_TIMEOUT,
    AnswerObject,
    AskParams,
    AskRequest,
    AskResponse,
    SourceItem,
    error_body,
)

logger = logging.getLogger("kbagent_service")

# 端到端超时(秒):略大于 Config.budget["end_to_end"]=4000ms
DEFAULT_TIMEOUT_S = 120.0
# appId 白名单环境变量:逗号分隔;为空则全部放行
ENV_APP_IDS = "KB_SERVICE_APP_IDS"
# 技能包目录:按本文件位置定位到仓库根,不依赖启动时 CWD
_SKILLS_DIR = str(Path(__file__).resolve().parents[2] / "skills")

# ── 路由前缀 ────────────────────────────────────────────────────────────────
# 形如 /api/{服务名}/{环境},业务端点挂在其下(如 {base}/retrieve)。
# 可经环境变量 KB_SERVICE_BASE_PATH 覆盖(如 /api/kb-agent-service/test-prod)。
ENV_BASE_PATH = "KB_SERVICE_BASE_PATH"
DEFAULT_BASE_PATH = "/api/kb-agent-service/prod"


def _resolve_base_path(base_path: Optional[str]) -> str:
    """规范化路由前缀:确保以 / 开头、无尾部 /。"""
    bp = (base_path or os.environ.get(ENV_BASE_PATH) or DEFAULT_BASE_PATH).strip()
    if not bp.startswith("/"):
        bp = "/" + bp
    return bp.rstrip("/")


def _now_str() -> str:
    """收到请求时间,格式 yyyy-MM-dd HH:mm:ss.SSS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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


def _default_es() -> Any:
    logger.warning("未注入 ES 客户端,使用离线 MockESClient(生产环境必须替换)")
    return MockESClient()


def create_app(model: Any = None, es: Any = None,
               timeout_s: float = DEFAULT_TIMEOUT_S,
               base_path: Optional[str] = None) -> FastAPI:
    """创建服务应用。

    model / es 可显式注入(测试或生产接真实依赖);
    base_path 为业务路由前缀,缺省取环境变量 KB_SERVICE_BASE_PATH,
    再缺省为 DEFAULT_BASE_PATH(/api/kb-agent-service/prod)。
    """
    base = _resolve_base_path(base_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or _default_model()
        app.state.es = es or _default_es()
        app.state.timeout_s = timeout_s
        logger.info("kbagent 服务就绪 base=%s model=%s es=%s skills=%s",
                    base, type(app.state.model).__name__,
                    type(app.state.es).__name__, _SKILLS_DIR)
        yield

    app = FastAPI(title="kbagent-service", version="1.0.0", lifespan=lifespan)
    _register_routes(app, base)
    return app


def _register_routes(app: FastAPI, base: str) -> None:

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError):
        detail = "; ".join(
            f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg', '')}"
            for e in exc.errors()[:5]
        )
        return JSONResponse(error_body(RTN_BAD_REQUEST, f"参数错误: {detail}"))

    # 健康检查固定在根路径,供网关/负载均衡探活
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post(f"{base}/retrieve", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request):
        arrived = _now_str()
        p = req.params

        allowed = {s.strip()
                   for s in os.environ.get(ENV_APP_IDS, "").split(",")
                   if s.strip()}
        if allowed and p.appId not in allowed:
            return JSONResponse(error_body(RTN_BAD_REQUEST, "appId 不允许"))

        query = _extract_query(p)
        if query is None:
            return JSONResponse(error_body(
                RTN_BAD_REQUEST, "conversations 中无有效用户消息(role=1)"))

        try:
            agent = MainAgent(model=request.app.state.model,
                              es=request.app.state.es,
                              skill_dirs=[_SKILLS_DIR])
            ans = await asyncio.wait_for(agent.arun(query),
                                         timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("requestId=%s 端到端超时", p.requestId)
            return JSONResponse(error_body(RTN_TIMEOUT, "服务处理超时"))
        except Exception:  # noqa: BLE001
            logger.exception("requestId=%s 未预期异常", p.requestId)
            return JSONResponse(error_body(RTN_INTERNAL, "服务内部错误"))

        logger.info("requestId=%s traceId=%s degraded=%s elapsedMs=%s sources=%d",
                    p.requestId, ans.trace_id, ans.degraded,
                    ans.elapsed_ms, len(ans.sources))
        return AskResponse(rtnCode=RTN_OK, rtnMsg="success",
                           object=_to_object(ans, p, arrived))


def _extract_query(p: AskParams) -> Optional[str]:
    """多轮对话 → 单轮 query:取最后一条用户消息(不拼接历史)。"""
    for conv in reversed(p.conversations):
        if conv.role == 1 and conv.content.strip():
            return conv.content.strip()
    return None


def _to_object(ans: FinalAnswer, p: AskParams, arrived: str) -> AnswerObject:
    """FinalAnswer → 灵犀 object 层。"""
    return AnswerObject(
        requestId=p.requestId,
        sessionId=p.sessionId,
        traceId=ans.trace_id,
        requestArrivedTime=arrived,
        degraded=ans.degraded,
        elapsedMs=ans.elapsed_ms,
        businessExplanation=ans.business_explanation or "",
        handlingSuggestion=ans.handling_suggestion or "",
        renderedText=ans.render(),
        sources=[
            SourceItem(chunkId=s.chunk_id, docTitle=s.doc_title,
                       snippet=s.snippet, updatedAt=s.updated_at,
                       stale=s.stale)
            for s in ans.sources
        ],
    )


# 默认应用实例:python -m uvicorn kbagent_service.app:app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
