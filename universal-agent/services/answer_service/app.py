# -*- coding: utf-8 -*-
"""answer 测试服务 — 只跑答案子智能体(真实大模型)的 FastAPI 封装。

与 kbagent_service 的区别:
    - 不做检索/处理:请求直接携带 query + chunks(检索/处理阶段的输出)
    - 只运行 kbagent.answer:select_fragments 精选 → LLM 组织答案 → 逐句锚定校验
    - 模型固定走真实网关:按 config.yaml 的 models[].use 解析
      (生产路径: kbagent.shared.lingxi_provider:LingxiSSLChatOpenAI),
      配置缺失/解析失败时**启动即报错**,不静默回退离线 ScriptedChatModel,
      避免"以为在测真实模型,实际跑的 mock"

启动(在 universal-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    Windows:  set PYTHONPATH=src;services && .venv\\Scripts\\python -m uvicorn answer_service.app:app --host 0.0.0.0 --port 8001
    Linux:    PYTHONPATH=src:services python -m uvicorn answer_service.app:app --host 0.0.0.0 --port 8001

并发模型:
    - model 全局共享(LingxiSSLChatOpenAI 内部 httpx client 线程安全)
    - AnswerSubAgent 持有 tracer,**每请求新建实例**
    - generate() 内部为同步 model.invoke,经 asyncio.to_thread 放入线程池,
      不阻塞事件循环;端到端用 wait_for 限时,超时返回 50002

排障端点(与 kbagent_service 一致):
    /diag          查看实际加载的模型类、密钥注入状态(脱敏)
    /diag/gateway  用当前配置里的 key 实测大模型网关,返回网关原始响应
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kbagent.answer.agent import AnswerSubAgent
from kbagent.shared.config import DEFAULT_CONFIG
from kbagent.shared.models import Chunk
from kbagent.shared.tracing import Tracer

from .models import (
    RTN_BAD_REQUEST,
    RTN_INTERNAL,
    RTN_OK,
    RTN_TIMEOUT,
    AnswerObject,
    AnswerParams,
    AnswerRequest,
    AnswerResponse,
    SentenceItem,
    SourceItem,
    error_body,
)

logger = logging.getLogger("answer_service")

# 端到端超时(秒):答案生成 1 次 + 每句锚定校验各 1 次真实大模型调用,
# 内网网关单次调用可达分钟级,故默认给足 300s
DEFAULT_TIMEOUT_S = 300.0
# appId 白名单环境变量:逗号分隔;为空则全部放行
ENV_APP_IDS = "ANSWER_SERVICE_APP_IDS"

# ── 路由前缀 ────────────────────────────────────────────────────────────────
# 形如 /api/{服务名}/{环境},业务端点挂在其下(如 {base}/answer)。
# 可经环境变量 ANSWER_SERVICE_BASE_PATH 覆盖。
ENV_BASE_PATH = "ANSWER_SERVICE_BASE_PATH"
DEFAULT_BASE_PATH = "/api/answer-service/prod"


def _resolve_base_path(base_path: Optional[str]) -> str:
    """规范化路由前缀:确保以 / 开头、无尾部 /。"""
    bp = (base_path or os.environ.get(ENV_BASE_PATH) or DEFAULT_BASE_PATH).strip()
    if not bp.startswith("/"):
        bp = "/" + bp
    return bp.rstrip("/")


def _now_str() -> str:
    """收到请求时间,格式 yyyy-MM-dd HH:mm:ss.SSS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _mask_secret(value: Any) -> str:
    """脱敏展示密钥:仅保留首尾 4 位;未展开的占位符原样标出。"""
    s = str(value) if value is not None else ""
    if not s:
        return "<empty>"
    if s.startswith("${"):
        return s + "  ← 占位符未展开(环境变量未注入)!"
    return (s[:4] + "****" + s[-4:]) if len(s) > 8 else "****"


def _default_model() -> Any:
    """按 config.yaml 的 models[].use 构建真实大模型。

    与 kbagent_service 不同:此处**不回退** ScriptedChatModel。
    本服务的用途就是验证真实模型下的答案质量,配置有问题宁可启动失败。
    """
    from uniagent.config.app_config import get_app_config
    from uniagent.imports.resolvers import resolve_class

    cfg = get_app_config()
    if not cfg.models:
        raise RuntimeError(
            "config.yaml 未配置 models — answer_service 必须使用真实大模型,"
            "请在 config.yaml 配置(参考: use: "
            "\"kbagent.shared.lingxi_provider:LingxiSSLChatOpenAI\")")
    mc = next((m for m in cfg.models if m.name == "default"), cfg.models[0])
    try:
        model_cls = resolve_class(mc.use)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"models.use={mc.use!r} 解析失败: {exc} — "
            "请检查导入路径(灵犀 SSL Provider 实际位于 "
            "kbagent.shared.lingxi_provider)") from exc
    model = model_cls(model=mc.model, temperature=mc.temperature, **mc.kwargs)
    logger.info("真实大模型就绪 use=%s model=%s", mc.use, mc.model)
    return model


def create_app(model: Any = None,
               timeout_s: float = DEFAULT_TIMEOUT_S,
               base_path: Optional[str] = None) -> FastAPI:
    """创建服务应用。

    model 可显式注入(单测注入 mock);缺省按 config.yaml 构建真实大模型。
    base_path 为业务路由前缀,缺省取环境变量 ANSWER_SERVICE_BASE_PATH,
    再缺省为 DEFAULT_BASE_PATH(/api/answer-service/prod)。
    """
    base = _resolve_base_path(base_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or _default_model()
        app.state.timeout_s = timeout_s
        app.state.config = DEFAULT_CONFIG
        logger.info("answer 测试服务就绪 base=%s model=%s timeout=%ss",
                    base, type(app.state.model).__name__, timeout_s)
        yield

    app = FastAPI(title="answer-service", version="1.0.0", lifespan=lifespan)
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

    # ── 排障端点:无 shell/容器权限时经 HTTP 自检 ──────────────────────────
    @app.get("/diag")
    async def diag(request: Request) -> dict:
        info: dict = {
            "model_class": type(request.app.state.model).__name__,
            "timeout_s": request.app.state.timeout_s,
            "env_QWEN_API_KEY": "set" if os.environ.get("QWEN_API_KEY") else "MISSING",
        }
        try:
            from uniagent.config.app_config import get_app_config
            cfg = get_app_config()
            if cfg.models:
                mc = next((m for m in cfg.models if m.name == "default"),
                          cfg.models[0])
                key = mc.kwargs.get("api_key")
                info["config"] = {
                    "use": mc.use,
                    "model": mc.model,
                    "base_url": str(mc.kwargs.get("base_url") or ""),
                    "api_key": _mask_secret(key),
                    "api_key_expanded": not (isinstance(key, str)
                                             and key.startswith("${")),
                }
            else:
                info["config"] = {"models": "未配置"}
        except Exception as exc:  # noqa: BLE001
            info["config"] = {"error": repr(exc)}
        return info

    @app.get("/diag/gateway")
    async def diag_gateway() -> dict:
        try:
            import httpx
            from uniagent.config.app_config import get_app_config
            cfg = get_app_config()
            if not cfg.models:
                return {"error": "未配置 models"}
            mc = cfg.models[0]
            base_url = str(mc.kwargs.get("base_url") or "").rstrip("/")
            if not base_url:
                return {"error": "base_url 未配置"}
            key = str(mc.kwargs.get("api_key") or "")
            async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {key}"})
            return {"status_code": resp.status_code,
                    "body": resp.text[:800]}
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    @app.post(f"{base}/answer", response_model=AnswerResponse)
    async def answer(req: AnswerRequest, request: Request):
        arrived = _now_str()
        p = req.params

        allowed = {s.strip()
                   for s in os.environ.get(ENV_APP_IDS, "").split(",")
                   if s.strip()}
        if allowed and p.appId not in allowed:
            return JSONResponse(error_body(RTN_BAD_REQUEST, "appId 不允许"))

        chunks = [_to_chunk(c) for c in p.chunks]
        tracer = Tracer()
        tracer.log("request", "received", requestId=p.requestId,
                   query=p.query, chunk_count=len(chunks))

        try:
            # AnswerSubAgent 持有实例状态,每请求新建;
            # 内部 model.invoke 为同步调用,放线程池避免阻塞事件循环
            agent = AnswerSubAgent(request.app.state.model,
                                   request.app.state.config, tracer)
            ans = await asyncio.wait_for(
                asyncio.to_thread(agent.run, p.query, chunks, tracer.trace_id),
                timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("requestId=%s 答案生成超时(%ss)",
                         p.requestId, request.app.state.timeout_s)
            return JSONResponse(error_body(RTN_TIMEOUT, "答案生成超时"))
        except Exception:  # noqa: BLE001
            logger.exception("requestId=%s 未预期异常", p.requestId)
            return JSONResponse(error_body(RTN_INTERNAL, "服务内部错误"))

        ans.elapsed_ms = tracer.elapsed_ms()
        tracer.log("finalize", "done", elapsed_ms=ans.elapsed_ms,
                   sentence_count=len(ans.sentences),
                   source_count=len(ans.sources))
        logger.info("requestId=%s traceId=%s elapsedMs=%s sentences=%d sources=%d",
                    p.requestId, ans.trace_id, ans.elapsed_ms,
                    len(ans.sentences), len(ans.sources))
        return AnswerResponse(rtnCode=RTN_OK, rtnMsg="success",
                              object=_to_object(ans, p, arrived, tracer))


def _to_chunk(c: Any) -> Chunk:
    """契约层 ChunkIn → 内部 Chunk 数据结构。"""
    return Chunk(
        chunk_id=c.chunkId, doc_id=c.docId, doc_title=c.docTitle,
        content=c.content, category=c.category, position=c.position,
        version=c.version, updated_at=c.updatedAt, score=c.score,
    )


def _to_object(ans: Any, p: AnswerParams, arrived: str,
               tracer: Tracer) -> AnswerObject:
    """FinalAnswer → object 层(附带全链路 trace 便于 badcase 回放)。"""
    return AnswerObject(
        requestId=p.requestId,
        sessionId=p.sessionId,
        traceId=ans.trace_id,
        requestArrivedTime=arrived,
        elapsedMs=ans.elapsed_ms,
        businessExplanation=ans.business_explanation or "",
        handlingSuggestion=ans.handling_suggestion or "",
        renderedText=ans.render(),
        sentences=[
            SentenceItem(text=s.text, citations=s.citations,
                         hardFact=s.hard_fact, anchored=s.anchored,
                         note=s.note)
            for s in ans.sentences
        ],
        sources=[
            SourceItem(chunkId=s.chunk_id, docTitle=s.doc_title,
                       snippet=s.snippet, updatedAt=s.updated_at,
                       stale=s.stale)
            for s in ans.sources
        ],
        trace=json.loads(tracer.export()),
    )


# 默认应用实例:python -m uvicorn answer_service.app:app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8001)
