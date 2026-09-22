# -*- coding: utf-8 -*-
"""llm_397b 服务 — 灵犀大模型网关直通透传的最小 FastAPI 封装。

不做任何智能体编排:POST /llm_397b_api 接收用户问题(query),
按 config.yaml 的 models[] 配置调用灵犀网关
(生产路径: kbagent.shared.lingxi_provider:LingxiSSLChatOpenAI,
模型 Qwen3.5-397B-A17B-FP8),一次性返回大模型预测 token 文本。

与 answer_service 一致:模型配置缺失/解析失败时**启动即报错**,
不静默回退离线 ScriptedChatModel,避免"以为在调真实模型,实际跑的 mock"。

启动(在 universal-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    Windows:  set PYTHONPATH=src;services && .venv\\Scripts\\python -m uvicorn llm_service.app:app --host 0.0.0.0 --port 8002
    Linux:    PYTHONPATH=src:services python -m uvicorn llm_service.app:app --host 0.0.0.0 --port 8002

请求/响应示例:
    POST /llm_397b_api
    {"query": "5G套餐怎么办理?"}
    →
    {"rtnCode": "0", "rtnMsg": "success",
     "object": {"requestId": "3f2b...", "answer": "预测文本...",
                "model": "Qwen3.5-397B-A17B-FP8", "elapsedMs": 1234}}

并发模型:
    - model 全局共享(LingxiSSLChatOpenAI 内部 httpx client 线程安全)
    - 同步 model.invoke 经 asyncio.to_thread 放入线程池,不阻塞事件循环;
      端到端用 wait_for 限时,超时返回 50002
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from .models import (
    RTN_BAD_REQUEST,
    RTN_INTERNAL,
    RTN_OK,
    RTN_TIMEOUT,
    LlmObject,
    LlmRequest,
    LlmResponse,
    error_body,
)

logger = logging.getLogger("llm_service")

# 端到端超时(秒):内网网关单次裸调用可达分钟级
# (config.yaml 中模型 timeout: 600),服务层兜底需略大于网关超时
DEFAULT_TIMEOUT_S = 660.0


def _default_model() -> Any:
    """按 config.yaml 的 models[].use 构建真实大模型。

    **不回退** ScriptedChatModel:本服务的用途就是灵犀直通,
    配置有问题宁可启动失败。
    """
    from uniagent.config.app_config import get_app_config
    from uniagent.imports.resolvers import resolve_class

    cfg = get_app_config()
    if not cfg.models:
        raise RuntimeError(
            "config.yaml 未配置 models — llm_service 必须使用真实大模型,"
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


def _model_name(model: Any) -> str:
    """尽力取模型名(ChatOpenAI 用 model_name,基类兜底 model)。"""
    for attr in ("model_name", "model"):
        value = getattr(model, attr, "")
        if isinstance(value, str) and value:
            return value
    return ""


def _text_of(content: Any) -> str:
    """AIMessage.content 可能是 str 或内容块列表,统一为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)


def create_app(model: Any = None,
               timeout_s: float = DEFAULT_TIMEOUT_S) -> FastAPI:
    """创建服务应用。

    model 可显式注入(单测注入 mock);缺省按 config.yaml 构建真实大模型,
    在 lifespan 启动阶段完成,配置有问题启动即报错。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or _default_model()
        app.state.timeout_s = timeout_s
        app.state.model_name = _model_name(app.state.model)
        logger.info("llm_397b 服务就绪 model=%s timeout=%ss",
                    app.state.model_name or type(app.state.model).__name__,
                    timeout_s)
        yield

    app = FastAPI(title="llm-397b-service", version="1.0.0", lifespan=lifespan)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:

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

    @app.post("/llm_397b_api", response_model=LlmResponse)
    async def llm_397b_api(req: LlmRequest, request: Request):
        request_id = uuid.uuid4().hex
        model = request.app.state.model
        logger.info("requestId=%s 收到请求 queryLen=%d", request_id, len(req.query))
        started = time.perf_counter()

        def _invoke() -> Any:
            return model.invoke([HumanMessage(content=req.query)])

        try:
            # 同步 invoke 放线程池,避免阻塞事件循环;wait_for 端到端限时
            result = await asyncio.wait_for(
                asyncio.to_thread(_invoke),
                timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("requestId=%s 大模型调用超时(%ss)",
                         request_id, request.app.state.timeout_s)
            return JSONResponse(error_body(RTN_TIMEOUT, "大模型调用超时"))
        except Exception:  # noqa: BLE001
            logger.exception("requestId=%s 大模型调用异常", request_id)
            return JSONResponse(error_body(RTN_INTERNAL, "大模型网关调用失败"))

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        answer = _text_of(getattr(result, "content", ""))
        usage = getattr(result, "usage_metadata", None)
        logger.info("requestId=%s elapsedMs=%s answerLen=%d usage=%s",
                    request_id, elapsed_ms, len(answer), usage or "-")
        return LlmResponse(
            rtnCode=RTN_OK, rtnMsg="success",
            object=LlmObject(requestId=request_id, answer=answer,
                             model=request.app.state.model_name,
                             elapsedMs=elapsed_ms))


# 默认应用实例:python -m uvicorn llm_service.app:app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8002)
