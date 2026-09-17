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
    - es:    缺省即 ProduceESClient(生产 ngkm 一体化流水线,intergrate_all 可用);
             仅 KB_SERVICE_ES=mock 时回退离线 MockESClient。
             也可 create_app(es=...) 显式注入覆盖。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from kbagent import MainAgent, MockESClient, ProduceESClient, ScriptedChatModel
from kbagent.shared.models import FinalAnswer

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
    UsabilityInfo,
    error_body,
)

logger = logging.getLogger("kbagent_service")


def _setup_logging() -> None:
    """配置根日志器,保证每请求 INFO 行可见。

    经 ``python -m uvicorn`` 启动时根日志器默认无 handler,INFO 会被丢弃、
    只有 WARNING+ 经 lastResort 落到 stderr(这正是生产只看到两条启动
    warning 的原因)。级别可用环境变量 KB_SERVICE_LOG_LEVEL 覆盖。
    """
    level = getattr(logging, os.environ.get("KB_SERVICE_LOG_LEVEL", "INFO").upper(),
                    logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    else:
        root.setLevel(level)


_setup_logging()

# 端到端超时(秒):略大于 Config.budget["end_to_end"]=120000ms;
# 覆盖内网大模型多轮调用(检索循环 + 答案生成),超时返回 50002
DEFAULT_TIMEOUT_S = 120.0
# appId 白名单环境变量:逗号分隔;为空则全部放行
ENV_APP_IDS = "KB_SERVICE_APP_IDS"
# 检索后端选择:默认 produce=生产 ngkm 一体化流水线(ProduceESClient);
# 仅当显式设 KB_SERVICE_ES=mock 时才用离线 MockESClient(内置 7 条样例,本地演示)
ENV_ES_BACKEND = "KB_SERVICE_ES"
# ProduceESClient 缺省区域(请求未携带省份时使用);支持省份名或区号
ENV_ES_REGION = "KB_SERVICE_ES_REGION"
# 是否在响应 object.processTrace 中透出智能体执行 trace(前端演示页
# "检索处理过程"时间线用);默认开启,设 KB_SERVICE_EXPOSE_TRACE=0 关闭
ENV_EXPOSE_TRACE = "KB_SERVICE_EXPOSE_TRACE"
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


def _mask_secret(value: Any) -> str:
    """脱敏展示密钥:仅保留首尾 4 位;未展开的占位符原样标出。"""
    s = str(value) if value is not None else ""
    if not s:
        return "<empty>"
    if s.startswith("${"):
        return s + "  ← 占位符未展开(环境变量未注入)!"
    return (s[:4] + "****" + s[-4:]) if len(s) > 8 else "****"


def _trace_dump(agent: Optional[MainAgent]) -> str:
    """导出请求级全链路追踪 JSON;agent 未及创建时给出占位说明。"""
    if agent is None:
        return "<MainAgent 未初始化>"
    return agent.tracer.export()


def _expose_trace() -> bool:
    """是否向调用方透出 processTrace(默认开;0/false/no 关闭)。"""
    return os.environ.get(ENV_EXPOSE_TRACE, "1").strip().lower() \
        not in ("0", "false", "no")


def _jsonable(v: Any) -> Any:
    """payload 值 JSON 安全化:原生类型原样,容器递归,其余 str() 兜底。

    tracer payload 里可能混入 Chunk/Pydantic 对象等非 JSON 原生类型,
    直接塞进响应会让 pydantic 序列化失败,统一在这里降级为字符串。
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _trace_events(agent: Optional[MainAgent]) -> list:
    """智能体 trace 事件列表,供响应 object.processTrace 透出。

    与 _trace_dump(落服务日志的完整 JSON)同源;此处输出结构化 list,
    前端演示页据此渲染"检索处理过程"时间线。未暴露/未初始化时为空。
    """
    if agent is None or not _expose_trace():
        return []
    return [
        {"ts_ms": e.ts_ms, "stage": e.stage, "event": e.event,
         "payload": _jsonable(e.payload)}
        for e in agent.tracer.events
    ]


def _sse(msg: dict) -> str:
    """SSE 帧:data: <json>\\n\\n(前端按空行分帧解析)。"""
    return "data: " + json.dumps(msg, ensure_ascii=False, default=str) + "\n\n"


# 流式接口轮询 tracer 新事件的间隔(秒);越小前端进度越细腻,CPU 略增
STREAM_POLL_INTERVAL_S = 0.15


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
    """按环境变量选择检索后端;**缺省即生产后端**。

    默认/KB_SERVICE_ES=produce → ProduceESClient(生产 ngkm 一体化流水线:
    槽位提取 → 知识主索引召回 → 原子表拼接,intergrate_all 的
    full_recall 必然可用,不会再落入"后端不支持"分支);
    仅 KB_SERVICE_ES=mock → 离线 MockESClient(内置样例,本地演示)。
    """
    backend = os.environ.get(ENV_ES_BACKEND, "").strip().lower()
    if backend == "mock":
        logger.warning("KB_SERVICE_ES=mock: 使用离线 MockESClient(仅 7 条内置样例)")
        return MockESClient()
    region = os.environ.get(ENV_ES_REGION, "000").strip() or "000"
    logger.info("检索后端: 生产 ngkm ProduceESClient region=%s", region)
    return ProduceESClient(region_code=region)


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
    _mount_frontend(app, base)
    return app


def _mount_frontend(app: FastAPI, base: str) -> None:
    """托管前端演示页(static/index.html);目录不存在时静默跳过。

    - 页面挂在两个路径:
        /            直连服务时访问
        {base}/ui    经网关访问时(网关可能只转发 {base} 前缀)
      两处共用同一静态目录,内容一致
    - frontend-config 告知页面实际业务路由前缀(兼容 KB_SERVICE_BASE_PATH
      覆盖);同样注册两个路径,页面 JS 用相对地址请求,两种入口都能命中
    - "/" 挂载必须最后注册:StaticFiles 会兜底所有未匹配路径,
      先注册的 /health、/diag、{base}/retrieve 不受影响
    - 纯静态单文件,零外部依赖/零 CDN,适配无外网生产环境
    """
    static_dir = Path(__file__).resolve().parent / "static"
    if not static_dir.is_dir():
        return

    async def frontend_config() -> dict:
        return {"basePath": base}

    app.add_api_route("/frontend-config", frontend_config, methods=["GET"])
    app.add_api_route(f"{base}/ui/frontend-config", frontend_config,
                      methods=["GET"])

    app.mount(f"{base}/ui", StaticFiles(directory=str(static_dir), html=True),
              name="ui-gateway")
    app.mount("/", StaticFiles(directory=str(static_dir), html=True),
              name="ui-root")
    logger.info("前端演示页已挂载: / 与 %s/ui (static=%s)", base, static_dir)


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
    # /diag         查看实际加载的模型类、密钥注入状态(脱敏)
    # /diag/gateway 用当前配置里的 key 实测大模型网关,返回网关原始响应
    # 注意:排障完成后建议移除或加鉴权
    @app.get("/diag")
    async def diag(request: Request) -> dict:
        info: dict = {
            "model_class": type(request.app.state.model).__name__,
            "es_class": type(request.app.state.es).__name__,
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
                return {"error": "base_url 未配置(可能是 ScriptedChatModel 离线模式)"}
            key = str(mc.kwargs.get("api_key") or "")
            async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
                resp = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {key}"})
            return {"status_code": resp.status_code,
                    "body": resp.text[:800]}
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

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
        logger.info("requestId=%s 收到请求 query=%r province=%r",
                    p.requestId, query, p.userInfo.province)

        agent: Optional[MainAgent] = None
        try:
            agent = MainAgent(model=request.app.state.model,
                              es=request.app.state.es,
                              skill_dirs=[_SKILLS_DIR])
            # 省份信息经 region_code 下传:检索一体化流水线据此选省级索引
            ans = await asyncio.wait_for(
                agent.arun(query, region_code=p.userInfo.province),
                timeout=request.app.state.timeout_s)
        except asyncio.TimeoutError:
            logger.error("requestId=%s 端到端超时,已执行链路:\n%s",
                         p.requestId, _trace_dump(agent))
            return JSONResponse(error_body(
                RTN_TIMEOUT, "服务处理超时", _trace_events(agent)))
        except Exception:  # noqa: BLE001
            logger.exception("requestId=%s 未预期异常,已执行链路:\n%s",
                             p.requestId, _trace_dump(agent))
            return JSONResponse(error_body(
                RTN_INTERNAL, "服务内部错误", _trace_events(agent)))

        logger.info("requestId=%s traceId=%s degraded=%s elapsedMs=%s sources=%d",
                    p.requestId, ans.trace_id, ans.degraded,
                    ans.elapsed_ms, len(ans.sources))
        # 全链路追踪落日志:缓存判定 / 每轮检索 DSL 与召回 / 降级原因 /
        # 处理快照 / 答案素材与锚定校验,排查"无检索结果"看这一段即可
        logger.info("requestId=%s 链路追踪:\n%s", p.requestId, _trace_dump(agent))
        return AskResponse(rtnCode=RTN_OK, rtnMsg="success",
                           object=_to_object(ans, p, arrived,
                                             _trace_events(agent)))

    @app.post(f"{base}/retrieve/stream")
    async def ask_stream(req: AskRequest, request: Request):
        """SSE 流式版检索:边执行边推送 trace 事件,前端实时展示进行到哪一步。

        帧格式(每条 data: 一个 JSON,空行分帧):
            {"type": "trace", "ts_ms":…, "stage":…, "event":…, "payload":…}
                — 执行过程中逐条推送(KB_SERVICE_EXPOSE_TRACE=0 时不推)
            {"type": "final", "response": <与 /retrieve 完全相同的响应信封>}
                — 结束帧(成功/超时/内部错误都有,前端据此收尾渲染)

        与 /retrieve 相同的契约校验与响应组装;仅传输方式不同。
        """
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
        logger.info("requestId=%s 收到流式请求 query=%r province=%r",
                    p.requestId, query, p.userInfo.province)

        timeout_s = request.app.state.timeout_s
        expose = _expose_trace()

        async def event_gen():
            # 每请求新建 MainAgent(实例持有 tracer,不可并发复用)
            agent = MainAgent(model=request.app.state.model,
                              es=request.app.state.es,
                              skill_dirs=[_SKILLS_DIR])
            task = asyncio.ensure_future(
                agent.arun(query, region_code=p.userInfo.province))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_s
            current_tracer = None
            sent = 0
            timed_out = False

            def new_frames() -> list:
                """自上次推送以来 tracer 新增的事件帧。"""
                nonlocal current_tracer, sent
                tracer = agent.tracer
                if tracer is not current_tracer:   # arun 启动时会重建 Tracer
                    current_tracer, sent = tracer, 0
                frames = []
                events = current_tracer.events if current_tracer else []
                while sent < len(events):
                    e = events[sent]
                    sent += 1
                    frames.append(_sse({
                        "type": "trace", "ts_ms": e.ts_ms, "stage": e.stage,
                        "event": e.event, "payload": _jsonable(e.payload)}))
                return frames

            while True:
                if expose:
                    for frame in new_frames():
                        yield frame
                if task.done():
                    break
                if loop.time() > deadline:
                    timed_out = True
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
                    break
                await asyncio.sleep(STREAM_POLL_INTERVAL_S)
            if expose:
                for frame in new_frames():     # 收尾:推完剩余事件
                    yield frame

            if timed_out:
                logger.error("requestId=%s 流式端到端超时,已执行链路:\n%s",
                             p.requestId, _trace_dump(agent))
                yield _sse({"type": "final", "response": error_body(
                    RTN_TIMEOUT, "服务处理超时", _trace_events(agent))})
                return
            try:
                ans = task.result()
            except Exception:  # noqa: BLE001
                logger.exception("requestId=%s 流式未预期异常,已执行链路:\n%s",
                                 p.requestId, _trace_dump(agent))
                yield _sse({"type": "final", "response": error_body(
                    RTN_INTERNAL, "服务内部错误", _trace_events(agent))})
                return
            logger.info("requestId=%s traceId=%s degraded=%s elapsedMs=%s "
                        "sources=%d (流式)",
                        p.requestId, ans.trace_id, ans.degraded,
                        ans.elapsed_ms, len(ans.sources))
            logger.info("requestId=%s 链路追踪:\n%s",
                        p.requestId, _trace_dump(agent))
            resp = AskResponse(rtnCode=RTN_OK, rtnMsg="success",
                               object=_to_object(ans, p, arrived,
                                                 _trace_events(agent)))
            yield _sse({"type": "final", "response": resp.model_dump(mode="json")})

        return StreamingResponse(
            event_gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-store",
                     "X-Accel-Buffering": "no"})   # 反向代理不缓冲,逐帧透传


def _extract_query(p: AskParams) -> Optional[str]:
    """多轮对话 → 单轮 query:取最后一条用户消息(不拼接历史)。"""
    for conv in reversed(p.conversations):
        if conv.role == 1 and conv.content.strip():
            return conv.content.strip()
    return None


def _to_object(ans: FinalAnswer, p: AskParams, arrived: str,
               trace_events: Optional[list] = None) -> AnswerObject:
    """FinalAnswer → 灵犀 object 层。"""
    return AnswerObject(
        requestId=p.requestId,
        sessionId=p.sessionId,
        traceId=ans.trace_id,
        requestArrivedTime=arrived,
        degraded=ans.degraded,
        elapsedMs=ans.elapsed_ms,
        script=ans.script,
        handlingSuggestion=ans.handling_suggestion or "",
        usability=UsabilityInfo(level=ans.usability.level,
                                reasons=ans.usability.reasons,
                                uncovered=ans.usability.uncovered),
        sources=[
            SourceItem(chunkId=s.chunk_id, docId=s.doc_id, docTitle=s.doc_title,
                       relevance=s.relevance, keyFragment=s.key_fragment,
                       content=s.content, updatedAt=s.updated_at,
                       stale=s.stale)
            for s in ans.sources
        ],
        processTrace=trace_events or [],
    )


# 默认应用实例:python -m uvicorn kbagent_service.app:app
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
