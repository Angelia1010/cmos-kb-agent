# -*- coding: utf-8 -*-
"""llm_397b 服务 — 灵犀大模型网关直通透传的最小 FastAPI 封装。

POST /llm_397b_api:输入用户问题(query),输出大模型预测 token 文本。

启动(在 universal-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    set PYTHONPATH=src;services        # Windows(Linux 用 src:services)
    python -m uvicorn llm_service.app:app --host 0.0.0.0 --port 8002
"""
from .app import app, create_app
from .models import LlmObject, LlmRequest, LlmResponse

__all__ = [
    "app", "create_app",
    "LlmRequest", "LlmResponse", "LlmObject",
]
