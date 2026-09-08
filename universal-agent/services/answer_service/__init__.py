# -*- coding: utf-8 -*-
"""answer 测试服务 — 只跑答案子智能体(真实大模型)的 FastAPI 封装。

启动(在 universal-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    set PYTHONPATH=src;services        # Windows(Linux 用 src:services)
    python -m uvicorn answer_service.app:app --host 0.0.0.0 --port 8001
"""
from .app import app, create_app
from .models import (
    AnswerObject,
    AnswerParams,
    AnswerRequest,
    AnswerResponse,
    ChunkIn,
    SentenceItem,
    SourceItem,
)

__all__ = [
    "app", "create_app",
    "AnswerRequest", "AnswerParams", "AnswerResponse", "AnswerObject",
    "ChunkIn", "SentenceItem", "SourceItem",
]
