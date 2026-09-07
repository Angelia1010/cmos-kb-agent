# -*- coding: utf-8 -*-
"""retrieval 服务层 — 纯检索结果的 FastAPI 封装。

启动(在 knowbase-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    set PYTHONPATH=src;services        # Windows(Linux 用 src:services)
    python -m uvicorn retrieval_service.app:app --host 0.0.0.0 --port 8000
"""
from .app import app, create_app
from .models import (
    RetrievalChunk,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseObject,
)

__all__ = [
    "app", "create_app",
    "RetrievalRequest", "RetrievalResponse", "RetrievalResponseObject",
    "RetrievalChunk",
]
