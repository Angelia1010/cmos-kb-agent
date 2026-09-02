# -*- coding: utf-8 -*-
"""kbagent 服务层 — 灵犀契约的 FastAPI 封装。

启动(在 knowbase-agent 项目根目录下;PYTHONPATH 需同时含 src 与 services):
    set PYTHONPATH=src;services        # Windows(Linux 用 src:services)
    python -m uvicorn kbagent_service.app:app --host 0.0.0.0 --port 8000
"""
from .app import app, create_app
from .models import (
    AnswerObject,
    AskParams,
    AskRequest,
    AskResponse,
    Conversation,
    Device,
    ExtInfo,
    Scene,
    SourceItem,
    UserInfo,
)

__all__ = [
    "app", "create_app",
    "AskRequest", "AskParams", "AskResponse", "AnswerObject",
    "Conversation", "UserInfo", "ExtInfo", "Device", "Scene", "SourceItem",
]
