# -*- coding: utf-8 -*-
"""带有模型级 TLS 策略的 Lingxi OpenAI 兼容 Provider。

源自灵犀侧 deerflow.community.lingxi.ssl_openai_provider,已 vendor 进本仓库;
config.yaml 经 ``use: "kbagent.lingxi_provider:LingxiSSLChatOpenAI"`` 解析。
"""

from __future__ import annotations

import logging
import ssl
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _coalesce_system_messages(messages: list[Any]) -> list[Any]:
    """把消息列表中所有 system 消息合并为一条,并固定放在第 0 位。

    内网网关(灵犀代理的 vLLM)严格要求 system 消息只能出现在开头,
    否则返回 400 'System message must be at the beginning'。
    而框架侧存在多处 system 注入(技能中间件追加在消息流尾部、
    GoalLoop 前置目标消息、工厂注入角色提示词),在此统一整形兜底。
    """
    system_parts = [
        m.content if isinstance(m.content, str) else str(m.content)
        for m in messages
        if isinstance(m, SystemMessage)
    ]
    if not system_parts:
        return messages
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return [SystemMessage(content="\n\n".join(system_parts))] + rest


def _redact_url_userinfo(url: Any) -> str:
    """返回移除用户信息、查询串和片段的日志安全端点地址。"""
    if url is None:
        return "<unset>"

    text = str(url)
    try:
        parts = urlsplit(text)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return "<invalid-url>"

    if parts.scheme not in {"http", "https"} or not hostname:
        return "<invalid-url>"

    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _validate_boolean_flag(name: str, value: Any) -> bool:
    """严格校验 TLS 策略开关，避免字符串真值意外启用安全降级。"""
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")
    return value


def _build_ssl_context(*, verify_ssl: bool, allow_legacy_dh: bool) -> ssl.SSLContext:
    """为单个模型构建 TLS 上下文。"""
    context = ssl.create_default_context()

    if not verify_ssl:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    if allow_legacy_dh:
        context.set_ciphers("DEFAULT:@SECLEVEL=1")

    return context


class LingxiSSLChatOpenAI(ChatOpenAI):
    """支持按模型控制证书校验和旧版 DH 兼容的 ChatOpenAI。

    ``cs_verify_ssl`` 默认开启证书与主机名校验。``cs_allow_legacy_dh`` 默认关闭；
    显式开启时，为该模型专用 TLS 上下文应用 OpenSSL
    ``DEFAULT:@SECLEVEL=1`` 密码策略。
    """

    def __init__(
        self,
        *args: Any,
        cs_verify_ssl: bool = True,
        cs_allow_legacy_dh: bool = False,
        **kwargs: Any,
    ) -> None:
        cs_verify_ssl = _validate_boolean_flag("cs_verify_ssl", cs_verify_ssl)
        cs_allow_legacy_dh = _validate_boolean_flag("cs_allow_legacy_dh", cs_allow_legacy_dh)
        custom_client_names = [
            name
            for name in ("http_client", "http_async_client")
            if kwargs.get(name) is not None
        ]
        tls_policy_enabled = cs_verify_ssl is False or cs_allow_legacy_dh
        if tls_policy_enabled and custom_client_names:
            joined_names = ", ".join(custom_client_names)
            raise ValueError(
                f"Lingxi TLS policy cannot be combined with custom HTTP clients: {joined_names}"
            )

        if tls_policy_enabled:
            kwargs.pop("http_client", None)
            kwargs.pop("http_async_client", None)

        model = kwargs.get("model") or "<unset>"
        endpoint = kwargs.get("base_url") or kwargs.get("openai_api_base")

        if cs_verify_ssl is False:
            logger.warning(
                "SSL certificate verification is disabled for Lingxi OpenAI-compatible model %r at endpoint %s. Use only for administratively approved endpoints.",
                model,
                _redact_url_userinfo(endpoint),
            )

        if cs_allow_legacy_dh:
            logger.warning(
                "Legacy DH compatibility (OpenSSL SECLEVEL=1) is enabled for Lingxi OpenAI-compatible model %r at endpoint %s. Use only for administratively approved endpoints.",
                model,
                _redact_url_userinfo(endpoint),
            )

        if tls_policy_enabled:
            ssl_context = _build_ssl_context(
                verify_ssl=cs_verify_ssl,
                allow_legacy_dh=cs_allow_legacy_dh,
            )
            kwargs["http_client"] = httpx.Client(verify=ssl_context)
            kwargs["http_async_client"] = httpx.AsyncClient(verify=ssl_context)

        super().__init__(*args, **kwargs)

    # ── 网关兼容:发送前合并/前置 system 消息 ────────────────────────────
    def _generate(self, messages: Any, stop: Any = None,
                  run_manager: Any = None, **kwargs: Any) -> Any:
        return super()._generate(
            _coalesce_system_messages(messages),
            stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages: Any, stop: Any = None,
                         run_manager: Any = None, **kwargs: Any) -> Any:
        return await super()._agenerate(
            _coalesce_system_messages(messages),
            stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages: Any, stop: Any = None,
                run_manager: Any = None, **kwargs: Any) -> Any:
        yield from super()._stream(
            _coalesce_system_messages(messages),
            stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages: Any, stop: Any = None,
                       run_manager: Any = None, **kwargs: Any) -> Any:
        async for chunk in super()._astream(
                _coalesce_system_messages(messages),
                stop=stop, run_manager=run_manager, **kwargs):
            yield chunk
