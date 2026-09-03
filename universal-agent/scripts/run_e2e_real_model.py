#!/usr/bin/env python
"""端到端验证:真实大模型(config.yaml → 灵犀网关 LingxiSSLChatOpenAI)跑通
主智能体全链路(缓存 → 检索 → 处理 → 答案 → 降级兜底)。

用法(在 universal-agent 目录下):
    # Windows
    set PYTHONPATH=src && python scripts/run_e2e_real_model.py
    # Linux / Git-Bash
    PYTHONPATH=src python scripts/run_e2e_real_model.py

环境变量:
    QWEN_API_KEY   灵犀网关密钥(config.yaml 中 ${QWEN_API_KEY} 注入)

检索后端:
    默认离线 MockESClient;接入生产 ngkm 时传 --es produce(内网环境)。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kbagent import MainAgent, MockESClient  # noqa: E402
from kbagent.shared.search import ProduceESClient  # noqa: E402


def build_model():
    """按 config.yaml 的 models[].use 构建真实大模型(不做离线回退)。"""
    from uniagent.config.app_config import get_app_config
    from uniagent.imports.resolvers import resolve_class
    cfg = get_app_config()
    if not cfg.models:
        raise SystemExit("config.yaml 未配置 models,无法跑真实大模型")
    mc = next((m for m in cfg.models if m.name == "default"), cfg.models[0])
    key = str(mc.kwargs.get("api_key") or "")
    if key.startswith("${"):
        raise SystemExit("api_key 占位符未展开 —— 请先注入对应环境变量")
    model = resolve_class(mc.use)(model=mc.model, temperature=mc.temperature,
                                  **mc.kwargs)
    print(f"[模型] {type(model).__name__} | use={mc.use} | model={mc.model}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="KB-Agent 真实大模型端到端验证")
    parser.add_argument("--es", choices=("mock", "produce"), default="mock",
                        help="检索后端:离线样例(mock,默认) / 生产ngkm(produce)")
    parser.add_argument("--query", action="append", default=None,
                        help="自定义查询,可多次传入;缺省跑内置示例")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    model = build_model()
    es = ProduceESClient() if args.es == "produce" else MockESClient()
    print(f"[检索] {type(es).__name__}")

    queries = args.query or [
        "用户想办理流量套餐,如何推荐?",
        "宽带新装怎么办理?",
    ]
    agent = MainAgent(model=model, es=es,
                      skill_dirs=[str(PROJECT_ROOT / "skills")])
    for q in queries:
        print(f"\n{'=' * 70}\n>> {q}\n{'=' * 70}")
        ans = agent.run(q)
        print(ans.render())
        print(f"\n[meta] trace_id={ans.trace_id} degraded={ans.degraded} "
              f"from_cache={ans.from_cache} elapsed_ms={ans.elapsed_ms}")


if __name__ == "__main__":
    main()
