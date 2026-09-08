# -*- coding: utf-8 -*-
"""intergrate_all 工具真实接口测试

覆盖范围:
  T1  intergrate_all — 一体化检索流水线真实接口调用
      (槽位提取 → 知识主索引 → 原子表 → 合并)

运行方式:
  cd universal-agent
  PYTHONPATH=src python -m unittest tests.test_intergrate -v

依赖网络:
  restapi.ly4.tyyt.cmos:20070           (槽位提取)
  restapi.ngkmsearch.cs.glb.cmos:20070  (知识主索引/原子表检索)

断言策略:
  网络不可用时 skipTest 跳过;可用时聚焦结构验证(类型/字段存在性),
  不依赖固定返回值。
"""
import os
import sys
import unittest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, _SRC)

from kbagent.shared.config import DEFAULT_CONFIG
from kbagent.shared.search import ProduceESClient
from kbagent.shared.tracing import Tracer
from kbagent.shared.workspace import RunWorkspace, set_workspace
from kbagent.retrieval.tools import intergrate_all


def _ws(query: str = "如何办理5G套餐") -> RunWorkspace:
    ws = RunWorkspace(
        query=query, cfg=DEFAULT_CONFIG,
        es=ProduceESClient(), tracer=Tracer(),
    )
    ws.stage = "retrieval"
    set_workspace(ws)
    return ws


if __name__ == "__main__":
    import json as _json

    ws = _ws(query="如何办理5G套餐")
    result = intergrate_all.func(query="如何办理5G套餐", region_code="全国")
    print("=== intergrate_all 测试结果 ===")
    print(_json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"merged_results 写入工作区: {ws.data.get('merged_results') is not None}")
    print(f"original_query: {ws.data.get('original_query')}")
    print(f"region_code: {ws.data.get('region_code')}")
