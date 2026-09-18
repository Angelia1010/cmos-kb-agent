#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量向量检索评测:读测试集(省份/字号/用户问)逐条调在线知识 embedding
向量检索服务,按固定格式保存每条的完整原始返回,供后续离线分析/标注。

用法(在 universal-agent 目录下):
    # Windows
    set PYTHONPATH=src && python scripts/batch_vector_search.py \
        --input 测试集.jsonl --output 向量结果_new.jsonl --mode new
    # Linux / Git-Bash
    PYTHONPATH=src python scripts/batch_vector_search.py \
        --input 测试集.csv --output 向量结果_old.jsonl --mode old

输入文件格式(表头/键名兼容别名):
    .jsonl  每行一个 JSON 对象;
    .json   一个 JSON 数组;
    .csv    首行表头。
    字段: 省份(province/provinceId)、字号(序号/serial/no,可缺省取行号)、
          用户问(query/question/content)。

输出(.jsonl,每行一个结果对象,逐条落盘,中断不丢已完成结果):
    {"省份": ..., "字号": ..., "用户问": ...,
     "response": <接口完整原始返回>, "success": true/false, "error": "..."}

说明:
    --mode 单次只跑一个模板(new 或 old);需要新旧对比时分别跑两次、
    输出到不同文件(response 字段只保存单次调用的完整原始返回)。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kbagent.shared.search import ProduceESClient  # noqa: E402

# 测试集列名 → 标准字段的别名映射
_FIELD_ALIASES = {
    "省份": ("省份", "省", "province", "provinceId", "province_id"),
    "字号": ("字号", "序号", "编号", "serial", "serial_number", "no", "id"),
    "用户问": ("用户问", "用户问题", "问", "query", "question", "content"),
}


def _pick(row: Dict[str, Any], names: tuple) -> str:
    """按别名顺序取第一个非空值。"""
    for name in names:
        v = row.get(name)
        if v not in (None, ""):
            return str(v).strip()
    return ""


def load_test_set(path: Path) -> List[Dict[str, str]]:
    """加载测试集为 [{省份, 字号, 用户问}, ...];字号缺省稍后用行号补齐。"""
    suffix = path.suffix.lower()
    rows: List[Dict[str, str]] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("--input 的 .json 文件须为对象数组")
        rows = data
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))
    else:
        raise SystemExit(f"不支持的输入格式: {suffix}(仅支持 .jsonl/.json/.csv)")

    items: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        query = _pick(row, _FIELD_ALIASES["用户问"])
        if not query:
            print(f"[跳过] 第 {idx} 行无'用户问'字段: {str(row)[:120]}")
            continue
        items.append({
            "省份": _pick(row, _FIELD_ALIASES["省份"]),
            "字号": _pick(row, _FIELD_ALIASES["字号"]) or str(idx),
            "用户问": query,
        })
    return items


def run_batch(input_path: Path, output_path: Path, mode: str,
              timeout: int) -> None:
    items = load_test_set(input_path)
    print(f"[输入] {input_path} → {len(items)} 条有效问题")
    print(f"[配置] mode={mode} timeout={timeout}s output={output_path}")

    # mode 逐次调用传给 raw_vector_search(单模板原始返回落盘),客户端不持有选路配置
    client = ProduceESClient(timeout=timeout)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    with output_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(items, start=1):
            search_result = client.raw_vector_search(
                query_text=item["用户问"], mode=mode,
                province=item["省份"], timeout=timeout)
            result_item = {
                "省份": item["省份"],
                "字号": item["字号"],
                "用户问": item["用户问"],
                # 接口完整原始返回结果
                "response": search_result["response"],
                # 请求状态
                "success": search_result["success"],
                "error": search_result["error"],
            }
            out.write(json.dumps(result_item, ensure_ascii=False,
                                 default=str) + "\n")
            out.flush()
            if search_result["success"]:
                ok += 1
                flag = "OK"
            else:
                fail += 1
                flag = f"FAIL {search_result['error'][:80]}"
            print(f"[{i}/{len(items)}] 字号={item['字号']} 省份={item['省份']} "
                  f"{flag}")
    print(f"\n[完成] 共 {len(items)} 条: 成功 {ok} / 失败 {fail}")
    print(f"[输出] {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="批量向量检索评测(原始返回落盘)")
    parser.add_argument("--input", required=True, help="测试集文件(.jsonl/.json/.csv)")
    parser.add_argument("--output", default="",
                        help="结果输出 .jsonl;缺省为 向量结果_<mode>.jsonl")
    parser.add_argument("--mode", choices=("new", "old"), default="new",
                        help="请求模板:新模板 new(默认)/ 旧模板 old")
    parser.add_argument("--timeout", type=int, default=30, help="单请求超时秒数")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")
    output_path = Path(args.output) if args.output else \
        Path(f"向量结果_{args.mode}.jsonl")
    run_batch(input_path, output_path, args.mode, args.timeout)


if __name__ == "__main__":
    main()
