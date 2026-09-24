# -*- coding: utf-8 -*-
"""把"已有 kbagent 结果"的 Excel 转成 judge 可吃的 responses.jsonl。

适配的 Excel 列(表头模糊匹配,自动去空白/换行/大小写):
  序号 / 省份 / 用户问(客户问题/query)
  Top100知识ID列表 / Top100知识标题列表   → object.retrievedDocs (召回全集, 算A4)
  Top3知识ID列表   / Top3知识标题列表     → object.sources      (Top3, 算A1)
  script                                → object.script
  handlingSuggestion(办理建议)           → object.handlingSuggestion
  可选: sources(原始JSON列, 含content)   → object.sources 整体替换Top3拼接(优先)
  可选: retrievedDocs(原始JSON列)        → object.retrievedDocs(优先)
  可选: content/原文/正文                → sources[i].content(无JSON列时按位置配对)
  可选: 知识ID列表/知识标题列表(gold标注) → gold_ids/gold_titles

单元格内多值用 换行/分号 分隔, ID 与标题按位置一一配对。

用法:
  python scripts/xlsx_to_responses.py --xlsx 结果.xlsx --out eval_run1
  python scripts/eval_kbagent.py judge --responses eval_run1/responses.jsonl \
      --llm-url http://.../llm_397b_api --out eval_run1 \
      [--testset 测试集2-300.xlsx]   # Excel 里无 gold 列时用测试集按 query 补齐
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_kbagent import read_xlsx_rows  # noqa: E402


def _norm(s: Any) -> str:
    """表头归一化: 去所有空白/换行, 小写, 去常见标点。"""
    return re.sub(r"[\s:：\-_/()（）\[\]【】]+", "", str(s or "")).lower()


def _split_cell(value: Any) -> List[str]:
    parts = re.split(r"[\n;；]+", str(value or ""))
    return [p.strip() for p in parts if p.strip()]


# 表头候选(归一化后精确匹配, 依次尝试)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "seq":        ["序号", "编号", "id"],
    "province":   ["省份", "省", "province"],
    "query":      ["用户问", "用户问题", "客户问题", "问题", "query", "question"],
    "top100_ids": ["top100知识id列表", "top100知识id", "召回知识id列表",
                   "retrieveddocids", "top100id"],
    "top100_titles": ["top100知识标题列表", "top100知识标题", "召回知识标题列表",
                      "retrieveddoctitles", "top100title"],
    "top3_ids":   ["top3知识id列表", "top3知识id", "top3id", "sourcesid"],
    "top3_titles": ["top3知识标题列表", "top3知识标题", "top3title", "sourcestitle"],
    "script":     ["script", "话术", "回答话术", "answerscript"],
    "suggestion": ["handlingsuggestion", "suggestion", "办理建议", "处理建议"],
    "content":    ["content", "原文", "正文", "知识内容", "top3正文", "top3content"],
    # 原始 JSON 列(kbagent 响应直接落表): 优先于 ID/标题列拼接
    "sources_json": ["sources", "source", "sourcesjson", "top3sources"],
    "retrieved_json": ["retrieveddocs", "retrieveddoc", "retrieveddocsjson",
                       "top100docs", "召回文档"],
    "gold_ids":   ["知识id列表", "知识id", "goldid", "goldids", "标准知识id"],
    "gold_titles": ["知识标题列表", "知识标题", "goldtitle", "goldtitles", "标准知识标题"],
}


def _find_columns(header: List[str]) -> Dict[str, int]:
    normed = [_norm(h) for h in header]
    idx: Dict[str, int] = {}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            a = _norm(alias)
            if a in normed:
                idx[key] = normed.index(a)
                break
    return idx


def _parse_json_cell(text: str) -> Any:
    """单元格若为 JSON 数组/对象则解析, 否则返回 None。容忍 excel 常见截断引号。"""
    s = str(text or "").strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_sources(parsed: Any) -> List[Dict[str, Any]]:
    """把 sources JSON 归一化为 [{docId, docTitle, content?, ...}]。"""
    if isinstance(parsed, dict):  # 可能整个 object 被塞进一格
        parsed = parsed.get("sources") or parsed.get("docs") or []
    out = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                out.append({
                    "docId": str(item.get("docId") or item.get("id")
                                 or item.get("doc_id") or ""),
                    "docTitle": str(item.get("docTitle") or item.get("title")
                                    or item.get("doc_title") or ""),
                    "content": str(item.get("content") or item.get("text")
                                   or item.get("chunk") or ""),
                    **{k: v for k, v in item.items()
                       if k in ("relevance", "keyFragment", "stale")},
                })
            elif isinstance(item, str) and item.strip():
                out.append({"docId": "", "docTitle": item.strip(), "content": ""})
    return out


def _normalize_retrieved(parsed: Any) -> List[Dict[str, str]]:
    if isinstance(parsed, dict):
        parsed = parsed.get("retrievedDocs") or parsed.get("docs") or []
    out = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                out.append({"id": str(item.get("id") or item.get("docId") or ""),
                            "title": str(item.get("title") or item.get("docTitle") or "")})
            elif isinstance(item, str) and item.strip():
                out.append({"id": "", "title": item.strip()})
    return out


def _pair(ids: List[str], titles: List[str]) -> List[Dict[str, str]]:
    """ID/标题按位置配对, 长度不齐时补空。"""
    n = max(len(ids), len(titles))
    return [{"id": ids[i] if i < len(ids) else "",
             "title": titles[i] if i < len(titles) else ""}
            for i in range(n)]


def convert(rows: List[List[str]]) -> List[Dict[str, Any]]:
    if not rows:
        raise SystemExit("xlsx 为空")
    cols = _find_columns(rows[0])
    if "query" not in cols:
        raise SystemExit(
            f"找不到问题列(用户问/客户问题/query), 实际表头: {rows[0]}")
    if "sources_json" not in cols:
        missing = [k for k in ("top3_ids", "top3_titles") if k not in cols]
        if missing:
            print(f"[warn] 未识别列: {missing}(且无 sources JSON列), "
                  f"对应字段将为空。实际表头: {rows[0]}", file=sys.stderr)

    def cell(row: List[str], key: str) -> str:
        i = cols.get(key, -1)
        return str(row[i]).strip() if 0 <= i < len(row) else ""

    records: List[Dict[str, Any]] = []
    seen_ids: Dict[str, int] = {}
    for row_no, row in enumerate(rows[1:], 2):
        query = re.sub(r"\s+", " ", cell(row, "query")).strip()
        if not query:
            continue
        case_id = f"c{cell(row, 'seq') or row_no}"
        if case_id in seen_ids:  # 去重
            seen_ids[case_id] += 1
            case_id = f"{case_id}_{seen_ids[case_id]}"
        else:
            seen_ids[case_id] = 0

        # sources: 优先解析原始 JSON 列(含 content 正文), 否则用 Top3 ID/标题拼接
        sources = _normalize_sources(_parse_json_cell(cell(row, "sources_json")))
        if not sources:
            top3_ids = _split_cell(cell(row, "top3_ids"))
            top3_titles = _split_cell(cell(row, "top3_titles"))
            contents = _split_cell(cell(row, "content"))
            for i, pair in enumerate(_pair(top3_ids, top3_titles)):
                src = {"docId": pair["id"], "docTitle": pair["title"],
                       "content": contents[i] if i < len(contents) else ""}
                sources.append(src)

        # retrievedDocs: 同理
        retrieved = _normalize_retrieved(_parse_json_cell(cell(row, "retrieved_json")))
        if not retrieved:
            retrieved = _pair(_split_cell(cell(row, "top100_ids")),
                              _split_cell(cell(row, "top100_titles")))

        obj: Dict[str, Any] = {
            "sources": sources,
            "retrievedDocs": retrieved,
            "script": cell(row, "script"),
            "handlingSuggestion": cell(row, "suggestion"),
            "degraded": False,
        }
        records.append({
            "case_id": case_id,
            "query": query,
            "province": cell(row, "province"),
            "gold_ids": _split_cell(cell(row, "gold_ids")),
            "gold_titles": _split_cell(cell(row, "gold_titles")),
            "object": obj,
            "error": None,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="结果Excel → responses.jsonl")
    ap.add_argument("--xlsx", required=True, help="输入 Excel 路径")
    ap.add_argument("--out", required=True, help="输出目录(生成 responses.jsonl)")
    args = ap.parse_args()

    records = convert(read_xlsx_rows(Path(args.xlsx)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_src = sum(1 for r in records if r["object"]["sources"])
    n_content = sum(1 for r in records
                    if any(s.get("content") for s in r["object"]["sources"]))
    n_scr = sum(1 for r in records if r["object"]["script"])
    print(f"转换完成: {len(records)} 条 (有Top3: {n_src}, 含正文: {n_content}, "
          f"有话术: {n_scr}) → {out_path}")


if __name__ == "__main__":
    main()
