# -*- coding: utf-8 -*-
"""
将 recall_eval.py rank 模式输出的 JSON 转换为 Excel 可打开的 CSV 文件。

零外部依赖(仅用 Python 标准库:json / csv / argparse / pathlib)。

用法:
    python json_to_excel.py                          # 自动找同目录 *_ranked.json
    python json_to_excel.py -i test_ranked.json -o result.csv

一行一个问题,top3 / top100 各占一列,单元格内格式:
    1. 标题(相似度) [原排名]
    2. 标题(相似度) [原排名]
    ...

JSON 结构要求(recall_eval.py rank 模式输出,同时支持 top3 + top100):
    [
        {
            "index": 1, "province": "河南",
            "user_query": "不做实名登记会有什么影响？",
            "expected_titles": ["关于实名制信息补登记的相关解释口径"],
            "top3":  [{"rank":1,"original_rank":2,"title":"...","score":1.0,"best_expected":"..."}, ...],
            "top3_count": 3, "top3_hit_count": 1, "top3_best_score": 1.0,
            "top100": [{"rank":1,"original_rank":74,"title":"...","score":1.0,"best_expected":"..."}, ...],
            "top100_count": 105, "top100_hit_count": 1, "top100_best_score": 1.0
        },
        ...
    ]
兼容旧格式(只有 top100 / top100_count / hit_count / best_score)。
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# TOP-N 字段名(与 recall_eval._TOPN_KEYS 保持一致)
_TOPN_NAMES = ("top3", "top100")

# 表头:一个问题一行,top3 / top100 各占两列(重排详情 + 统计)
HEADER = [
    "序号",
    "省份",
    "客户问题",
    "期望知识标题",
    "top3重排",
    "top3统计",
    "top100重排",
    "top100统计",
]


def find_input_file() -> Path:
    """自动寻找同目录下的 JSON 文件(优先级:*_ranked.json > *.json)。"""
    script_dir = Path(__file__).resolve().parent

    candidates = sorted(script_dir.glob("*_ranked.json"))
    if candidates:
        return candidates[0]

    candidates = sorted(script_dir.glob("*.json"))
    if candidates:
        candidates = [c for c in candidates
                      if "测试集" not in c.stem and "sample" not in c.stem.lower()]
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"同目录下未找到 JSON 文件。请将 rank 模式输出的 JSON 放到:\n  {script_dir}\n"
        f"或使用 --input 指定路径。"
    )


def load_json(path: Path) -> list:
    """加载 JSON,兼容数组和带 data 键的对象。"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError(f"JSON 顶层必须是 list 或含 data 键的 dict,实际类型:{type(data).__name__}")


def _format_topn_detail(topn_list: list, threshold: float = 0.85) -> str:
    """把 TOP-N 重排列表格式化为多行字符串,放一个单元格里。

    每条标注是否命中(score >= threshold 显示 ✓,否则显示 ✗)。

    格式:
        1. 关于实名制信息补登记的相关解释口径(1.0000 ✓) [原74]
        2. 手机挂失补卡(0.0600 ✗) [原58]
        3. ...
    """
    if not topn_list:
        return ""
    lines = []
    for t in topn_list:
        rank = t.get("rank", "")
        title = t.get("title", "")
        score = t.get("score", 0.0)
        orig = t.get("original_rank", "")
        hit = "✓" if score >= threshold else "✗"
        lines.append(f"{rank}. {title}({score:.4f} {hit}) [原{orig}]")
    return "\n".join(lines)


def _format_topn_stat(item: dict, topn_name: str) -> str:
    """把 TOP-N 统计格式化为单行字符串。

    格式: 总数=105 命中=1 最佳=1.0000
    """
    # 新命名
    count_key = f"{topn_name}_count"
    hit_key = f"{topn_name}_hit_count"
    best_key = f"{topn_name}_best_score"
    # 旧格式 fallback(仅 top100)
    if topn_name == "top100":
        count = item.get(count_key, item.get("top100_count", 0))
        hit = item.get(hit_key, item.get("hit_count", 0))
        best = item.get(best_key, item.get("best_score", 0.0))
    else:
        count = item.get(count_key, 0)
        hit = item.get(hit_key, 0)
        best = item.get(best_key, 0.0)
    return f"总数={count} 命中={hit} 最佳={best:.4f}"


def flatten(ranked: list) -> list:
    """把 rank 模式嵌套 JSON 扁平化为一行一问题的二维表。

    top3 / top100 各占两列(重排详情 + 统计),单元格内换行用 \n(Excel 需开启自动换行)。
    """
    rows = []
    for item in ranked:
        idx = item.get("index", "")
        province = item.get("province", "")
        query = item.get("user_query", "")
        expected = " | ".join(item.get("expected_titles", []))

        # top3
        top3_list = item.get("top3")
        if top3_list is not None:
            top3_detail = _format_topn_detail(top3_list)
            top3_stat = _format_topn_stat(item, "top3")
        else:
            top3_detail = ""
            top3_stat = ""

        # top100(兼容旧格式)
        top100_list = item.get("top100")
        if top100_list is not None:
            top100_detail = _format_topn_detail(top100_list)
            top100_stat = _format_topn_stat(item, "top100")
        else:
            top100_detail = ""
            top100_stat = ""

        rows.append([
            idx, province, query, expected,
            top3_detail, top3_stat,
            top100_detail, top100_stat,
        ])
    return rows


def write_csv(rows: list, path: Path) -> None:
    """写 CSV,utf-8-sig BOM 让 Excel 正确识别中文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 recall_eval.py rank 模式输出的 JSON 转为 Excel CSV(一行一问题)",
    )
    parser.add_argument("--input", "-i", help="输入 JSON 文件路径(缺省同目录自动寻找)")
    parser.add_argument("--output", "-o", help="输出 CSV 文件路径(缺省与输入同 stem)")
    args = parser.parse_args()

    # ── 1. 定位输入 ──────────────────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input).resolve()
    else:
        try:
            input_path = find_input_file()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    print(f"[输入]  {input_path}")

    # ── 2. 加载 JSON ─────────────────────────────────────────────────────────
    try:
        ranked = load_json(input_path)
    except Exception as e:
        print(f"❌ 解析 JSON 失败: {e}")
        sys.exit(1)
    print(f"[JSON]  {len(ranked)} 条请求")

    # ── 3. 扁平化(一行一问题) ────────────────────────────────────────────────
    rows = flatten(ranked)
    print(f"[扁平]  {len(rows)} 行(一行一问题)")

    # ── 4. 写 CSV ────────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_name(input_path.stem + ".csv")
    write_csv(rows, output_path)
    print(f"[输出]  {output_path}")

    # ── 5. 快速统计 ─────────────────────────────────────────────────────────
    total_req = len(ranked)
    print(f"\n  总请求数 : {total_req}")
    for topn_name in _TOPN_NAMES:
        if any(topn_name in r for r in ranked):
            hit_req = sum(1 for r in ranked
                          if r.get(f"{topn_name}_hit_count",
                                   r.get("hit_count" if topn_name == "top100" else "", 0)) > 0)
            print(f"  {topn_name} 至少 1 条命中: {hit_req}/{total_req} "
                  f"({hit_req / max(total_req, 1) * 100:.2f}%)")
    print(f"  ✅ 完成!Excel 打开后开启自动换行即可查看单元格内完整重排。")


if __name__ == "__main__":
    main()
