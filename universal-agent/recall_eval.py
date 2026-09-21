# -*- coding: utf-8 -*-
"""基于编辑距离的召回重排与评测。

两大模式:
  rank  ——  重排模式:单一输入文件(Excel/JSON),每行含知识标题 + TOP100 列表,
            对 TOP100 每条算与知识标题的编辑距离相似度,按分数降序重排后
            写回 JSON,每条召回附带 score 与 original_rank。
  eval  ——  评测模式(保留):分测试集/召回结果两文件,按阈值统计召回率。

TOP100 列兼容格式:
  a) JSON 数组   ["标题1", "标题2", ...]
  b) 换行/分隔字符串  "标题1\\n标题2\\n..." 或 "标题1,标题2"
  c) HTML span 标签  '<span class="classname">标题</span>'(自动剥离)
  d) 多条记录同 query 时每条单值(聚合层自动合并)

CLI:
    # 重排模式(单一输入文件)
    python -m kbagent.retrieval.recall_eval \\
        --mode rank \\
        --input   合并文件.xlsx \\
        --output  重排结果.json \\
        --top-n   100

    # 评测模式(分两文件,兼容旧用法)
    python -m kbagent.retrieval.recall_eval \\
        --mode eval \\
        --testset 测试集.json --recall 召回结果.jsonl \\
        --threshold 0.85 --output 评测报告.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("kbagent.retrieval.recall_eval")

# ── 字段别名 ────────────────────────────────────────────────────────────────
# 序号
_INDEX_KEYS = ("index", "序号", "serial", "id")
# 省份
_PROVINCE_KEYS = ("province", "省份", "provinceId")
# 用户问题
_QUERY_KEYS = ("user_query", "客户问题", "用户问", "用户问题",
                "query", "question")
# 期望知识标题(单值或列表均可,支持多行/多标题)
_EXPECTED_TITLE_KEYS = ("expected_titles", "knowledge_name", "knowledgeName",
                        "knowledge_names", "expected_title", "title",
                        "知识名称", "客户问题涉及的\n知识标题")
# TOP-N 召回列别名:key=输出字段名,value=Excel/JSON 中的列名候选
_TOPN_KEYS: Dict[str, Tuple[str, ...]] = {
    "top3":  ("top3", "top3_titles", "Top3知识标题列表", "Top3知识标题",
              "top3_title", "recalled_top3"),
    "top100": ("top100", "top100_titles", "recalled_top100",
               "Top100知识标题列表", "Top100知识标题", "top_100",
               "top100_title"),
}
# 召回结果查询字段(评测模式)
_RECALL_QUERY_KEYS = ("user_query", "客户问题", "用户问", "用户问题",
                      "query", "question")
# 召回结果标题字段(评测模式)
_RECALL_TITLE_KEYS = ("recalled_titles", "knowledge_names", "knowledgeName",
                      "titles", "doc_titles", "doc_title", "knowledge_name")


# ===========================================================================
# 1. 编辑距离(Levenshtein Distance)与相似度(复用原有)
# ===========================================================================
def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两字符串的编辑距离(插入/删除/替换各计 1 步)。
    滚动数组实现,O(min(m,n)) 空间。"""
    s1 = s1 or ""
    s2 = s2 or ""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s1) + 1))
    for j, ch2 in enumerate(s2, start=1):
        curr = [j] + [0] * len(s1)
        for i, ch1 in enumerate(s1, start=1):
            cost = 0 if ch1 == ch2 else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[-1]


def title_similarity(s1: str, s2: str) -> float:
    """基于编辑距离的归一化相似度,值域 [0.0, 1.0]。
    公式: sim = 1 - dist / max(len(s1), len(s2))。"""
    s1 = s1 or ""
    s2 = s2 or ""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


# ===========================================================================
# 2. 通用工具
# ===========================================================================
# 匹配 HTML 标签:<tag ...>、</tag>、<tag .../>
# 说明:使用贪婪 + DOTALL 确保 <span class="a"><span class="b">嵌套</span></span>
#      这种嵌套结构被完整剥离,而不是只剥外层留下内层。
_HTML_TAG_RE = re.compile(r"<[^>]*>")


def strip_html(value: str) -> str:
    """剥离所有 HTML / 类 HTML 标签,保留纯文本内容。

    处理规则:
      1. 匹配所有 <...> 形态(开标签、闭标签、自闭合标签),整段删除;
      2. 不区分标签名(span/div/a/img/br 等一律删除),避免 span 优先导致
         非 span 普通文本(如 "实名认证<span>...</span>")被意外丢弃;
      3. 嵌套标签(<span><span>嵌套</span></span>)一次性剥净;
      4. 删除后合并多余空白、换行。
    """
    if not value:
        return ""
    # 第 1 轮:剥离所有 <...> 标签
    text = _HTML_TAG_RE.sub("", value)
    # 第 2 轮:压缩多余空白(换行/制表/多空格 → 单空格)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _coerce_titles(value: Any) -> List[str]:
    """将字段值规整为字符串列表。
    支持 None / str / list / tuple / JSON 字符串 / 带 HTML 的字符串。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out = []
        for x in value:
            s = strip_html(str(x)).strip() if x is not None else ""
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # 尝试解析为 JSON
        if s.startswith("["):
            try:
                arr = json.loads(s)
                return _coerce_titles(arr)
            except json.JSONDecodeError:
                pass
        # 按常见分隔符拆分(换行 / 逗号 / 顿号 / 分号)
        parts = re.split(r"[\n\r,，、;；]+", s)
        out = []
        for p in parts:
            p = strip_html(p).strip()
            if p:
                out.append(p)
        return out
    s = strip_html(str(value)).strip()
    return [s] if s else []


def _pick(row: Dict[str, Any], keys: Sequence[str]) -> str:
    """按别名顺序取第一个非空字段值,返回 str。"""
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return strip_html(str(v)).strip()
    return ""


def _pick_titles(row: Dict[str, Any], keys: Sequence[str],
                  override: Optional[str] = None) -> List[str]:
    """按别名探测并规整标题列表,override 非空时强制用该字段。"""
    if override:
        return _coerce_titles(row.get(override))
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return _coerce_titles(row.get(k))
    return []


# ===========================================================================
# 3. 重排模式核心
# ===========================================================================
@dataclass
class RankedRecall:
    """单条召回重排结果。"""
    original_rank: int            # 原始序号(0-based)
    title: str                    # 召回标题
    score: float                  # 与期望标题的相似度
    best_expected: str = ""       # 与之匹配度最高的期望标题


def rank_top100(
    top_titles: Sequence[str],
    expected_titles: Sequence[str],
) -> List[RankedRecall]:
    """对 TOP100 每条召回算相似度、重排。

    多条期望标题时取 **最高** 匹配分(非平均)。

    Args:
        top_titles: 原始召回标题列表(保持原序,用于 original_rank)。
        expected_titles: 该条目的期望知识标题集合。

    Returns:
        按 score 降序排列的 RankedRecall 列表。
    """
    expected_list = [t for t in (expected_titles or []) if t]
    results: List[RankedRecall] = []
    for i, title in enumerate(top_titles):
        title = (title or "").strip()
        if not title:
            continue
        best_score = 0.0
        best_exp = ""
        if expected_list:
            for exp in expected_list:
                exp = exp.strip()
                if not exp:
                    continue
                sim = title_similarity(title, exp)
                if sim > best_score:
                    best_score = sim
                    best_exp = exp
        results.append(RankedRecall(
            original_rank=i,
            title=title,
            score=round(best_score, 4),
            best_expected=best_exp,
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def rank_rows(
    rows: List[Dict[str, Any]],
    top_n: int = 0,
    expected_titles_key: Optional[str] = None,
    top3_key: Optional[str] = None,
    top100_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """对合并文件的每行做 TOP-N 重排(同时支持 top3 和 top100 两列)。

    同一 user_query 出现多行时,自动聚合各 TOP-N 标题后分别统一重排。

    Args:
        rows: 输入行列表(每行一个 dict)。
        top_n: 重排时截断前 N 条(0 表示保留全部原始条数)。
        expected_titles_key: 强制指定期望标题列名;None 时按别名探测。
        top3_key: 强制指定 top3 列名;None 时按别名探测。
        top100_key: 强制指定 top100 列名;None 时按别名探测。

    Returns:
        新 dict 列表,每行同时含 top3 / top100(视输入有无)重排结果,
        每个 TOP-N 字段替换为 [{"title","score","original_rank","best_expected"}, ...] 格式,
        保留原有 index/province/user_query/expected_titles 等。
    """
    # ── Step 1: 聚合各 TOP-N ───────────────────────────────────────────────
    agg: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for row in rows:
        idx_val = _pick(row, _INDEX_KEYS)
        province = _pick(row, _PROVINCE_KEYS)
        query = _pick(row, _QUERY_KEYS)
        try:
            idx = int(idx_val) if idx_val else 0
        except ValueError:
            idx = 0

        key = (idx, province, query)
        bucket = agg.get(key)
        if bucket is None:
            expected = _pick_titles(row, _EXPECTED_TITLE_KEYS, expected_titles_key)
            bucket = {
                "index": idx,
                "province": province,
                "user_query": query,
                "expected_titles": expected,
            }
            for topn_name in _TOPN_KEYS:
                bucket[f"{topn_name}_titles"] = []
            agg[key] = bucket

        # 同时收集 top3 和 top100
        explicit_keys = {"top3": top3_key, "top100": top100_key}
        for topn_name, aliases in _TOPN_KEYS.items():
            titles = _pick_titles(row, aliases, explicit_keys.get(topn_name))
            # 去重累积(保序)
            existing = bucket[f"{topn_name}_titles"]
            seen = set(existing)
            for t in titles:
                if t not in seen:
                    existing.append(t)
                    seen.add(t)

    # ── Step 2: 各 TOP-N 分别重排 + 写回 ────────────────────────────────────
    output: List[Dict[str, Any]] = []
    for key, bucket in sorted(agg.items(), key=lambda kv: kv[0][0]):  # 按 index 排序
        item: Dict[str, Any] = {
            "index": bucket["index"],
            "province": bucket["province"],
            "user_query": bucket["user_query"],
            "expected_titles": bucket["expected_titles"],
        }
        # 对每个 TOP-N 独立重排
        for topn_name in _TOPN_KEYS:
            raw_titles = bucket[f"{topn_name}_titles"]
            if not raw_titles:
                continue  # 输入中没有这一列,跳过
            ranked = rank_top100(raw_titles, bucket["expected_titles"])
            effective_n = top_n if top_n > 0 else len(ranked)
            ranked = ranked[:effective_n]
            # 找该 TOP-N 对应的阈值(硬编码 0.85,同 recall_eval 口径)
            threshold = 0.85
            item[topn_name] = [
                {
                    "rank": i + 1,
                    "original_rank": r.original_rank + 1,
                    "title": r.title,
                    "score": r.score,
                    "best_expected": r.best_expected,
                }
                for i, r in enumerate(ranked)
            ]
            item[f"{topn_name}_count"] = len(raw_titles)
            item[f"{topn_name}_hit_count"] = sum(
                1 for r in ranked if r.score >= threshold
            )
            item[f"{topn_name}_best_score"] = ranked[0].score if ranked else 0.0
        output.append(item)
    return output


# ===========================================================================
# 4. 文件加载(零外部依赖:zipfile+xml 解析 .xlsx;stdlib 处理 JSON/CSV)
# ===========================================================================
_XML_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XML_NS_REL  = "http://schemas.openxmlformats.org/package/2006/relationships"


def _col_letter_to_index(letters: str) -> int:
    """Excel 列字母 → 0-based 索引(A→0, AA→26, ...)。"""
    idx = 0
    for c in letters:
        idx = idx * 26 + (ord(c.upper()) - ord("A") + 1)
    return idx - 1


def _parse_xlsx(xlsx_path: Path) -> List[Dict[str, Any]]:
    """原生解析 .xlsx(零依赖):zipfile 解压 + xml.etree 读 sheet + sharedStrings。

    只支持 .xlsx;.xls(旧二进制 Excel)回退为报错提示安装 openpyxl。
    """
    import zipfile
    import xml.etree.ElementTree as ET

    def _ns(tag: str) -> str:
        return f"{{{_XML_NS_MAIN}}}{tag}"

    rows_out: List[Dict[str, Any]] = []

    with zipfile.ZipFile(xlsx_path, "r") as z:
        # ── 1. 读共享字符串表 ────────────────────────────────────────────
        shared: List[str] = []
        ss_name = "xl/sharedStrings.xml"
        if ss_name in z.namelist():
            ss_root = ET.fromstring(z.read(ss_name))
            for si in ss_root.findall(_ns("si")):
                # <si><t>纯文本</t></si> 或 <si><r><t>富文本</t></r></si>
                texts = []
                for t in si.iter(_ns("t")):
                    if t.text:
                        texts.append(t.text)
                shared.append("".join(texts))

        # ── 2. 读第一个 sheet ─────────────────────────────────────────────
        # workbook.xml 给出 sheet 名 → 但我们只读 sheet1.xml,简化处理
        sheet_files = sorted(
            [n for n in z.namelist()
             if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")],
            key=lambda n: int(n.replace("xl/worksheets/sheet", "").replace(".xml", "")),
        )
        if not sheet_files:
            return rows_out

        sheet_root = ET.fromstring(z.read(sheet_files[0]))

        # ── 3. 逐行逐格读取 ────────────────────────────────────────────────
        raw_rows: List[List[Any]] = []
        max_cols = 0
        for row_el in sheet_root.iter(_ns("row")):
            row_data: Dict[int, Any] = {}
            for c in row_el.iter(_ns("c")):
                cell_ref = c.get("r", "")  # 如 "A1", "BC27"
                letters = "".join(ch for ch in cell_ref if ch.isalpha())
                col_idx = _col_letter_to_index(letters) if letters else len(row_data)
                cell_type = c.get("t", "n")  # n=数字, s=共享字符串, inlineStr=内联
                v_el = c.find(_ns("v"))
                val: Any = None
                if v_el is not None and v_el.text is not None:
                    if cell_type == "s":
                        idx = int(v_el.text)
                        val = shared[idx] if 0 <= idx < len(shared) else ""
                    elif cell_type == "b":
                        val = bool(v_el.text == "1")
                    else:
                        # 数字 / 日期(简化:保留原始文本,不做日期格式转换)
                        try:
                            fval = float(v_el.text)
                            val = int(fval) if fval == int(fval) else fval
                        except ValueError:
                            val = v_el.text
                else:
                    # inlineStr 内联字符串
                    is_el = c.find(_ns("is"))
                    if is_el is not None:
                        texts = [t.text or "" for t in is_el.iter(_ns("t"))]
                        val = "".join(texts) or None
                row_data[col_idx] = val
                max_cols = max(max_cols, col_idx + 1)
            if row_data:
                # 转为稠密列表
                dense = [row_data.get(i) for i in range(max_cols)]
                raw_rows.append(dense)

    if not raw_rows:
        return rows_out

    # ── 4. 第一行当表头,其余转为 dict ──────────────────────────────────
    headers = [str(h).strip() if h is not None else f"col_{i}"
               for i, h in enumerate(raw_rows[0])]
    for row in raw_rows[1:]:
        record: Dict[str, Any] = {}
        for i, h in enumerate(headers):
            v = row[i] if i < len(row) else None
            record[h] = v
        rows_out.append(record)
    return rows_out


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    """按扩展名加载文件为 dict 列表。零外部依赖(仅 stdlib)。"""
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return _parse_xlsx(path)

    if suffix == ".xls":
        # .xls 是旧二进制格式,stdlib 无法解析;提示装 openpyxl 或另存为 .xlsx
        raise SystemExit(
            f".xls 为旧版二进制格式,stdlib 无法解析。请将 {path.name} "
            f"另存为 .xlsx 后重试,或 pip install xlrd 后自行改造。")

    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit(f"{path} 的 .json 文件须为对象数组")
        return data

    if suffix == ".csv":
        import csv
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    raise SystemExit(f"不支持的输入格式: {suffix}(仅支持 .xlsx/.json/.jsonl/.csv)")


# ===========================================================================
# 5. 评测模式(保留,简化)
# ===========================================================================
@dataclass
class _TitleMatch:
    recalled_title: str
    best_expected: str
    best_score: float
    is_hit: bool


def _evaluate_item(
    recalled_titles: Sequence[str],
    expected_titles: Sequence[str],
    index: int = 0,
    user_query: str = "",
    province: str = "",
    threshold: float = 0.85,
) -> Dict[str, Any]:
    expected_list = [t for t in (expected_titles or []) if t]
    recalled_list = [t for t in (recalled_titles or []) if t]
    matched: List[_TitleMatch] = []
    exp_best: Dict[str, float] = {e: 0.0 for e in expected_list}
    for r in recalled_list:
        best_title, best_score, is_hit = "", 0.0, False
        for exp in expected_list:
            sim = title_similarity(r, exp)
            if sim > best_score:
                best_score = sim
                best_title = exp
        is_hit = best_score >= threshold
        matched.append(_TitleMatch(r, best_title, best_score, is_hit))
        if is_hit and best_score > exp_best.get(best_title, 0.0):
            exp_best[best_title] = best_score
    hit_expected = [e for e in expected_list if exp_best.get(e, 0.0) >= threshold]
    return {
        "index": index, "user_query": user_query, "province": province,
        "expected_titles": expected_list, "recalled_titles": recalled_list,
        "hit_expected": hit_expected, "is_hit": len(hit_expected) > 0,
        "hit_ratio": (len(hit_expected) / len(expected_list)) if expected_list else 0.0,
        "avg_sim": (sum(m.best_score for m in matched) / len(matched)) if matched else 0.0,
    }


def _run_eval(
    testset_path: Path, recall_path: Path,
    threshold: float = 0.85, titles_key: Optional[str] = None,
    recall_titles_key: Optional[str] = None,
) -> Dict[str, Any]:
    ts_rows = _load_rows(testset_path)
    rc_rows = _load_rows(recall_path)

    # 聚合召回
    recall_index: Dict[str, List[str]] = {}
    for r in rc_rows:
        q = _pick(r, _RECALL_QUERY_KEYS)
        if not q:
            continue
        titles = _pick_titles(r, _RECALL_TITLE_KEYS, recall_titles_key)
        if titles:
            bucket = recall_index.setdefault(q, [])
            seen = set(bucket)
            for t in titles:
                if t not in seen:
                    bucket.append(t); seen.add(t)

    items: List[Dict[str, Any]] = []
    for row in ts_rows:
        q = _pick(row, _QUERY_KEYS)
        if not q:
            continue
        try:
            idx = int(_pick(row, _INDEX_KEYS) or 0)
        except ValueError:
            idx = 0
        province = _pick(row, _PROVINCE_KEYS)
        expected = _pick_titles(row, _EXPECTED_TITLE_KEYS, titles_key)
        recalled = recall_index.get(q, [])
        items.append(_evaluate_item(recalled, expected, idx, q, province, threshold))

    total = len(items)
    hit_count = sum(1 for it in items if it["is_hit"])
    total_exp = sum(len(it["expected_titles"]) for it in items)
    total_hit = sum(len(it["hit_expected"]) for it in items)
    total_rc = sum(len(it["recalled_titles"]) for it in items)
    return {
        "mode": "eval", "threshold": threshold, "total": total,
        "hit_count": hit_count,
        "item_hit_rate": (hit_count / total) if total else 0.0,
        "total_expected": total_exp, "total_hit_expected": total_hit,
        "recall": (total_hit / total_exp) if total_exp else 0.0,
        "precision": 0.0, "items": items,
    }


# ===========================================================================
# 6. 打印辅助
# ===========================================================================
def _print_rank_summary(rows: List[Dict[str, Any]], top_n: int) -> None:
    total = len(rows)
    print("\n" + "=" * 64)
    print(f"  重排结果概览 (截断 top_n={top_n or '全部'})")
    print("=" * 64)
    print(f"  条目数              : {total}")
    # 逐个 TOP-N 打印统计
    for topn_name in _TOPN_KEYS:
        if not any(topn_name in r for r in rows):
            continue  # 输入中没有这个 TOP-N
        hit = sum(1 for r in rows if r.get(f"{topn_name}_hit_count", 0) > 0)
        total_titles = sum(r.get(f"{topn_name}_count", 0) for r in rows)
        total_hits = sum(r.get(f"{topn_name}_hit_count", 0) for r in rows)
        avg_best = (sum(r.get(f"{topn_name}_best_score", 0.0) for r in rows) /
                    total if total else 0.0)
        print(f"  ── {topn_name} ──")
        print(f"    累计标题数        : {total_titles}")
        print(f"    至少 1 条命中条目 : {hit} ({hit/max(total,1):.2%})")
        print(f"    累计命中(≥0.85)  : {total_hits} ({total_hits/max(total_titles,1):.2%})")
        print(f"    平均最高相似度    : {avg_best:.4f}")
    print("=" * 64)
    # 逐条概览(前 10)
    for r in rows[:10]:
        parts = []
        for topn_name in _TOPN_KEYS:
            if topn_name not in r:
                continue
            hit_c = r.get(f"{topn_name}_hit_count", 0)
            cnt = r.get(f"{topn_name}_count", 0)
            best = r.get(f"{topn_name}_best_score", 0.0)
            parts.append(f"{topn_name}:hit={hit_c}/{cnt} best={best:.2f}")
        print(f"  #{r['index']:>3} | {r['user_query'][:30]}")
        if parts:
            print(f"         {' | '.join(parts)}")
        # 展示 top100 前 3 条(或 top3 全部)
        for topn_name in ("top100", "top3"):
            if topn_name not in r or not r[topn_name]:
                continue
            show_n = 3 if topn_name == "top100" else len(r[topn_name])
            head = ", ".join(
                f"{x['title'][:15]}({x['score']:.2f})" for x in r[topn_name][:show_n]
            )
            if head:
                print(f"         {topn_name}[:{show_n}]: {head}")
            break
    if total > 10:
        print(f"  ...(共 {total} 条,仅展示前 10)")
    print("=" * 64 + "\n")


def _print_eval_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print(f"  召回率评测报告 (threshold={report['threshold']})")
    print("=" * 64)
    print(f"  测试集条目数          : {report['total']}")
    print(f"  条目级命中数          : {report['hit_count']}")
    print(f"  条目级命中率          : {report['item_hit_rate']:.2%}")
    print(f"  标题级召回率(recall)  : {report['recall']:.2%}")
    print("=" * 64 + "\n")


# ===========================================================================
# 7. CLI + 直接运行
# ===========================================================================

def _find_input_file(script_dir: Path) -> Optional[Path]:
    """在脚本同目录下自动寻找合并输入文件(rank 模式)。

    查找优先级:
      1. test.xlsx / test.XLSX(截图中文件名)
      2. 第一个 .xlsx / .xls
      3. 第一个 .json
      4. 第一个 .jsonl
    """
    candidates = []
    for name in ("test.xlsx", "test.XLSX", "test.xls", "test.XLS"):
        p = script_dir / name
        if p.exists():
            return p
    for ext in ("*.xlsx", "*.xls", "*.json", "*.jsonl", "*.csv"):
        matches = sorted(script_dir.glob(ext))
        if matches:
            candidates.extend(matches)
    # 排除自身和已输出文件
    script_stem = Path(__file__).stem
    for c in candidates:
        if script_stem in c.stem:
            continue
        if c.stem.endswith("_ranked") or c.stem.endswith("_eval"):
            continue
        return c
    return None


def main() -> None:
    # ── auto-detect: 无命令行参数时,自动用脚本同目录的文件 ──────────────
    script_dir = Path(__file__).resolve().parent
    auto_input = _find_input_file(script_dir)

    parser = argparse.ArgumentParser(
        description="编辑距离召回重排 + 评测(rank 模式直接运行,同目录放 test.xlsx 即可)")
    parser.add_argument("--mode", choices=("rank", "eval"), default="rank",
                        help="rank=重排模式(默认);eval=评测模式")
    # rank 模式参数
    parser.add_argument("--input", default="",
                        help="重排模式:合并输入文件(.xlsx/.json/.jsonl/.csv);"
                             "缺省自动查找脚本同目录的 test.xlsx")
    parser.add_argument("--top-n", type=int, default=0,
                        help="重排模式:截断前 N 条(0=保留原始条数,默认 0)")
    parser.add_argument("--top3-key", default="",
                        help="重排模式:强制指定 top3 列名(默认自动探测)")
    parser.add_argument("--top100-key", default="",
                        help="重排模式:强制指定 top100 列名(默认自动探测)")
    parser.add_argument("--expected-key", default="",
                        help="重排模式:强制指定期望标题列名(默认自动探测)")
    # eval 模式参数
    parser.add_argument("--testset", default="",
                        help="评测模式:测试集文件(.json/.jsonl/.csv)")
    parser.add_argument("--recall", default="",
                        help="评测模式:召回结果文件(.json/.jsonl/.csv)")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="评测模式:相似度命中阈值(默认 0.85)")
    parser.add_argument("--titles-key", default="",
                        help="评测模式:强制指定测试集期望标题字段名")
    parser.add_argument("--recall-titles-key", default="",
                        help="评测模式:强制指定召回标题字段名")
    # 公共参数
    parser.add_argument("--output", default="",
                        help="输出路径(.json);缺省自动写入同目录 *_ranked.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # ── 解析输入路径:命令行参数优先,否则用 auto-detect ──────────────────
    if args.mode == "rank":
        input_str = args.input or (str(auto_input) if auto_input else "")
        if not input_str:
            raise SystemExit(
                f"未找到输入文件。请将 test.xlsx 放到:\n  {script_dir}\n"
                f"或用 --input 指定路径。")
        input_path = Path(input_str)
        if not input_path.is_absolute():
            input_path = script_dir / input_path
        if not input_path.exists():
            raise SystemExit(f"输入文件不存在: {input_path}")

        rows = _load_rows(input_path)
        print(f"[重排模式] 输入 {input_path.resolve()} → {len(rows)} 行")
        print(f"[配置] top_n={args.top_n}")

        ranked = rank_rows(
            rows, top_n=args.top_n or 0,
            top3_key=args.top3_key or None,
            top100_key=args.top100_key or None,
            expected_titles_key=args.expected_key or None,
        )
        _print_rank_summary(ranked, args.top_n or 0)

        # 输出路径:显式 --output → 缺省 <输入_stem>_ranked.json 同目录
        if args.output:
            out = Path(args.output)
        else:
            out = input_path.with_name(f"{input_path.stem}_ranked.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(ranked, f, ensure_ascii=False, indent=2)
        print(f"[输出] 重排结果已写入: {out.resolve()}")

    else:  # eval 模式
        ts_str = args.testset or ""
        rc_str = args.recall or ""
        if not ts_str or not rc_str:
            raise SystemExit("--mode eval 时必须提供 --testset 和 --recall")
        ts_path = Path(ts_str)
        rc_path = Path(rc_str)
        if not ts_path.is_absolute():
            ts_path = script_dir / ts_path
        if not rc_path.is_absolute():
            rc_path = script_dir / rc_path
        if not ts_path.exists():
            raise SystemExit(f"测试集不存在: {ts_path}")
        if not rc_path.exists():
            raise SystemExit(f"召回结果不存在: {rc_path}")

        print(f"[评测模式] 测试集 {ts_path} / 召回 {rc_path} "
              f"/ threshold={args.threshold}")
        report = _run_eval(
            ts_path, rc_path, threshold=args.threshold,
            titles_key=args.titles_key or None,
            recall_titles_key=args.recall_titles_key or None,
        )
        _print_eval_summary(report)

        if args.output:
            out = Path(args.output)
        else:
            out = script_dir / "eval_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[输出] 评测报告已写入: {out.resolve()}")


if __name__ == "__main__":
    main()
