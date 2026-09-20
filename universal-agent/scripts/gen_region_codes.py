# -*- coding: utf-8 -*-
"""从《知识库统一检索接口》docx §6「省份地市编码列表」生成完整编码映射模块。

用法::

    python scripts/gen_region_codes.py "D:\\cmos\\Agent-知识库\\材料\\知识库统一检索接口V1.6.9-20260622.docx"

输出: src/kbagent/shared/region_codes.py (自动生成,勿手改)。

规则:
- 表0(省份编码/省份名称) → PROVINCE_CODE_TO_NAME;
- 表1(地区编码/地区名称) → CITY_CODE_TO_NAME(编码重复时首见优先);
- REGION_NAME_TO_CODE(名称→编码, 检索透传换算用):
  * 省份名最高优先(如 河北→311);
  * 地市名唯一(或与另一候选仅差前导零, 如 衡水→0318/318)时收录, 前导零
    变体优先取无前导零形式(与省份编码风格一致);
  * 一名多码且非前导零变体(如 朝阳→010I/0421)视为歧义, 不自动换算,
    收录在 AMBIGUOUS_CITY_NAME_TO_CODES 供调用方自行消歧。
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJ_ROOT / "src" / "kbagent" / "shared" / "region_codes.py"


def _cell_text(tc_xml: str) -> str:
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", tc_xml)).strip()


def _extract_tables(docx_path: Path):
    """返回 [(header_cells, [(code, name), ...]), ...] 全部表格。"""
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    tables = []
    for tbl in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        rows = []
        for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S):
            cells = [_cell_text(tc) for tc in re.findall(r"<w:tc[ >].*?</w:tc>", tr, re.S)]
            if cells:
                rows.append(cells)
        if rows:
            tables.append((rows[0], rows[1:]))
    return tables


def _pick_table(tables, header_key: str):
    for header, rows in tables:
        if header and header_key in header[0]:
            pairs = OrderedDict()
            for r in rows:
                if len(r) >= 2 and r[0] and r[1]:
                    pairs.setdefault((r[0].strip(), r[1].strip()), None)
            return [(c, n) for c, n in pairs]
    raise SystemExit(f"未找到表头含 {header_key!r} 的表格")


def _fmt_dict(d, per_line=4, indent="    "):
    """紧凑 dict 字面量输出。"""
    if not d:
        return "{}"
    items = [f"{k!r}: {v!r}" for k, v in d.items()]
    lines = []
    for i in range(0, len(items), per_line):
        lines.append(indent + ", ".join(items[i:i + per_line]) + ",")
    return "{\n" + "\n".join(lines) + "\n}"


def _fmt_dict_of_lists(d, indent="    "):
    if not d:
        return "{}"
    items = [f"{k!r}: {list(v)!r}" for k, v in d.items()]
    lines = [indent + it + "," for it in items]
    return "{\n" + "\n".join(lines) + "\n}"


def build(docx_path: Path) -> str:
    tables = _extract_tables(docx_path)
    provinces = _pick_table(tables, "省份编码")
    cities = _pick_table(tables, "地区编码")

    province_code_to_name = OrderedDict((c, n) for c, n in provinces)
    city_code_to_name = OrderedDict()
    for c, n in cities:
        city_code_to_name.setdefault(c, n)  # 编码重复首见优先

    # 地市 名称 → 编码集合(保持文档顺序)
    city_name_to_codes = OrderedDict()
    for c, n in cities:
        city_name_to_codes.setdefault(n, [])
        if c not in city_name_to_codes[n]:
            city_name_to_codes[n].append(c)

    def _collapse(codes):
        """前导零变体归并: 0318/318 视为同一地区, 取无前导零形式。"""
        out = []
        seen_stripped = {}
        for c in codes:
            s = c.lstrip("0") or c
            if s in seen_stripped:
                # 已有变体, 保留更短(无前导零)的那个
                prev = seen_stripped[s]
                if len(c) < len(prev):
                    out[out.index(prev)] = c
                    seen_stripped[s] = c
                continue
            seen_stripped[s] = c
            out.append(c)
        return out

    name_to_code = OrderedDict()
    # 1) 省份名最高优先
    for c, n in provinces:
        name_to_code[n] = c
    # 2) 地市名: 唯一或仅前导零变体 → 收录; 歧义 → 单独存放
    ambiguous = OrderedDict()
    for n, codes in city_name_to_codes.items():
        if n in name_to_code:
            continue  # 省份名已占用
        collapsed = _collapse(codes)
        if len(collapsed) == 1:
            name_to_code[n] = collapsed[0]
        else:
            ambiguous[n] = codes

    src = io.StringIO()
    w = src.write
    w('# -*- coding: utf-8 -*-\n')
    w('"""省份地市编码全量映射表 — 本文件由脚本自动生成, 请勿手改。\n\n')
    w(f'数据源: {docx_path.name} §6 省份地市编码列表\n')
    w(f'再生成: python scripts/gen_region_codes.py "{docx_path.as_posix()}"\n\n')
    w('- PROVINCE_CODE_TO_NAME: 省份编码 → 省份名称 (%d 条)\n' % len(province_code_to_name))
    w('- CITY_CODE_TO_NAME:     地区编码 → 地区名称 (%d 条, 编码重复首见优先)\n' % len(city_code_to_name))
    w('- REGION_NAME_TO_CODE:   名称 → 编码 (检索阶段透传换算; 省份名优先,\n')
    w('                         地市名仅收录无歧义者, 前导零变体取无前导零形式)\n')
    w('- AMBIGUOUS_CITY_NAME_TO_CODES: 一名多码的地市名 → 全部候选编码,\n')
    w('                         不参与自动换算, 由调用方消歧后透传\n\n')
    w('透传约定: 调用方传入的值若已是编码(或表外未知值), 检索链路原样透传;\n')
    w('仅当传入的是可唯一换算的名称时才替换为编码。\n')
    w('"""\n')
    w('from __future__ import annotations\n\n')
    w('from typing import Dict, List\n\n\n')
    w(f'PROVINCE_CODE_TO_NAME: Dict[str, str] = {_fmt_dict(province_code_to_name)}\n\n\n')
    w(f'CITY_CODE_TO_NAME: Dict[str, str] = {_fmt_dict(city_code_to_name)}\n\n\n')
    w(f'REGION_NAME_TO_CODE: Dict[str, str] = {_fmt_dict(name_to_code)}\n\n\n')
    w('AMBIGUOUS_CITY_NAME_TO_CODES: Dict[str, List[str]] = '
      f'{_fmt_dict_of_lists(ambiguous)}\n\n\n')
    w('''def normalize_region(value: str) -> str:
    """名称 → 编码; 已是编码或表外未知值原样返回(透传)。"""
    if not value:
        return value
    return REGION_NAME_TO_CODE.get(value.strip(), value)


def is_known_region_code(code: str) -> bool:
    """编码是否在省份/地市编码表内(容忍前导零差异, 如 0591↔591)。"""
    if not code:
        return False
    c = code.strip()
    variants = {c, c.lstrip("0") or c, "0" + c}
    return bool(variants & (set(PROVINCE_CODE_TO_NAME) | set(CITY_CODE_TO_NAME)))


def region_code_to_name(code: str) -> str:
    """编码 → 名称; 未知编码返回空串。省份表优先。"""
    if not code:
        return ""
    c = code.strip()
    if c in PROVINCE_CODE_TO_NAME:
        return PROVINCE_CODE_TO_NAME[c]
    if c in CITY_CODE_TO_NAME:
        return CITY_CODE_TO_NAME[c]
    stripped = c.lstrip("0") or c
    for table in (PROVINCE_CODE_TO_NAME, CITY_CODE_TO_NAME):
        for k, v in table.items():
            if (k.lstrip("0") or k) == stripped:
                return v
    return ""
''')
    content = src.getvalue()

    stats = (f"provinces={len(province_code_to_name)} cities={len(city_code_to_name)} "
             f"name_to_code={len(name_to_code)} ambiguous={len(ambiguous)}")
    print(stats)
    return content


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docx", help="知识库统一检索接口 docx 路径")
    args = ap.parse_args()
    content = build(Path(args.docx))
    OUT_PATH.write_text(content, encoding="utf-8")
    print(f"written: {OUT_PATH} ({len(content)} chars)")


if __name__ == "__main__":
    sys.exit(main())
