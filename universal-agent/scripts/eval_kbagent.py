# -*- coding: utf-8 -*-
"""KB-Agent 生产评测脚本 — LLM-as-Judge 评估检索准确性与话术生成质量。

零第三方依赖(纯标准库),可在生产/离线环境直接运行。

评测维度
────────
A. 检索质量(对 kbagent_service 返回的 sources Top3):
   A1  Top3命中率(核心)   : judge 逐篇判 yes/partial/no,≥1 篇 yes → 案例命中
   A1' 客观Top3准确率     : gold 知识ID/标题 与 Top3 docId/docTitle 比对(不经 LLM)
   A2  Precision@3        : (yes×1 + partial×0.5) / 3 的均值
   A3  MRR@3              : 首个 yes 文档排名的倒数,无 yes 记 0
   A4  召回命中率         : gold 与 retrievedDocs(重排前召回)比对,
                           用于区分"检索没捞到" vs "重排排掉了"
B. 话术质量(judge 按 0/1/2 打分):
   B1 回答相关性  B2 忠实性(幻觉检测)  B3 与原文一致性(矛盾一票否决)
   B4 完整性      B5 坐席可用性
   B6 自评校准    : judge 独立给出的 usability vs 系统 usability.level 一致率
   话术通过线     : B1≥1 且 B2≥1 且 B3=2

用法(两阶段,天然支持"实时采集"与"已有结果文件"两种版本)
────────────────────────────────────────────────────────
  # 阶段1 collect:实时调 kbagent_service 采集响应 → responses.jsonl
  python scripts/eval_kbagent.py collect \
      --testset 测试集.xlsx \
      --kbagent-url http://10.x.x.x:8000/api/kb-agent-service/prod/retrieve \
      --out eval_run1 [--limit 50] [--workers 2]

  # 阶段2 judge:读 responses.jsonl,调生产 /llm_397b_api 评审 → verdicts.jsonl + 报表
  python scripts/eval_kbagent.py judge \
      --responses eval_run1/responses.jsonl \
      --llm-url http://10.x.x.x:8002/llm_397b_api \
      --out eval_run1 [--workers 3]

  # 版本2:已有 kbagent 响应文件时,整理成 responses.jsonl(每行一个 JSON,
  # 至少含 {"query": ..., "object": {...kbagent响应object...}})后直接跑 judge。
  # judge 可加 --testset 用 gold 标注补齐客观指标(按 query 文本关联)。

  # 报表可单独重算(不重新调 LLM):
  python scripts/eval_kbagent.py report --verdicts eval_run1/verdicts.jsonl --out eval_run1

输出
────
  responses.jsonl   每案例的 kbagent 原始响应(collect 产出,judge 输入)
  verdicts.jsonl    每案例评审明细(逐案例落盘,断点续跑)
  eval_summary.json 汇总指标
  badcases.csv      失败案例清单(Excel 可直接打开,utf-8-sig)
  eval_details.xlsx 全案例完整明细(明细+汇总两个sheet,40列,纯标准库写出)

断点续跑:collect/judge 启动时读取已有输出文件,自动跳过已完成案例。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import random
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Windows 控制台 GBK 编码兜底,避免中文日志 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

DEFAULT_DOC_CHARS = 3000        # 评审时单篇文档正文截断长度
DEFAULT_KB_TIMEOUT = 600.0      # kbagent 全链路超时(与线上服务一致)
DEFAULT_LLM_TIMEOUT = 300.0     # 单次 judge 调用超时


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 测试集加载(xlsx 用标准库解析;兼容 services/retrieval_service/测试集.json)
# ═══════════════════════════════════════════════════════════════════════════

def _col_index(ref: str, fallback: int) -> int:
    """单元格引用 'B3' → 列下标 1;无引用时退回顺序位置。"""
    m = re.match(r"([A-Z]+)", ref or "")
    if not m:
        return fallback
    idx = 0
    for ch in m.group(1):
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def read_xlsx_rows(path: Path) -> List[List[str]]:
    """最小 xlsx 读取器:返回首个工作表的字符串二维表(含表头行)。

    按单元格引用(r="B3")定位列,空单元格补空串,保证各行与表头列对齐。
    """
    with zipfile.ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{_M}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{_M}t")))
        sheet_names = [n for n in zf.namelist()
                       if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)]
        if not sheet_names:
            raise ValueError(f"{path} 中找不到工作表")
        sheet = ET.fromstring(zf.read(sorted(sheet_names)[0]))
        rows: List[List[str]] = []
        for row in sheet.findall(f".//{_M}row"):
            cells: Dict[int, str] = {}
            for seq, c in enumerate(row.findall(f"{_M}c")):
                v = c.find(f"{_M}v")
                is_node = c.find(f"{_M}is")
                if c.get("t") == "inlineStr" and is_node is not None:
                    text = "".join(t.text or "" for t in is_node.iter(f"{_M}t"))
                elif v is None:
                    text = ""
                elif c.get("t") == "s":
                    text = shared[int(v.text or 0)]
                else:
                    text = v.text or ""
                cells[_col_index(c.get("r", ""), seq)] = text
            if cells:
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
        if rows:
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
        return rows


def _split_cell_list(value: str) -> List[str]:
    """单元格内多值:换行/分号分隔 → 去空列表。"""
    parts = re.split(r"[\n;；]+", str(value or ""))
    return [p.strip() for p in parts if p.strip()]


def _cases_from_xlsx_rows(rows: List[List[str]]) -> List[Dict[str, Any]]:
    """xlsx 二维表 → 评测案例列表,自动识别两种表头格式:

    格式1(测试集.xlsx):   序号/省份/用户问/正确知识数量/知识ID列表/知识标题列表/标注来源列表
    格式2(人工反馈表):     序号/省份/提供日期/反馈人/客户问题/客户问题涉及的知识标题/
                          客户问题涉及的原子名称/原子个数
    - 表头单元格内换行按无空白归一化后匹配;
    - 格式2同一问题可能按"原子"拆成多行 → 按(省份,问题)合并 gold 标注为一个案例。
    """
    if not rows:
        raise ValueError("xlsx 为空")
    header = [re.sub(r"\s+", "", str(h)) for h in rows[0]]

    def col(*names: str) -> int:
        for name in names:
            if name in header:
                return header.index(name)
        return -1

    i_seq = col("序号")
    i_prov = col("省份")
    i_query = col("用户问", "用户问题", "客户问题", "query")
    i_ids = col("知识ID列表", "知识ID")
    i_titles = col("知识标题列表", "知识标题",
                   "客户问题涉及的知识标题", "涉及知识标题")
    if i_query < 0:
        raise ValueError(
            f"xlsx 缺少问题列(用户问/客户问题), 实际表头: {header}")

    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row_no, row in enumerate(rows[1:], 2):
        def cell(idx: int) -> str:
            return str(row[idx]).strip() if 0 <= idx < len(row) else ""
        query = re.sub(r"\s+", " ", cell(i_query)).strip()
        if not query:
            continue
        province = cell(i_prov)
        key = (province, query)
        case = merged.get(key)
        if case is None:
            case = {"case_id": f"c{cell(i_seq) or row_no}", "query": query,
                    "province": province, "gold_ids": [], "gold_titles": []}
            merged[key] = case
        for gold_id in _split_cell_list(cell(i_ids)):
            if gold_id not in case["gold_ids"]:
                case["gold_ids"].append(gold_id)
        for gold_title in _split_cell_list(cell(i_titles)):
            if gold_title not in case["gold_titles"]:
                case["gold_titles"].append(gold_title)
    return list(merged.values())


def load_testset(path: str, province: Optional[str] = None,
                 limit: Optional[int] = None, offset: int = 0,
                 sample_seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载测试集 → [{case_id, query, province, gold_ids, gold_titles}]。

    支持 .xlsx(两种表头格式,见 _cases_from_xlsx_rows)与 .json
    (user_query/province/knowledge_ids)。
    """
    p = Path(path)
    cases: List[Dict[str, Any]] = []
    if p.suffix.lower() in {".xlsx", ".xlsm"}:
        cases = _cases_from_xlsx_rows(read_xlsx_rows(p))
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        for idx, item in enumerate(data, 1):
            cases.append({
                "case_id": f"c{idx}",
                "query": item.get("user_query") or item.get("query") or "",
                "province": item.get("province", ""),
                "gold_ids": list(item.get("knowledge_ids") or []),
                "gold_titles": list(item.get("knowledge_titles") or []),
            })
        cases = [c for c in cases if c["query"]]

    if province:
        cases = [c for c in cases if c["province"] == province]
    # case_id 去重
    seen: Dict[str, int] = {}
    for c in cases:
        if c["case_id"] in seen:
            seen[c["case_id"]] += 1
            c["case_id"] = f"{c['case_id']}-{seen[c['case_id']]}"
        else:
            seen[c["case_id"]] = 1
    if sample_seed is not None and limit and limit < len(cases):
        cases = random.Random(sample_seed).sample(cases, len(cases))
    if offset:
        cases = cases[offset:]
    if limit:
        cases = cases[:limit]
    return cases


# ═══════════════════════════════════════════════════════════════════════════
# HTTP(标准库;--insecure 跳过证书校验,兼容内网自签)
# ═══════════════════════════════════════════════════════════════════════════

def post_json(url: str, payload: Dict[str, Any], timeout: float,
              insecure: bool = False) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="POST")
    ctx = None
    if insecure and url.lower().startswith("https"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_llm(url: str, prompt: str, timeout: float,
             insecure: bool = False) -> str:
    """调生产 /llm_397b_api,返回 answer 文本。"""
    data = post_json(url, {"query": prompt}, timeout, insecure)
    if str(data.get("rtnCode")) != "0":
        raise RuntimeError(f"llm_server 返回错误: {data.get('rtnCode')} "
                           f"{data.get('rtnMsg')}")
    return str((data.get("object") or {}).get("answer") or "")


def call_kbagent(url: str, case: Dict[str, Any], app_id: str,
                 timeout: float, insecure: bool = False) -> Dict[str, Any]:
    """调 kbagent_service /retrieve,返回完整响应信封。"""
    payload = {"params": {
        "appId": app_id,
        "requestId": f"eval-{case['case_id']}-{uuid.uuid4().hex[:8]}",
        "sessionId": f"eval-sess-{case['case_id']}",
        "userInfo": {"phone": "eval_encrypted_000",
                     "province": case.get("province") or "福建"},
        "extInfo": {},
        "conversations": [{"role": 1, "content": case["query"]}],
    }}
    return post_json(url, payload, timeout, insecure)


# ═══════════════════════════════════════════════════════════════════════════
# JSONL 读写(逐行落盘 = checkpoint)
# ═══════════════════════════════════════════════════════════════════════════

class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = path.open("a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # 断点续跑时最后一行可能被截断,跳过
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 阶段1: collect — 实时调 kbagent_service
# ═══════════════════════════════════════════════════════════════════════════

def run_collect(args: argparse.Namespace) -> None:
    cases = load_testset(args.testset, args.province, args.limit,
                         args.offset, args.sample_seed)
    out = Path(args.out)
    writer = JsonlWriter(out / "responses.jsonl")
    done = {r.get("case_id") for r in read_jsonl(out / "responses.jsonl")}
    todo = [c for c in cases if c["case_id"] not in done]
    _log(f"测试集 {len(cases)} 条, 已完成 {len(cases) - len(todo)}, 待采集 {len(todo)}")

    counter = {"ok": 0, "fail": 0}
    lock = threading.Lock()

    def one(case: Dict[str, Any]) -> None:
        record: Dict[str, Any] = {**case, "object": None, "error": None}
        started = time.perf_counter()
        try:
            resp = call_kbagent(args.kbagent_url, case, args.app_id,
                                args.kb_timeout, args.insecure)
            if str(resp.get("rtnCode")) != "0":
                record["error"] = (f"rtnCode={resp.get('rtnCode')} "
                                   f"rtnMsg={resp.get('rtnMsg')}")
            else:
                record["object"] = resp.get("object")
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        writer.write(record)
        with lock:
            if record["error"]:
                counter["fail"] += 1
                _log(f"✗ {case['case_id']} {record['error'][:120]}")
            else:
                counter["ok"] += 1
                obj = record["object"] or {}
                _log(f"✓ {case['case_id']} {record['elapsed_ms']}ms "
                     f"sources={len(obj.get('sources') or [])} "
                     f"degraded={obj.get('degraded')}")

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    writer.close()
    _log(f"采集完成: 成功 {counter['ok']}, 失败 {counter['fail']} → "
         f"{out / 'responses.jsonl'}")


# ═══════════════════════════════════════════════════════════════════════════
# 阶段2: judge — LLM 评审
# ═══════════════════════════════════════════════════════════════════════════

RETRIEVAL_JUDGE_PROMPT = """你是10086客服知识库的检索评测专家。给定用户问题和检索系统返回的Top3文档,请逐篇独立判断该文档与问题的相关性。

[用户问题]
{query}

[候选文档]
{docs}

[判定标准]
- yes: 文档内容与用户问题直接相关,且能回答问题的全部或核心部分
- partial: 文档主题相关,但不足以回答问题(只有背景/周边信息)
- no: 与问题无关,或无法为回答提供任何有效信息

[输出要求]
只输出一个JSON对象,不要任何解释文字,不要markdown代码块标记,格式:
{{"judgments":[{{"doc":"D1","verdict":"yes","reason":"一句话理由"}}]}}
judgments 必须按顺序包含全部候选文档。"""

ANSWER_JUDGE_PROMPT = """你是10086客服话术质量评测专家。给定用户问题、系统检索到的参考文档、以及系统生成的坐席话术,请按以下5个维度独立打分(每维0/1/2分)。

[评分维度]
b1_relevance 回答相关性: 2=直接完整回应用户问题 1=部分回应或避重就轻 0=答非所问
b2_faithfulness 忠实性: 2=话术中所有事实陈述(数字/条件/时间/流程)都能在参考文档中找到依据 1=大部分有依据但个别陈述找不到来源 0=存在多处无依据的编造
b3_consistency 一致性: 2=与参考文档无任何矛盾 1=个别表述不精确但无实质矛盾 0=存在与参考文档直接矛盾的陈述
b4_completeness 完整性: 2=覆盖用户问题的全部关键点 1=遗漏了次要关键点 0=遗漏了核心关键点
b5_usability 坐席可用性: 2=口语化、可直接照读给用户、无内部术语或文档ID泄漏 1=基本可用但需小幅修改 0=无法直接照读

另给出综合可用性判定 judge_usability,三选一:
- directly_usable: 可直接照读
- verify_first: 需人工核实后使用
- not_usable: 不可用

[用户问题]
{query}

[坐席话术]
{script}

[办理建议]
{suggestion}

[参考文档]
{docs}

[输出要求]
只输出一个JSON对象,不要任何解释文字,不要markdown代码块标记,格式:
{{"b1_relevance":2,"b2_faithfulness":2,"b3_consistency":2,"b4_completeness":2,"b5_usability":2,"contradictions":["与原文矛盾的陈述,无则空数组"],"uncovered":["遗漏的关键点,无则空数组"],"reasons":{{"b1":"一句话","b2":"一句话","b3":"一句话","b4":"一句话","b5":"一句话"}},"judge_usability":"directly_usable"}}"""


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + f"…(截断,原文{len(text)}字)"


def _format_docs(sources: Sequence[Dict[str, Any]], doc_chars: int) -> str:
    blocks = []
    for i, s in enumerate(sources[:3], 1):
        blocks.append(
            f"<D{i}> 标题: {s.get('docTitle', '')}\n"
            f"正文: {_clip(s.get('content', ''), doc_chars)}\n</D{i}>")
    return "\n\n".join(blocks) if blocks else "(无检索结果)"


def _extract_json(raw: str) -> Dict[str, Any]:
    """从模型输出提取首个完整 JSON 对象(容忍围栏/前后废话)。"""
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("输出中无 JSON 对象")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("JSON 对象不完整")


def _judge_with_retry(url: str, prompt: str, timeout: float, insecure: bool,
                      validate) -> Tuple[Any, Optional[str]]:
    """调 judge 并校验,失败重试1次;返回 (结果, 错误信息)。"""
    last_err = ""
    current_prompt = prompt
    for attempt in range(2):
        try:
            raw = call_llm(url, current_prompt, timeout, insecure)
            data = _extract_json(raw)
            return validate(data), None
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            current_prompt = (prompt + "\n\n[注意]你上一次的输出无法解析。"
                              "请严格只输出一个JSON对象,不要包含任何其他文字或代码块标记。")
    return None, last_err


def _validate_retrieval(data: Dict[str, Any], expect_n: int) -> List[Dict[str, str]]:
    judgments = data.get("judgments")
    if not isinstance(judgments, list) or not judgments:
        raise ValueError("缺少 judgments 列表")
    out = []
    for j in judgments[:expect_n]:
        verdict = str(j.get("verdict", "")).lower().strip()
        if verdict not in {"yes", "partial", "no"}:
            raise ValueError(f"非法 verdict: {verdict!r}")
        out.append({"doc": str(j.get("doc", "")), "verdict": verdict,
                    "reason": str(j.get("reason", ""))[:200]})
    if len(out) != expect_n:
        raise ValueError(f"judgments 数量 {len(out)} ≠ 候选数 {expect_n}")
    return out


def _score(value: Any) -> int:
    score = int(value)
    return max(0, min(2, score))


def _validate_answer(data: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["b1_relevance", "b2_faithfulness", "b3_consistency",
            "b4_completeness", "b5_usability"]
    scores = {k: _score(data[k]) for k in keys}  # KeyError → 校验失败
    level = str(data.get("judge_usability", "")).strip()
    if level not in {"directly_usable", "verify_first", "not_usable"}:
        raise ValueError(f"非法 judge_usability: {level!r}")
    return {
        **scores,
        "contradictions": [str(x)[:200] for x in (data.get("contradictions") or [])][:5],
        "uncovered": [str(x)[:200] for x in (data.get("uncovered") or [])][:5],
        "reasons": {k: str(v)[:200] for k, v in (data.get("reasons") or {}).items()},
        "judge_usability": level,
    }


# ── 客观指标:gold 匹配(ID 前缀匹配容忍截断;标题归一化后互含匹配) ──────────

def _norm_id(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _norm_title(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or "")).lower()
    return re.sub(r"[《》()（）\[\]【】:：,，.。!！?？'\"’‘“”-]", "", text)


def _id_match(gold: str, actual: str) -> bool:
    g, a = _norm_id(gold), _norm_id(actual)
    if not g or not a:
        return False
    if g == a:
        return True
    # xlsx 中知识ID可能被截断:≥8位时允许前缀匹配
    return len(g) >= 8 and len(a) >= 8 and (a.startswith(g) or g.startswith(a))


def _title_match(gold: str, actual: str) -> bool:
    g, a = _norm_title(gold), _norm_title(actual)
    if not g or not a:
        return False
    return g == a or g in a or a in g


def gold_hit(gold_ids: Sequence[str], gold_titles: Sequence[str],
             actual_ids: Sequence[str], actual_titles: Sequence[str]) -> Optional[bool]:
    """gold 任一命中 actual 任一 → True;无 gold 标注 → None(不计入)。"""
    if not gold_ids and not gold_titles:
        return None
    if not actual_ids and not actual_titles:
        return False
    for g in gold_ids:
        if any(_id_match(g, a) for a in actual_ids):
            return True
    for g in gold_titles:
        if any(_title_match(g, a) for a in actual_titles):
            return True
    return False


# ── 单案例评审 ──────────────────────────────────────────────────────────────

def judge_case(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    case_id = record.get("case_id", "?")
    query = str(record.get("query") or "")
    obj = record.get("object") or {}
    sources = list(obj.get("sources") or [])
    retrieved = list(obj.get("retrievedDocs") or [])
    verdict: Dict[str, Any] = {
        "case_id": case_id, "query": query,
        "province": record.get("province", ""),
        "gold_ids": record.get("gold_ids") or [],
        "gold_titles": record.get("gold_titles") or [],
        "degraded": bool(obj.get("degraded")),
        "collect_error": record.get("error"),
        "system_usability": str((obj.get("usability") or {}).get("level") or ""),
        "script": str(obj.get("script") or ""),
        "suggestion": str(obj.get("handlingSuggestion") or ""),
        "objective": {}, "retrieval": {}, "answer": {},
    }
    if record.get("error") or not obj:
        verdict["service_error"] = record.get("error") or "object 为空"
        return verdict

    top3_ids = [str(s.get("docId") or "") for s in sources[:3]]
    top3_titles = [str(s.get("docTitle") or "") for s in sources[:3]]
    verdict["objective"]["top3_ids"] = top3_ids
    verdict["objective"]["top3_titles"] = top3_titles
    verdict["objective"]["top3_gold_hit"] = gold_hit(
        verdict["gold_ids"], verdict["gold_titles"], top3_ids, top3_titles)
    verdict["objective"]["recall_gold_hit"] = gold_hit(
        verdict["gold_ids"], verdict["gold_titles"],
        [str(d.get("id") or "") for d in retrieved],
        [str(d.get("title") or "") for d in retrieved],
    ) if retrieved else None

    # ── A: 检索评审(有 Top3 才调 LLM) ──
    if sources:
        docs = _format_docs(sources, args.doc_chars)
        prompt = RETRIEVAL_JUDGE_PROMPT.format(query=query, docs=docs)
        n = min(3, len(sources))
        judgments, err = _judge_with_retry(
            args.llm_url, prompt, args.llm_timeout, args.insecure,
            lambda d: _validate_retrieval(d, n))
        if err:
            verdict["retrieval"]["judge_error"] = err
        else:
            verdicts_seq = [j["verdict"] for j in judgments]
            first_yes = next((i for i, v in enumerate(verdicts_seq, 1)
                              if v == "yes"), None)
            verdict["retrieval"] = {
                "judgments": judgments,
                "any_yes": any(v == "yes" for v in verdicts_seq),
                "any_relevant": any(v in ("yes", "partial") for v in verdicts_seq),
                "precision_at_3": round(sum(
                    1.0 if v == "yes" else 0.5 if v == "partial" else 0.0
                    for v in verdicts_seq) / 3.0, 4),
                "rr": round(1.0 / first_yes, 4) if first_yes else 0.0,
            }
    else:
        verdict["retrieval"] = {"any_yes": False, "any_relevant": False,
                                "precision_at_3": 0.0, "rr": 0.0,
                                "judgments": [], "judge_error": None,
                                "note": "sources 为空"}

    # ── B: 话术评审(有话术才调 LLM) ──
    script = str(obj.get("script") or "")
    suggestion = str(obj.get("handlingSuggestion") or "")
    if script or suggestion:
        prompt = ANSWER_JUDGE_PROMPT.format(
            query=query, script=script or "(空)", suggestion=suggestion or "(空)",
            docs=_format_docs(sources, args.doc_chars))
        answer, err = _judge_with_retry(
            args.llm_url, prompt, args.llm_timeout, args.insecure,
            _validate_answer)
        if err:
            verdict["answer"]["judge_error"] = err
        else:
            answer = dict(answer)
            answer["pass"] = (answer["b1_relevance"] >= 1
                              and answer["b2_faithfulness"] >= 1
                              and answer["b3_consistency"] == 2)
            verdict["answer"] = answer
    else:
        verdict["answer"] = {"pass": False, "note": "话术为空",
                             "judge_error": None}
    return verdict


def run_judge(args: argparse.Namespace) -> None:
    responses = read_jsonl(Path(args.responses))
    if not responses:
        raise SystemExit(f"responses 文件为空或不存在: {args.responses}")
    # 可选:用测试集按 query 关联 gold 标注(版本2:自带响应文件无 gold 时)
    if args.testset:
        golds = {c["query"]: c for c in load_testset(args.testset)}
        for r in responses:
            g = golds.get(str(r.get("query") or ""))
            if g and not r.get("gold_ids") and not r.get("gold_titles"):
                r["gold_ids"] = g["gold_ids"]
                r["gold_titles"] = g["gold_titles"]
                r.setdefault("province", g["province"])

    out = Path(args.out)
    writer = JsonlWriter(out / "verdicts.jsonl")
    done = {v.get("case_id") for v in read_jsonl(out / "verdicts.jsonl")}
    todo = [r for r in responses if r.get("case_id") not in done]
    _log(f"响应 {len(responses)} 条, 已评审 {len(responses) - len(todo)}, 待评审 {len(todo)}")

    counter: Dict[str, Any] = {"done": 0, "err": 0, "err_kinds": {}}
    lock = threading.Lock()

    def one(record: Dict[str, Any]) -> None:
        try:
            verdict = judge_case(record, args)
        except Exception as exc:  # noqa: BLE001 - 单案例失败不拖垮整体
            verdict = {"case_id": record.get("case_id", "?"),
                       "query": record.get("query", ""),
                       "fatal_error": f"{type(exc).__name__}: {exc}"}
        writer.write(verdict)
        with lock:
            counter["done"] += 1
            err = (verdict.get("retrieval", {}).get("judge_error")
                   or verdict.get("answer", {}).get("judge_error")
                   or verdict.get("fatal_error")
                   or verdict.get("service_error"))
            if err:
                counter["err"] += 1
                kind = str(err)[:160]
                counter["err_kinds"][kind] = counter["err_kinds"].get(kind, 0) + 1
            r = verdict.get("retrieval", {})
            a = verdict.get("answer", {})
            _log(f"[{counter['done']}/{len(todo)}] {verdict.get('case_id')} "
                 f"any_yes={r.get('any_yes')} gold={verdict.get('objective', {}).get('top3_gold_hit')} "
                 f"pass={a.get('pass')}" + (f" ⚠judge_err: {err}" if err else ""))

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    writer.close()
    _log(f"评审完成: {counter['done']} 条(异常 {counter['err']}) → {out / 'verdicts.jsonl'}")
    if counter["err_kinds"]:
        _log("错误汇总(次数 × 错误):")
        for kind, n in sorted(counter["err_kinds"].items(), key=lambda x: -x[1]):
            _log(f"  {n:>4} × {kind}")
    build_report(read_jsonl(out / "verdicts.jsonl"), out)


# ═══════════════════════════════════════════════════════════════════════════
# 阶段3: report — 汇总 + badcase 导出
# ═══════════════════════════════════════════════════════════════════════════

# ── 极简 xlsx 写出(纯标准库 zipfile+XML, 服务器零依赖; inlineStr 无共享字符串表) ──

_XLSX_INVALID_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _xlsx_escape(text: str) -> str:
    text = _XLSX_INVALID_RE.sub("", str(text))
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def _col_letter(idx: int) -> str:
    """0-based 列号 → Excel 列字母(0→A, 26→AA)。"""
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _sheet_xml(rows: List[List[Any]]) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
             '<worksheet xmlns="http://schemas.openxmlformats.org/'
             'spreadsheetml/2006/main">',
             '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" '
             'topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
             '</sheetView></sheetViews>',
             '<sheetData>']
    for r, row in enumerate(rows, 1):
        parts.append(f'<row r="{r}">')
        for c, value in enumerate(row):
            if value is None or value == "":
                continue
            ref = f"{_col_letter(c)}{r}"
            if isinstance(value, bool):
                value = "TRUE" if value else "FALSE"
            if isinstance(value, (int, float)):
                parts.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                parts.append(
                    f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">'
                    f'{_xlsx_escape(value)}</t></is></c>')
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def write_xlsx(path: Path, sheets: List[Tuple[str, List[List[Any]]]]) -> None:
    """sheets: [(表名, 二维行数据)];首行视为表头(冻结)。"""
    ns_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
    ns_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    ns_doc = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ctypes = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
              f'<Types xmlns="{ns_ct}">',
              '<Default Extension="rels" ContentType="application/'
              'vnd.openxmlformats-package.relationships+xml"/>',
              '<Default Extension="xml" ContentType="application/xml"/>',
              '<Override PartName="/xl/workbook.xml" ContentType="application/'
              'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    workbook_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                     f'<Relationships xmlns="{ns_rel}">']
    sheets_tag = []
    for i, (name, _) in enumerate(sheets, 1):
        ctypes.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType='
            '"application/vnd.openxmlformats-officedocument.'
            'spreadsheetml.worksheet+xml"/>')
        sheets_tag.append(f'<sheet name="{_xlsx_escape(name)}" sheetId="{i}" '
                          f'r:id="rId{i}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{i}" Type="{ns_doc}/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>')
    ctypes.append("</Types>")
    workbook_rels.append("</Relationships>")
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 f'<Relationships xmlns="{ns_rel}">'
                 f'<Relationship Id="rId1" Type="{ns_doc}/officeDocument" '
                 'Target="xl/workbook.xml"/></Relationships>')
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/'
                'spreadsheetml/2006/main" '
                f'xmlns:r="{ns_doc}"><sheets>{"".join(sheets_tag)}'
                '</sheets></workbook>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(ctypes))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for i, (_, rows) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows))


# ── 全量明细导出 ────────────────────────────────────────────────────────────

_DETAIL_HEADER = [
    "case_id", "省份", "query", "失败原因", "degraded", "服务错误", "采集错误",
    "gold_ID", "gold_标题", "Top3_ID", "Top3_标题", "Top3命中gold", "召回命中gold",
    "D1判定", "D1理由", "D2判定", "D2理由", "D3判定", "D3理由",
    "any_yes", "any_relevant", "precision@3", "rr", "检索评审错误",
    "B1相关", "B2忠实", "B3一致", "B4完整", "B5可用", "话术通过",
    "judge可用性", "系统可用性", "B6一致",
    "矛盾陈述", "遗漏点", "judge理由", "话术评审错误", "话术备注",
    "话术", "办理建议",
]


def _detail_row(v: Dict[str, Any], fail_reasons: List[str]) -> List[Any]:
    r = v.get("retrieval", {}) or {}
    a = v.get("answer", {}) or {}
    o = v.get("objective", {}) or {}
    judgments = r.get("judgments") or []

    def jd(i: int, key: str) -> str:
        return str(judgments[i].get(key, "")) if i < len(judgments) else ""

    judge_u = a.get("judge_usability") or ""
    system_u = v.get("system_usability") or ""
    b6 = (judge_u == system_u) if judge_u and system_u else None
    return [
        v.get("case_id"), v.get("province"), v.get("query"),
        ";".join(fail_reasons), v.get("degraded"),
        v.get("service_error") or v.get("fatal_error") or "",
        v.get("collect_error") or "",
        " | ".join(v.get("gold_ids") or []),
        " | ".join(v.get("gold_titles") or []),
        " | ".join(o.get("top3_ids") or []),
        " | ".join(o.get("top3_titles") or []),
        o.get("top3_gold_hit"), o.get("recall_gold_hit"),
        jd(0, "verdict"), jd(0, "reason"),
        jd(1, "verdict"), jd(1, "reason"),
        jd(2, "verdict"), jd(2, "reason"),
        r.get("any_yes"), r.get("any_relevant"),
        r.get("precision_at_3"), r.get("rr"),
        r.get("judge_error") or "",
        a.get("b1_relevance"), a.get("b2_faithfulness"),
        a.get("b3_consistency"), a.get("b4_completeness"),
        a.get("b5_usability"), a.get("pass"),
        judge_u, system_u, b6,
        " | ".join(a.get("contradictions") or []),
        " | ".join(a.get("uncovered") or []),
        "; ".join(f"{k}: {val}" for k, val in (a.get("reasons") or {}).items()),
        a.get("judge_error") or "", a.get("note") or "",
        v.get("script", ""), v.get("suggestion", ""),
    ]


def _summary_rows(summary: Dict[str, Any]) -> List[List[Any]]:
    """eval_summary 拍平成 (分组, 指标, 值) 三列。"""
    rows: List[List[Any]] = [["分组", "指标", "值"]]
    for key, val in summary.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                rows.append([key, k2, v2])
        else:
            rows.append(["总览", key, val])
    return rows


def _rate(values: Sequence[Optional[bool]]) -> Optional[float]:
    effective = [v for v in values if v is not None]
    if not effective:
        return None
    return round(sum(1 for v in effective if v) / len(effective), 4)


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def build_report(verdicts: List[Dict[str, Any]], out: Path) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    total = len(verdicts)
    service_err = [v for v in verdicts if v.get("service_error") or v.get("fatal_error")]
    judged = [v for v in verdicts if v not in service_err]
    retrieval_ok = [v for v in judged if not v.get("retrieval", {}).get("judge_error")]
    answer_ok = [v for v in judged if not v.get("answer", {}).get("judge_error")
                 and "note" not in v.get("answer", {})]

    # A 指标
    a1 = _rate([v.get("retrieval", {}).get("any_yes") for v in retrieval_ok])
    a1_loose = _rate([v.get("retrieval", {}).get("any_relevant") for v in retrieval_ok])
    a1_obj = _rate([v.get("objective", {}).get("top3_gold_hit") for v in judged])
    a2 = _mean([v["retrieval"]["precision_at_3"] for v in retrieval_ok
                if "precision_at_3" in v.get("retrieval", {})])
    a3 = _mean([v["retrieval"]["rr"] for v in retrieval_ok
                if "rr" in v.get("retrieval", {})])
    a4 = _rate([v.get("objective", {}).get("recall_gold_hit") for v in judged])

    # B 指标
    b_keys = ["b1_relevance", "b2_faithfulness", "b3_consistency",
              "b4_completeness", "b5_usability"]
    b_means = {k: _mean([v["answer"][k] for v in answer_ok if k in v.get("answer", {})])
               for k in b_keys}
    b_pass = _rate([v["answer"].get("pass") for v in answer_ok if "pass" in v.get("answer", {})])
    b3_fatal = _rate([v["answer"]["b3_consistency"] == 0 for v in answer_ok
                      if "b3_consistency" in v.get("answer", {})])
    # B6 自评校准:judge_usability vs system usability.level
    pairs = [(v["answer"].get("judge_usability"), v.get("system_usability"))
             for v in answer_ok
             if v.get("answer", {}).get("judge_usability") and v.get("system_usability")]
    b6 = round(sum(1 for j, s in pairs if j == s) / len(pairs), 4) if pairs else None
    # 召回→重排漏斗:A4命中但A1'未命中 = 重排排掉了
    funnel_lost = sum(
        1 for v in judged
        if v.get("objective", {}).get("recall_gold_hit") is True
        and v.get("objective", {}).get("top3_gold_hit") is False)

    summary = {
        "total_cases": total,
        "service_errors": len(service_err),
        "judge_errors": {
            "retrieval": sum(1 for v in judged if v.get("retrieval", {}).get("judge_error")),
            "answer": sum(1 for v in judged if v.get("answer", {}).get("judge_error")),
        },
        "degraded_count": sum(1 for v in judged if v.get("degraded")),
        "A_retrieval": {
            "A1_top3命中率(judge,≥1篇yes)": a1,
            "A1_loose(judge,含partial)": a1_loose,
            "A1_objective_top3准确率(gold比对)": a1_obj,
            "A2_precision@3": a2,
            "A3_mrr@3": a3,
            "A4_召回命中率(gold比对retrievedDocs)": a4,
            "召回命中但Top3丢失(重排损耗)": funnel_lost,
            "有效样本": len(retrieval_ok),
        },
        "B_answer": {
            "话术通过率(B1≥1且B2≥1且B3=2)": b_pass,
            "B3致命矛盾率": b3_fatal,
            **{f"{k}_均分(0-2)": b_means[k] for k in b_keys},
            "B6_自评一致率(judge vs system)": b6,
            "B6_样本数": len(pairs),
            "有效样本": len(answer_ok),
        },
    }

    (out / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # badcases.csv
    def fail_reasons(v: Dict[str, Any]) -> List[str]:
        reasons = []
        if v.get("service_error") or v.get("fatal_error"):
            reasons.append("service_error")
        r, a, o = v.get("retrieval", {}), v.get("answer", {}), v.get("objective", {})
        if r.get("judge_error"):
            reasons.append("retrieval_judge_error")
        if a.get("judge_error"):
            reasons.append("answer_judge_error")
        if v.get("degraded"):
            reasons.append("degraded")
        if r and r.get("any_yes") is False:
            reasons.append("top3无相关文档")
        if o.get("top3_gold_hit") is False:
            reasons.append("top3未命中gold")
        if o.get("recall_gold_hit") is True and o.get("top3_gold_hit") is False:
            reasons.append("重排丢失gold")
        if a.get("pass") is False and "note" not in a:
            for k in b_keys:
                if k in a and (a[k] == 0 or (k == "b3_consistency" and a[k] < 2)):
                    reasons.append(f"{k}={a[k]}")
        if a.get("note") == "话术为空":
            reasons.append("话术为空")
        return reasons

    bad = [(v, fail_reasons(v)) for v in verdicts]
    bad = [(v, rs) for v, rs in bad if rs]
    csv_path = out / "badcases.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "province", "query", "失败原因", "degraded",
                    "top3标题", "any_yes", "gold_top3", "gold_recall",
                    "b1", "b2", "b3", "b4", "b5", "judge_usability",
                    "system_usability", "矛盾陈述", "遗漏点", "话术(截断)",
                    "judge理由"])
        for v, rs in bad:
            a = v.get("answer", {})
            r = v.get("retrieval", {})
            o = v.get("objective", {})
            w.writerow([
                v.get("case_id"), v.get("province"), v.get("query"),
                ";".join(rs), v.get("degraded"),
                " | ".join(o.get("top3_titles") or []),
                r.get("any_yes"), o.get("top3_gold_hit"), o.get("recall_gold_hit"),
                a.get("b1_relevance"), a.get("b2_faithfulness"),
                a.get("b3_consistency"), a.get("b4_completeness"),
                a.get("b5_usability"), a.get("judge_usability"),
                v.get("system_usability"),
                " | ".join(a.get("contradictions") or []),
                " | ".join(a.get("uncovered") or []),
                _clip(v.get("script", ""), 200),
                json.dumps(r.get("judgments") or a.get("reasons") or {},
                           ensure_ascii=False)[:500],
            ])

    # eval_details.xlsx — 全案例完整明细 + 汇总两个 sheet
    # (纯标准库写出,无 openpyxl 的服务器也能生成)
    detail_rows = [_DETAIL_HEADER]
    for v in verdicts:
        detail_rows.append(_detail_row(v, fail_reasons(v)))
    xlsx_path = out / "eval_details.xlsx"
    write_xlsx(xlsx_path, [("明细", detail_rows),
                           ("汇总", _summary_rows(summary))])
    _log(f"完整明细已导出: {len(verdicts)} 条(失败 {len(bad)} 条) → {xlsx_path}")

    # 控制台报告
    def pct(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x * 100:.1f}%"

    def num(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.3f}"

    A, B = summary["A_retrieval"], summary["B_answer"]
    print("\n" + "═" * 62)
    print(f"  评测报告  案例总数={total}  服务错误={len(service_err)}  "
          f"降级={summary['degraded_count']}")
    print("═" * 62)
    print("  A. 检索质量")
    print(f"    A1  Top3命中率(judge ≥1篇yes)      : {pct(a1)}")
    print(f"    A1' 客观Top3准确率(gold比对)       : {pct(a1_obj)}")
    print(f"    A1~ 宽松命中(含partial)            : {pct(a1_loose)}")
    print(f"    A2  Precision@3                    : {num(a2)}")
    print(f"    A3  MRR@3                          : {num(a3)}")
    print(f"    A4  召回命中率(retrievedDocs)      : {pct(a4)}")
    print(f"    └─ 召回命中但Top3丢失(重排损耗)   : {funnel_lost} 例")
    print("  B. 话术质量")
    print(f"    话术通过率(B1≥1,B2≥1,B3=2)        : {pct(b_pass)}")
    for k in b_keys:
        print(f"    {k:<28s}: {num(b_means[k])} / 2")
    print(f"    B3  致命矛盾率                     : {pct(b3_fatal)}")
    print(f"    B6  自评一致率(样本{len(pairs)})          : {pct(b6)}")
    print("─" * 62)
    print(f"  badcase {len(bad)} 例 → {csv_path}")
    print(f"  汇总 → {out / 'eval_summary.json'}")
    print("═" * 62 + "\n")
    return summary


def run_report(args: argparse.Namespace) -> None:
    verdicts = read_jsonl(Path(args.verdicts))
    if not verdicts:
        raise SystemExit(f"verdicts 文件为空或不存在: {args.verdicts}")
    build_report(verdicts, Path(args.out))


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KB-Agent 生产评测(LLM-as-Judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out", required=True, help="输出目录")
        sp.add_argument("--workers", type=int, default=2, help="并发数(默认2)")
        sp.add_argument("--insecure", action="store_true",
                        help="https 时跳过证书校验(内网自签)")

    p_collect = sub.add_parser("collect", help="实时调 kbagent_service 采集响应")
    p_collect.add_argument("--testset", required=True, help="测试集 .xlsx 或 .json")
    p_collect.add_argument("--kbagent-url", required=True,
                           help="kbagent_service /retrieve 完整地址")
    p_collect.add_argument("--app-id", default="eval-script", help="请求 appId")
    p_collect.add_argument("--kb-timeout", type=float, default=DEFAULT_KB_TIMEOUT)
    p_collect.add_argument("--limit", type=int, default=None, help="只跑前N条")
    p_collect.add_argument("--offset", type=int, default=0)
    p_collect.add_argument("--province", default=None, help="只跑指定省份")
    p_collect.add_argument("--sample-seed", type=int, default=None,
                           help="随机抽样种子(与 --limit 联用)")
    common(p_collect)
    p_collect.set_defaults(func=run_collect)

    p_judge = sub.add_parser("judge", help="读 responses.jsonl 调 llm_server 评审")
    p_judge.add_argument("--responses", required=True, help="collect 产出的 jsonl")
    p_judge.add_argument("--llm-url", required=True,
                         help="生产 llm_server 地址,如 http://IP:8002/llm_397b_api")
    p_judge.add_argument("--llm-timeout", type=float, default=DEFAULT_LLM_TIMEOUT)
    p_judge.add_argument("--doc-chars", type=int, default=DEFAULT_DOC_CHARS,
                         help="评审时单篇文档截断长度(默认3000)")
    p_judge.add_argument("--testset", default=None,
                         help="可选:用测试集按 query 补齐 gold 标注")
    common(p_judge)
    p_judge.set_defaults(func=run_judge)

    p_report = sub.add_parser("report", help="由 verdicts.jsonl 重算报表")
    p_report.add_argument("--verdicts", required=True)
    p_report.add_argument("--out", required=True)
    p_report.set_defaults(func=run_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
