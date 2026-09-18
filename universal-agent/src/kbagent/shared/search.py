# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import requests
from jinja2 import Template as JinjaTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from .models import Chunk, RetrievalParams, new_id

logger = logging.getLogger("kbagent.search")

#导入各类请求的模板配置
from .search_config import (
    _ATOM_RECALL_TEMPLATE, #原子请求模板
    _INFO_RECALL_TEMPLATE, #信息请求模板
    _NGKM_SEARCH_URL, #ngkm 检索接口
    _PROVINCE_TO_REGION, #省份名 → 区号 映射
    _VECTOR_CONTENT_FIELD, #vector 搜索内容字段
    _VECTOR_ID_FIELD, #vector 搜索ID字段
    _VECTOR_PAYLOADS, #vector 搜索请求体
    _VECTOR_SEARCH_URL, #vector 搜索接口
    _VECTOR_TITLE_FIELD, #vector 搜索标题字段
)
def _region_code(value: str) -> str:
    """省份名 → 区号;已是区号(或其他值)原样返回。"""
    return _PROVINCE_TO_REGION.get(value, value)

def _preview(value: Any, limit: int = 800) -> str:
    """日志安全预览:dict/list 转 JSON,控制台换行压成空格,超长截断。"""
    try:
        s = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        s = str(value)
    s = s.replace("\n", " ").replace("\r", " ")
    return s if len(s) <= limit else s[:limit] + f"...(共{len(s)}字符)"

def _extract_doc_list(parsed: Any) -> List[dict]:
    """从 ngkm 检索响应的 object 解析结果中提取知识条目列表。

    生产响应结构:object 是 JSON 对象,真正的条目列表在 ``docment`` 字段;
    兼容解析结果本身就是 list 的旧结构;dict 且无 docment 字段时按单条处理。
    """
    if isinstance(parsed, list):
        return [d for d in parsed if isinstance(d, dict)]
    if isinstance(parsed, dict):
        for key in ("document","data"):
            docs = parsed.get(key)
            if isinstance(docs, list):
                # logger.info("ngkm 响应从字段 %r 提取条目列表,共 %d 条",
                #             key, len(docs))
                return [d for d in docs if isinstance(d, dict)]
        return [parsed] if parsed else []
    return []

def vresult_to_chunks(resp: Any, source: str = "vector") -> List[Chunk]:
    """处理 vector_search 返回的结果 → 标准 Chunk 列表。

    支持两种输入:
    1. both 模式返回的 dict(含 new/old/all 键):直接处理 all 条目列表(已去重);
    2. new/old 单路原始响应:walk 提取条目(兼容 object JSON 字符串包裹)。
    """
    chunks: List[Chunk] = []
    seen: set = set()

    def _info_of(entry: Dict[str, Any]) -> Dict[str, Any]:
        info = entry.get("info")
        return info if isinstance(info, dict) else {}

    def add_entry(entry: Dict[str, Any]) -> None:
        info = _info_of(entry)
        kid = str(entry.get(_VECTOR_ID_FIELD) or info.get(_VECTOR_ID_FIELD) or "")
        title = str(entry.get(_VECTOR_TITLE_FIELD) or info.get(_VECTOR_TITLE_FIELD) or "")
        content = str(entry.get(_VECTOR_CONTENT_FIELD) or "") or title
        key = kid or new_id("vec")
        if kid and key in seen:
            return
        seen.add(key)
        chunks.append(Chunk(
            chunk_id=f"{kid}" if kid else key,
            doc_id=kid or "unknown",
            doc_title=title,
            content=content,
            category="",
            position={"knowledge_id": kid, "rank": len(chunks)},
            score=0.0,
            extra={"source": "vector",
                   "vector_channel": source,
                   "raw": entry},
        ))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            raw_obj = node.get("object")
            if isinstance(raw_obj, str):
                try:
                    walk(json.loads(raw_obj))
                except (json.JSONDecodeError, TypeError):
                    pass
            info = _info_of(node)
            has_id = (node.get(_VECTOR_ID_FIELD) not in (None, "")
                      or info.get(_VECTOR_ID_FIELD) not in (None, ""))
            has_title = (node.get(_VECTOR_TITLE_FIELD) not in (None, "")
                         or info.get(_VECTOR_TITLE_FIELD) not in (None, ""))
            has_content = node.get(_VECTOR_CONTENT_FIELD) not in (None, "")
            if (has_id or has_title) and has_content:
                add_entry(node)
                return
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    if isinstance(resp, dict) and "all" in resp:
        for entry in resp["all"]:
            if isinstance(entry, dict):
                add_entry(entry)
    else:
        walk(resp)
    return chunks

def kresult_to_chunks(kresult: List[Dict[str, Any]]) -> List[Chunk]:
    """一体化流水线的知识条目(info+atoms)→ 标准 Chunk 列表。

    一条知识映射一个 Chunk:content 由原子字段拼接(参数名:内容),
    原始条目完整保留在 extra 供溯源;生产侧无显式相关性得分,按出现顺序衰减。
    """
    chunks: List[Chunk] = []
    for rank, entry in enumerate(kresult or []):
        if not isinstance(entry, dict):
            continue
        kid = str(entry.get("knowledgeId") or entry.get("knowledge_id") or "")
        title = str(entry.get("knowledgeName") or entry.get("knowledge_name") or "")
        lines: List[str] = []
        atoms = entry.get("atoms") or []
        for atom in atoms:
            if not isinstance(atom, dict) or atom.get("error"):
                continue
            name = str(atom.get("paramName") or "").strip()
            text = str(atom.get("content") or "").strip()
            if not text:
                continue
            lines.append(f"{name}:{text}" if name else text)
        content = "\n".join(lines) or str(entry.get("content") or "") or title
        if not content:
            continue
        updated_at = ""
        for key in ("updateTime", "update_time", "srcTime", "createTime"):
            if entry.get(key):
                updated_at = str(entry[key])
                break
        chunks.append(Chunk(
            chunk_id=f"{kid}" if kid else new_id("ngkm"),
            doc_id=kid or "unknown",
            doc_title=title,
            content=content,
            category=str(entry.get("category") or ""),
            position={"knowledge_id": kid},
            updated_at=updated_at,
            score=round(max(0.5, 1.0 - 0.05 * rank), 4),
            extra={"region_code": str(entry.get("region_code") or ""),
                   "status": str(entry.get("status") or ""),
                   "source": "ngkm",
                   "atoms": atoms},
        ))
    if len(chunks) < len(kresult or []):
        # logger.info("kresult_to_chunks: %d 条知识条目 → %d 条有效 Chunk"
        #             "(无内容/原子全失败的条目被丢弃)",
        #             len(kresult or []), len(chunks))
        pass
    for c in chunks[:5]:
        # logger.info("kresult_to_chunks 产出: id=%s title=%r content_len=%d",
        #             c.chunk_id, c.doc_title, len(c.content))
        pass
    return chunks

def get_kid_score(keyword_kid: List[str], vector_kid: List[str]) -> Dict[str, float]:
    """根据 keyword/vector 两路 kid 列表计算每个 kid 的得分。
    """
    keyword_set = set(keyword_kid or [])
    vector_set = set(vector_kid or [])
    scores: Dict[str, float] = {}
    for kid in keyword_set | vector_set:
        score = 0.0
        if kid in keyword_set:
            score += 1.0
        if kid in vector_set:
            score += 1.0
        scores[kid] = score
    return scores

class ESClient(ABC):
    @abstractmethod
    def keyword_search(self, dsl: Dict[str, Any]) -> List[Chunk]: ...

    @abstractmethod
    def vector_search(self, query_text: str, filters: Dict[str, str],
                      size: int = 10, vector_mode: str = "both",
                      vector_weights: Optional[Dict[str, float]] = None
                      ) -> Any: ...

class ProduceESClient(ESClient):

    def __init__(self, region_code: str = "000", model: Any = None,
                 timeout: int = 30):
        self.region_code = region_code      # 支持省份名,内部自动转区号
        self.model = model                  # BaseChatModel,供 _extract_keywords 调用
        self.timeout = timeout              # ngkm HTTP 超时(秒)
    def vector_search(self, query_text: str, region_code: str, vector_mode: str = "both") -> Any:
        """向量通道:按 ``vector_mode`` 选择召回函数,直接返回对应结果。

        new/old 单路只请求对应模板;both 时按 vector_weights 混合召回。
        """
        if not query_text or not query_text.strip():
            return None
        mode = vector_mode 
        province = _region_code(region_code or "")
        if mode == "new":
            return self._vector_recall_new(query_text, province)
        if mode == "old":
            return self._vector_recall_old(query_text, province)
        return self._vector_recall(query_text, province)

    def _vector_recall_new(self, query_text: str, province: str) -> Optional[dict]:
        try:
            resp = self._post_vector_search("new", query_text, province=province)
            # logger.info("向量召回(新模板) query=%r 完成", query_text)
            return resp
        except Exception as exc:
            # logger.warning("向量召回(新模板)失败,降级为空 query=%r err=%r",
            #                query_text, exc)
            return None

    def _vector_recall_old(self, query_text: str, province: str) -> Optional[dict]:
        try:
            resp = self._post_vector_search("old", query_text, province=province)
            # logger.info("向量召回(旧模板) query=%r 完成", query_text)
            return resp
        except Exception as exc:  # noqa: BLE001
            # logger.warning("向量召回(旧模板)失败,降级为空 query=%r err=%r",
            #                query_text, exc)
            return None

    def _vector_recall(self, query_text: str, province: str) -> Dict[str, Any]:
        """根据权重召回混合的新旧模板响应,返回各路原始响应 + all(去重合并)。

        权重 > 0 的路才请求;返回 {"new": raw, "old": raw, "all": [...]}
        (new/old 为原始响应,all 为两路条目按 kid 去重后的合并列表)。
        """
        results: Dict[str, Any] = {}

        results["new"] = self._vector_recall_new(query_text, province)
        results["old"] = self._vector_recall_old(query_text, province)

        seen_kids: set = set()
        all_entries: List[dict] = []
        for source in ("new", "old"):
            raw = results.get(source)
            if not raw:
                continue
            entries = _extract_doc_list(raw)
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                kid = str(entry.get(_VECTOR_ID_FIELD)
                         or "")
                if kid and kid in seen_kids:
                    continue
                if kid:
                    seen_kids.add(kid)
                all_entries.append(entry)
        results["all"] = all_entries
        return results

    def _post_vector_search(self, mode: str, query_text: str, province: str = "") -> dict:
        payload = dict(_VECTOR_PAYLOADS[mode])
        payload["content"] = query_text
        payload["provinceId"] = province
        payload["reqId"] = uuid.uuid4().hex
        payload["xTransId"] = uuid.uuid4().hex
        # logger.info("向量检索请求 mode=%s content=%r top=%s embeddingTop=%s "
        #             "isEnableNewRouteExp=%s",
        #             mode, query_text, payload.get("top"),
        #             payload.get("embeddingTop"), payload.get("isEnableNewRouteExp"))
        try:
            resp = requests.post(
                _VECTOR_SEARCH_URL,
                headers={"Content-Type": "application/json"},
                json=payload
            )
        except Exception as exc:  # noqa: BLE001
            # logger.warning("向量检索请求失败(网络/超时/DNS) mode=%s url=%s err=%r",
            #                mode, _VECTOR_SEARCH_URL, exc)
            raise
        # logger.info("向量检索响应 mode=%s status=%s",
        #             mode, resp.status_code)
        resp.raise_for_status()
        return resp.json()

    def keyword_search(self, query: str, region_code: str = "",
                       keywords: Optional[List[str]] = None,
                       timeout: int = 30) -> dict:
        """完整流水线,返回 {keywords, knowledge_ids, info, atom, merged_count, merged}。
        若显式传入 keywords,直接使用,跳过 _extract_keywords 调用。
        """
        region_code = _region_code(region_code or self.region_code)
        # logger.info("keyword_search 开始 query=%r region=%r→%r "
        #             "索引=ngkm.knowledges_%s / ngkm.knowledge_atom_%s",
        #             query, region_code, region_code, region_code)
        if keywords:
            # logger.info("keyword_search 外部传入 keywords,跳过提取: %s", keywords)
            pass
        else:
            try:
                keywords = self._extract_keywords(query)
                # logger.info("槽位提取关键词 query=%r: %s", query, keywords)
            except Exception as exc:  # noqa: BLE001
                # logger.warning("keyword_search 槽位提取异常,流水线终止: %r", exc)
                return {"error": f"keyword 调用失败: {exc}", "merged": []}
        if not keywords:
            # logger.warning("keyword_search 未提取到有效关键词 → 零召回 query=%r", query)
            return {"keywords": [], "info": [], "atom": [], "merged": [],
                    "message": "未提取到有效关键词"}
        return self._info_atom_recall(keywords, region_code, timeout)

    def _extract_keywords(self, query: str) -> List[str]:
        from ..retrieval.prompt import _KEYWORD_EXTRACT_SYSTEM
        print("━━━ _extract_keywords query =", repr(query))                # 加这行
        print("    self.model is None =", self.model is None)              # ← 重点看这行
        if self.model is None:
            print("    → fallback 1: model is None, 返回原始 query")       # 加
            return [query.strip()] if query.strip() else []
        t0 = time.time()
        try:
            resp = self.model.invoke([
                SystemMessage(content=_KEYWORD_EXTRACT_SYSTEM),
                HumanMessage(content=f"用户问题:{query}"),
            ])
        except Exception as exc:  # noqa: BLE001
            print("    → LLM 调用异常:", repr(exc))
            # logger.warning("关键词提取 LLM 调用异常: %r", exc)
            raise
        elapsed = time.time() - t0
        raw = str(getattr(resp, "content", resp))
        print(f"    LLM 耗时={elapsed:.2f}s raw={raw[:300]!r}")
        # logger.info("关键词提取 LLM 返回 耗时%.1fs 长度=%d 内容=%s",
        #             elapsed, len(raw), raw[:500].replace("\n", " "))
        cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
        print(f"    cleaned={cleaned[:300]!r}")
        print("    LLM raw =", repr(raw[:200]))                            # 加
        print("    LLM cleaned =", repr(cleaned[:200]))                    # 加

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            print("    → fallback 2: JSON 解析失败, err=", exc)            # 加
            # logger.warning("关键词提取 JSON 解析失败 err=%s 原始=%s",
            #                exc, cleaned[:300])
            return [query.strip()] if query.strip() else []
        keywords = data.get("keywords", []) if isinstance(data, dict) else []
        print(f"    JSON data={data!r}  keywords(raw)={keywords!r}")
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        seen: set = set()
        deduped = [str(k).strip() for k in keywords
                   if k and not (str(k).strip() in seen or seen.add(str(k).strip()))]
        # logger.info("关键词提取完成 query=%r → keywords=%s", query, deduped)
        print("    deduped =", deduped)                                    # 加
        if not deduped:
            print("    → fallback 3: deduped 空,返回原始 query")            # 加
            # logger.warning("关键词提取返回空,降级为原始 query")
            return [query.strip()] if query.strip() else []
        print(f"    ✓ 正常返回 deduped={deduped!r}")
        return deduped#此处返回的是["k1","k2","k3"]
    
    # def _extract_keywords(self, query: str) -> List[str]: #优化槽位提取结果，只保留有效信息（代办）
    #     """Step 1:槽位抽取服务提取检索关键词。"""
    #     payload = {
    #         "query": query,
    #         "context": {
    #             "app_id": "hint_server",
    #             "province_id": "test_pro",
    #             "channel_id": "web",
    #         },
    #         "confidence_threshold": 0.5,
    #     }
    #     try:
    #         resp = requests.post(_SLOT_EXTRACT_URL,
    #                              headers={"Content-Type": "application/json"},
    #                              json=payload, timeout=self.timeout)
    #     except Exception as exc:  # noqa: BLE001
    #         logger.warning("槽位提取请求失败(网络/超时/DNS) url=%s err=%r",
    #                        _SLOT_EXTRACT_URL, exc)
    #         raise
    #     if resp.status_code != 200:
    #         logger.warning("槽位提取返回非200 status=%s body=%s",
    #                        resp.status_code, resp.text[:300])
    #     resp.raise_for_status()
    #     data = resp.json()
    #     logger.info("槽位提取原始响应 query=%r: %s", query, _preview(data))
    #     slots = data.get("slots", []) if isinstance(data, dict) else []
    #     keywords: List[str] = []
    #     for slot in slots:
    #         raw = slot.get("slot_value", "") if isinstance(slot, dict) else ""
    #         if not raw:
    #             continue
    #         try:
    #             parsed = ast.literal_eval(raw)
    #             if isinstance(parsed, (list, tuple)):
    #                 keywords.extend(str(v).strip() for v in parsed if v)
    #             else:
    #                 keywords.append(str(parsed).strip())
    #         except (ValueError, SyntaxError):
    #             keywords.append(raw.strip())
    #     seen: set = set()
    #     deduped = [k for k in keywords if k and not (k in seen or seen.add(k))]
    #     logger.info("槽位提取完成 query=%r slots=%d个 → keywords=%s",
    #                 query, len(slots), deduped)
    #     if not deduped:
    #         logger.warning("槽位提取返回空关键词,响应体=%s", str(data)[:300])
    #     return deduped
    
    def _info_atom_recall(self, keywords: List[str], region_code: str,
                          timeout: int) -> dict:
        """Step 2-4:info 召回 → 按 knowledgeId 拉 atom → 合并。"""
        region_code = _region_code(region_code)

        info_eg: dict = {}
        info_parsed_eg: Any = {}
        info_list_eg: List[dict] = []
        atom_eg: dict = {}
        atom_parsed_eg: Any = {}
        atom_list_eg: List[dict] = []

        # ---- Step 2: 收集所有 info 条目(跨所有 keyword) ----
        all_infos: List[dict] = []
        for kw in keywords:
            try:
                info_resp = self._get_info(keyword=kw, region_code=region_code,
                                           timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                # logger.warning("info 召回失败,跳过该关键词 keyword=%r "
                #                "索引=ngkm.knowledges_%s err=%r",
                #                kw, region_code, exc)
                continue
            # logger.info("info 原始响应 keyword=%r: %s", kw, _preview(info_resp))
            raw_obj = info_resp.get("object", "") if isinstance(info_resp, dict) else ""
            try:
                parsed = json.loads(raw_obj) if isinstance(raw_obj, str) else raw_obj or {}
            except json.JSONDecodeError as exc:
                # logger.warning("info 响应 object 非 JSON,跳过 keyword=%r err=%r raw=%s",
                #                kw, exc, str(raw_obj)[:200])
                continue
            infos = _extract_doc_list(parsed)
            if not info_eg and isinstance(info_resp, dict):
                info_eg = dict(info_resp)
            if not info_parsed_eg:
                info_parsed_eg = parsed
            if not info_list_eg and infos:
                info_list_eg = [dict(d) for d in infos[:1] if isinstance(d, dict)]
            for info in infos:
                if not isinstance(info, dict):
                    continue
                info["_keyword"] = kw
                all_infos.append(info)
            # logger.info("info 召回 keyword=%r → %d 条", kw, len(infos))
            for i, info in enumerate(infos[:5]):
                if isinstance(info, dict):
                    # logger.info("  info[%d]: knowledgeId=%s name=%r keys=%s", i,
                    #             info.get("knowledgeId") or info.get("knowledge_id"),
                    #             info.get("knowledgeName") or info.get("knowledge_name"),
                    #             sorted(info.keys())[:12])
                    pass

        # 收集所有 knowledgeId(去重)
        seen_kids: set = set()
        kid_order: List[str] = []
        for info in all_infos:
            kid = info.get("knowledgeId") or info.get("knowledge_id") or ""
            if kid and kid not in seen_kids:
                seen_kids.add(kid)
                kid_order.append(kid)
        # logger.info("info 召回汇总: keywords=%s 总条目=%d knowledgeIds=%s",
        #             keywords, len(all_infos), kid_order)
        if not all_infos:
            # logger.warning("所有关键词均无 info 召回——请检查索引 "
            #                "ngkm.knowledges_%s 是否存在、其中有无匹配知识",
            #                region_code)
            pass

        # ---- Step 3: 按 knowledgeId 检索 atom(去重复用) ----
        atoms_cache: Dict[str, List[dict]] = {}
        for kid in kid_order:
            try:
                atom_resp = self._get_atom(knowledgeId=kid, region_code=region_code,
                                           timeout=timeout)
                # logger.info("atom 原始响应 knowledgeId=%s: %s", kid, _preview(atom_resp))
                raw_obj = atom_resp.get("object", "") if isinstance(atom_resp, dict) else ""
                parsed = json.loads(raw_obj) if isinstance(raw_obj, str) else raw_obj or {}
                atoms = _extract_doc_list(parsed)
                if not atom_eg and isinstance(atom_resp, dict):
                    atom_eg = dict(atom_resp)
                if not atom_parsed_eg:
                    atom_parsed_eg = parsed
                if not atom_list_eg and atoms:
                    atom_list_eg = [dict(d) for d in atoms[:1] if isinstance(d, dict)]
                for a in atoms:
                    if isinstance(a, dict):
                        a["knowledgeId"] = kid
                # logger.info("atom 召回 knowledgeId=%s → %d 条", kid, len(atoms))
            except Exception as exc:  # noqa: BLE001
                # logger.warning("atom 召回失败 knowledgeId=%s "
                #                "索引=ngkm.knowledge_atom_%s err=%r",
                #                kid, region_code, exc)
                atoms = [{"knowledgeId": kid, "error": f"atom 调用失败: {exc}"}]
            atoms_cache[kid] = atoms

        # ---- Step 4: 合并 info + atom ----
        merged: List[dict] = []
        for info in all_infos:
            kid = info.get("knowledgeId") or info.get("knowledge_id") or ""
            entry = dict(info)
            entry["atoms"] = atoms_cache.get(kid, []) if kid else []
            entry.pop("_keyword", None)
            merged.append(entry)

        all_atoms: List[dict] = [a for atoms in atoms_cache.values() for a in atoms]
        # logger.info("keyword_search 完成: merged=%d 条 (info=%d, atom=%d) knowledgeIds=%s",
        #             len(merged), len(all_info_clean), len(all_atoms), kid_order)
        return {
            "keywords": keywords,
            "knowledge_ids": kid_order,
            "info": all_infos,
            "atom": all_atoms,
            "merged_count": len(merged),
            "merged": merged,
        }

    # ------------------------------------------------------------------
    # ngkm HTTP 调-----------
    def _get_info(self, keyword: str, region_code: str = "000",
                  timeout: int = 30) -> dict:
        """知识主索引关键词检索(ngkm.knowledges_{region_code})。"""
        rendered = JinjaTemplate(_INFO_RECALL_TEMPLATE).render(
            keyword=keyword, region_code=_region_code(region_code))
        # logger.info("ngkm info 请求体 keyword=%r: %s", keyword,
        #             rendered.replace("\n", " "))
        try:
            # 观测完整请求耗时期间不设置 HTTP 超时(requests 缺省 timeout=None 即无限等待)
            # resp = requests.post(
            #     _NGKM_SEARCH_URL,
            #     headers={"Content-Type": "application/json"},
            #     json=json.loads(rendered), timeout=timeout,
            # )
            resp = requests.post(
                _NGKM_SEARCH_URL,
                headers={"Content-Type": "application/json"},
                json=json.loads(rendered),
            )
        except Exception as exc:  # noqa: BLE001
            # logger.warning("ngkm info 请求失败(网络/超时/DNS) url=%s keyword=%r err=%r",
            #                _NGKM_SEARCH_URL, keyword, exc)
            raise
        # logger.info("ngkm info 响应 keyword=%r status=%s body=%s",
        #             keyword, resp.status_code, resp.text[:800])
        if resp.status_code != 200:
            # logger.warning("ngkm info 检索非200 keyword=%r 索引=ngkm.knowledges_%s "
            #                "status=%s", keyword, _region_code(region_code),
            #                resp.status_code)
            pass
        resp.raise_for_status()
        return resp.json()

    def _get_atom(self, knowledgeId: str, region_code: str = "000",
                  timeout: int = 30) -> dict:
        """原子表按 knowledgeId 检索(ngkm.knowledge_atom_{region_code})。"""
        rendered = JinjaTemplate(_ATOM_RECALL_TEMPLATE).render(
            knowledgeId=knowledgeId, region_code=_region_code(region_code))
        # logger.info("ngkm atom 请求体 knowledgeId=%s: %s", knowledgeId,
        #             rendered.replace("\n", " "))
        try:
            # 观测完整请求耗时期间不设置 HTTP 超时(requests 缺省 timeout=None 即无限等待)
            # resp = requests.post(
            #     _NGKM_SEARCH_URL,
            #     headers={"Content-Type": "application/json"},
            #     json=json.loads(rendered), timeout=timeout,
            # )
            resp = requests.post(
                _NGKM_SEARCH_URL,
                headers={"Content-Type": "application/json"},
                json=json.loads(rendered),
            )
        except Exception as exc:  # noqa: BLE001
            # logger.warning("ngkm atom 请求失败(网络/超时/DNS) url=%s knowledgeId=%s err=%r",
            #                _NGKM_SEARCH_URL, knowledgeId, exc)
            raise
        # logger.info("ngkm atom 响应 knowledgeId=%s status=%s body=%s",
        #             knowledgeId, resp.status_code, resp.text[:800])
        if resp.status_code != 200:
            # logger.warning("ngkm atom 检索非200 knowledgeId=%s "
            #                "索引=ngkm.knowledge_atom_%s status=%s",
            #                knowledgeId, _region_code(region_code),
            #                resp.status_code)
            pass
        resp.raise_for_status()
        return resp.json()
