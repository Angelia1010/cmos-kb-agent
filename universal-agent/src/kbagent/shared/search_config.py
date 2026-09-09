# -*- coding: utf-8 -*-
"""检索配置常量 — 模板/URL/字段名/映射表/payload 集中存放。

本模块只承载静态配置常量,不含任何逻辑;
search.py 与其他模块经 ``from .search_config import ...`` 引用。
"""
from __future__ import annotations

from typing import Any, Dict


# ── 字段白名单 ──────────────────────────────────────────────────────────────
ALLOWED_FILTER_FIELDS = {"category", "status", "region"}
ALLOWED_BOOST_FIELDS = {"title", "content", "keywords"}


# ── ngkm 检索请求模板(Jinja2 占位符 {{ var }}) ─────────────────────────
_INFO_RECALL_TEMPLATE = """{
  "beans": [],
  "params": {
    "indexType": "knowledges_info",
    "indexName": "ngkm.knowledges_{{ region_code }}",
    "page": "1",
    "size": "100",
    "keyWord": "{{ keyword }}",
    "searchInfo": "knowledgeName=10,klgAliasName=5",
    "analysisType": "smart",
    "relCalculus": "OR",
    "highlightField": "knowledgeName,klgAliasName"
  }
}"""

_ATOM_RECALL_TEMPLATE = """{
  "beans": [
    {"column": "knowledgeId", "value": "{{ knowledgeId }}", "type": "any"}
  ],
  "params": {
        "indexType": "_doc",
        "mandatoryField": "knowledgeId,paramType,klgAttrAtomId,paramName,content,wkuntt,srcTemplateAttrAtomId,channelCode,groupId,except,annotation,isSendMessage,srcTmpltGrpngId",
        "indexName": "ngkm.knowledge_atom_{{ region_code }}",
        "ignoreField": "_id"
    }
}"""


# ── 省份名 → 区号 映射 ─────────────────────────────────────────────────────
_PROVINCE_TO_REGION = {
        "福建": "591", "甘肃": "931", "海南": "898", "河北": "311",
        "黑龙江": "451", "河南": "371", "宁夏": "951", "四川": "280",
        "云南": "871", "全国": "000",
    }


# ── 服务端点 URL ────────────────────────────────────────────────────────────
_SLOT_EXTRACT_URL = "http://restapi.ly4.tyyt.cmos:20070/slot_extract_unified"
_NGKM_SEARCH_URL = ("http://restapi.ngkmsearch.cs.glb.cmos:20070"
                    "/ngkmSearch/ws/int/busiSearcher/busiSearcherInterService")
_VECTOR_SEARCH_URL = ("http://192.168.212.199:8902/group/online-knowledge-embedding"
                      "/open/embedding/search/vector")


# ── 在线知识 embedding 向量检索请求体(新旧两套模板) ──────────────────────
# 公共字段;每次请求动态写入: content(用户问题) / reqId / xTransId(重新生成),
# sessionId 暂时保持原请求中的固定值。
_VECTOR_PAYLOAD_COMMON: Dict[str, Any] = {
    "esTop": 0,
    "isEnableGroupContent": True,
    "faqThresholdScore": 0.75,
    "rerankModelType": "ZY_QW",
    "isEnableGroupName": False,
    "faqContentThresholdScore": 0.7,
    "knowledgeThresholdScore": 0.35,
    "isOn": False,
    "isEnableResetRerank": False,
    "knowledgeNameSeparateThresholdScore": 0.9,
    "isEnableTotalKnowledge": "1",
    "isEnableFaq": True,
    "recessivityFlag": "0",
    "isEnableQa": True,
    "intentNM": "",
    "provinceId": "",
    "isEnableKnowledgeName": True,
    "reqId": "",                       # 每次请求重新生成
    "groupBindQuestionThresholdScore": 0.8,
    "sysChnlCode": "lingxi",
    "isEnableAtomName": False,
    "qaType": "out",
    "groupQuestionThresholdScore": 0.8,
    "knowledgeNameGroupThresholdScore": 0.3,
    "knowledgePath": "",
    "klgState": "2",
    "isEnableGroupBindQuestion": True,
    "isEnableFaqContent": True,
    "qaThresholdScore": 0.6,
    "enableKeywordReplace": "true",
    "faqoutScore": 0.85,
    "content": "",                     # 每次请求替换为用户问题
    "isEnableReRank": False,
    "isEnableQaContent": True,
    "channelCode": "1,lingxi,zaixian",
    "maxTop": 100,
    "replyModelScore": 0.9,
    "xTransId": "",                    # 每次请求重新生成
    "thresholdScore": 0.3,
    "newRouteExpThresholdScore": 0.3,
    "knowledgeGroupThresholdScore": 0.72,
    "sessionId": "53NaJ6mPpiiIxSMkb7k3t3eBgPMNRHd9",
    "knowledgeNameCoefficient": 0.35,
    "isEnableGroupQuestion": True,
    "knowledgeTop": 10,
}

# 新模板:启用新路由实验(newRouteExp),top=1 精排,embeddingTop=10
_VECTOR_PAYLOAD_NEW: Dict[str, Any] = {
    **_VECTOR_PAYLOAD_COMMON,
    "knowledgeNameGroupTop": 2,
    "embeddingTop": 10,
    "isEnableNewRouteExp": True,
    "newRouteExpTop": 100,
    "groupBindQuestionTop": 5,
    "top": 1,
}

# 旧模板:关闭新路由实验,top=100 宽召回,embeddingTop=30,含 faq/qaContent 独立条数
_VECTOR_PAYLOAD_OLD: Dict[str, Any] = {
    **_VECTOR_PAYLOAD_COMMON,
    "knowledgeNameGroupTop": 20,
    "embeddingTop": 30,
    "isEnableNewRouteExp": False,
    "newRouteExpTop": 0,
    "faqTop": 20,
    "groupBindQuestionTop": 20,
    "top": 100,
    "qaContentTop": 20,
}

_VECTOR_PAYLOADS: Dict[str, Dict[str, Any]] = {
    "new": _VECTOR_PAYLOAD_NEW,
    "old": _VECTOR_PAYLOAD_OLD,
}


# ── 向量检索响应字段名(解析返回条目时取值用) ────────────────────────────
_VECTOR_ID_FIELD = "knowledgeId"
_VECTOR_TITLE_FIELD = "knowledgeName"
_VECTOR_CONTENT_FIELD = "content"

