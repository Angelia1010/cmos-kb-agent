# -*- coding: utf-8 -*-
"""共享词表:类目提示词 / 同义词扩展。"""

CATEGORY_HINTS = {
    "套餐": "套餐", "流量": "套餐", "资费": "套餐",
    "宽带": "宽带", "光纤": "宽带",
    "账单": "账单", "话费": "账单", "扣费": "账单",
    "投诉": "投诉",
}

SYNONYMS = {
    "流量": ["流量包", "上网流量", "达量限速"],
    "套餐": ["资费套餐", "套餐办理"],
    "宽带": ["家庭宽带", "光纤"],
    "账单": ["话费账单", "消费明细"],
    "投诉": ["客户投诉", "投诉处理"],
}


def detect_categories(text: str):
    return sorted({cat for token, cat in CATEGORY_HINTS.items() if token in text})


def extract_keywords(text: str):
    kws = []
    for token in CATEGORY_HINTS:
        if token in text and token not in kws:
            kws.append(token)
    return kws or [text[:8]]


def expand_terms(keywords):
    out = []
    for k in keywords:
        out.extend(SYNONYMS.get(k, []))
    return out
