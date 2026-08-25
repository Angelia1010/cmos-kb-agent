# -*- coding: utf-8 -*-
"""KB 领域配置(判据阈值/延迟预算)。"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    # ---- 检索循环 ----
    max_retrieval_rounds: int = 2
    recall_size: int = 10
    fuse_top_n: int = 8

    # ---- 充分性判据(规则层) ----
    top3_score_threshold: float = 0.4
    min_chunk_count: int = 3

    # ---- 延迟预算 ms ----
    budget: Dict[str, int] = field(default_factory=lambda: {
        "retrieval_total": 2000,
        "data_processing": 300,
        "answer_first_token": 1200,
        "end_to_end": 4000,
    })

    # ---- 溯源 ----
    stale_days: int = 365


DEFAULT_CONFIG = Config()
