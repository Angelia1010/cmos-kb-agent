# -*- coding: utf-8 -*-
"""KB 领域配置(判据阈值/延迟预算)。循环上限由 uniagent Budget(max_iterations) 承担。"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Config:
    # ---- 检索循环(方案 3.1) ----
    max_retrieval_rounds: int = 2        # 硬上限,第一版固定 2 轮
    recall_size: int = 10
    fuse_top_n: int = 8

    # ---- 充分性判据(方案 3.4) ----
    top3_score_threshold: float = 0.4    # top3 得分阈值(与召回打分同量纲)
    min_chunk_count: int = 3

    # ---- 延迟预算 ms(方案 六;超时即降级) ----
    budget: Dict[str, int] = field(default_factory=lambda: {
        "fast_path": 50,
        "retrieval_total": 2000,
        "data_processing": 300,
        "answer_first_token": 1200,
        "end_to_end": 4000,
    })

    # ---- 溯源 ----
    stale_days: int = 365                # 知识超过该天数标记"可能过旧"

    # ---- 缓存 ----
    cache_sim_threshold: float = 0.85


DEFAULT_CONFIG = Config()
