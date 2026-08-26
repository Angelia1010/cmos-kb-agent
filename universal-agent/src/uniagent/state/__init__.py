"""线程状态、归约器与持久化后端(KB 适配版:移除 FeatureList/WIP/日志)。"""

from uniagent.state.thread_state import ThreadState
from uniagent.state.reducers import idempotent_merge, dedup_list_merge, last_wins
from uniagent.state.backend import StateBackend, LocalFileBackend, get_backend
from uniagent.state.redis_backend import RedisBackend

__all__ = [
    "ThreadState",
    "idempotent_merge",
    "dedup_list_merge",
    "last_wins",
    "StateBackend",
    "LocalFileBackend",
    "RedisBackend",
    "get_backend",
]
