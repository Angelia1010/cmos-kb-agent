"""state 模块测试 —— 归约器、ThreadState、LocalFileBackend。"""

import asyncio
import os
import tempfile
import unittest

from uniagent.state.reducers import last_wins, idempotent_merge, dedup_list_merge
from uniagent.state.backend import LocalFileBackend


class TestLastWins(unittest.TestCase):
    def test_returns_new(self):
        self.assertEqual(last_wins("old", "new"), "new")
        self.assertEqual(last_wins(1, 2), 2)
        self.assertIsNone(last_wins("x", None))


class TestIdempotentMerge(unittest.TestCase):
    def test_merge_disjoint(self):
        result = idempotent_merge({"a": 1}, {"b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_merge_overlap_new_wins(self):
        result = idempotent_merge({"a": 1, "b": 2}, {"b": 99, "c": 3})
        self.assertEqual(result, {"a": 1, "b": 99, "c": 3})

    def test_merge_empty(self):
        self.assertEqual(idempotent_merge({}, {"x": 1}), {"x": 1})
        self.assertEqual(idempotent_merge({"x": 1}, {}), {"x": 1})


class TestDedupListMerge(unittest.TestCase):
    def test_no_dups(self):
        result = dedup_list_merge([1, 2], [3, 4])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_dedup(self):
        result = dedup_list_merge([1, 2, 3], [2, 3, 4])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_preserves_order(self):
        result = dedup_list_merge(["a", "b"], ["b", "c", "a"])
        self.assertEqual(result, ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(dedup_list_merge([], [1, 2]), [1, 2])
        self.assertEqual(dedup_list_merge([1, 2], []), [1, 2])


class TestLocalFileBackend(unittest.TestCase):
    """LocalFileBackend: JSON文件持久化。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.backend = LocalFileBackend(state_dir=self.tmpdir)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_save_and_load(self):
        self._run(self.backend.save("test_key", {"x": 42}))
        data = self._run(self.backend.load("test_key"))
        self.assertEqual(data, {"x": 42})

    def test_load_missing(self):
        data = self._run(self.backend.load("nonexistent"))
        self.assertIsNone(data)

    def test_delete(self):
        self._run(self.backend.save("to_delete", {"a": 1}))
        self._run(self.backend.delete("to_delete"))
        data = self._run(self.backend.load("to_delete"))
        self.assertIsNone(data)

    def test_list_keys(self):
        self._run(self.backend.save("alpha", {}))
        self._run(self.backend.save("beta", {}))
        keys = self._run(self.backend.list_keys())
        self.assertIn("alpha", keys)
        self.assertIn("beta", keys)

    def test_list_keys_with_prefix(self):
        self._run(self.backend.save("foo_1", {}))
        self._run(self.backend.save("foo_2", {}))
        self._run(self.backend.save("bar_1", {}))
        keys = self._run(self.backend.list_keys("foo"))
        self.assertEqual(len(keys), 2)


if __name__ == "__main__":
    unittest.main()
