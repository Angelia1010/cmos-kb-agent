"""runtime/context + checkpointer + protocols 测试。"""

import unittest
from dataclasses import dataclass

from uniagent.runtime.context import (
    CurrentUser,
    get_current_user,
    set_current_user,
)
from uniagent.runtime.checkpointer import create_checkpointer
from uniagent.runtime.protocols import LoopSignalBase, LoopHookBase


@dataclass
class FakeUser:
    user_id: str
    display_name: str


class TestCurrentUserContext(unittest.TestCase):
    def tearDown(self):
        set_current_user(None)

    def test_default_none(self):
        set_current_user(None)
        self.assertIsNone(get_current_user())

    def test_set_and_get(self):
        user = FakeUser(user_id="u123", display_name="Alice")
        set_current_user(user)
        got = get_current_user()
        self.assertIsNotNone(got)
        self.assertEqual(got.user_id, "u123")
        self.assertEqual(got.display_name, "Alice")

    def test_protocol_check(self):
        user = FakeUser(user_id="u1", display_name="Bob")
        self.assertIsInstance(user, CurrentUser)

    def test_clear(self):
        set_current_user(FakeUser("u1", "X"))
        set_current_user(None)
        self.assertIsNone(get_current_user())


class TestCheckpointer(unittest.TestCase):
    def test_memory_backend(self):
        cp = create_checkpointer("memory")
        self.assertIsNotNone(cp)

    def test_invalid_backend_resolves(self):
        """不存在的点分路径应抛异常。"""
        with self.assertRaises(Exception):
            create_checkpointer("nonexistent.module:Cls")


class TestProtocols(unittest.TestCase):
    def test_loop_signal_base_values(self):
        self.assertIsNotNone(LoopSignalBase.CONTINUE)
        self.assertIsNotNone(LoopSignalBase.BREAK)
        self.assertIsNotNone(LoopSignalBase.RETRY)
        self.assertIsNotNone(LoopSignalBase.ROLLBACK)

    def test_loop_hook_base_is_abstract(self):
        with self.assertRaises(TypeError):
            LoopHookBase()  # type: ignore  # 抽象类不能实例化


if __name__ == "__main__":
    unittest.main()
