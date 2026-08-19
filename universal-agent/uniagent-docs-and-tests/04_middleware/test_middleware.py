"""middleware 模块测试 —— 基类、链组装、排序装饰器。"""

import asyncio
import unittest

from uniagent.middleware.base import Middleware
from uniagent.middleware.chain import assemble_middleware_chain, MiddlewareChainError
from uniagent.middleware.positioning import after, before, get_after_anchors, get_before_anchors


class TestMiddlewareBase(unittest.TestCase):
    """Middleware 基类默认行为。"""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_before_agent_default_none(self):
        mw = Middleware()
        result = self._run(mw.before_agent({"messages": []}))
        self.assertIsNone(result)

    def test_after_agent_default_none(self):
        mw = Middleware()
        result = self._run(mw.after_agent({"messages": []}))
        self.assertIsNone(result)

    def test_loop_hooks_default_empty(self):
        mw = Middleware()
        self.assertEqual(mw.loop_hooks(), [])

    def test_auto_name_from_class(self):
        class MyCustomMiddleware(Middleware):
            pass
        mw = MyCustomMiddleware()
        self.assertEqual(mw.name, "MyCustomMiddleware")


class TestPositioning(unittest.TestCase):
    """@after / @before 装饰器。"""

    def test_after_decorator(self):
        class AnchorMW(Middleware):
            pass

        @after(AnchorMW)
        class MyMW(Middleware):
            pass

        anchors = get_after_anchors(MyMW)
        self.assertIn(AnchorMW, anchors)

    def test_before_decorator(self):
        class AnchorMW(Middleware):
            pass

        @before(AnchorMW)
        class MyMW(Middleware):
            pass

        anchors = get_before_anchors(MyMW)
        self.assertIn(AnchorMW, anchors)

    def test_no_anchors_by_default(self):
        class PlainMW(Middleware):
            pass
        self.assertEqual(get_after_anchors(PlainMW), set())
        self.assertEqual(get_before_anchors(PlainMW), set())


class TestAssembleChain(unittest.TestCase):
    """assemble_middleware_chain: 链组装与约束检查。"""

    def test_basic_order_preserved(self):
        class A(Middleware): name = "A"
        class B(Middleware): name = "B"
        class C(Middleware): name = "C"
        chain = assemble_middleware_chain([A(), B(), C()])
        names = [m.name for m in chain]
        self.assertEqual(names, ["A", "B", "C"])

    def test_extras_appended(self):
        class A(Middleware): name = "A"
        class D(Middleware): name = "D"
        chain = assemble_middleware_chain([A()], extras=[D()])
        names = [m.name for m in chain]
        self.assertIn("D", names)

    def test_after_anchor_positioning(self):
        class A(Middleware): name = "A"
        class B(Middleware): name = "B"

        @after(A)
        class X(Middleware): name = "X"

        chain = assemble_middleware_chain([A(), B()], extras=[X()])
        names = [m.name for m in chain]
        self.assertGreater(names.index("X"), names.index("A"))

    def test_before_anchor_positioning(self):
        class A(Middleware): name = "A"
        class B(Middleware): name = "B"

        @before(B)
        class X(Middleware): name = "X"

        chain = assemble_middleware_chain([A(), B()], extras=[X()])
        names = [m.name for m in chain]
        self.assertLess(names.index("X"), names.index("B"))

    def test_conflicting_anchors_raises(self):
        class A(Middleware): name = "A"
        class B(Middleware): name = "B"

        @after(B)
        @before(A)
        class X(Middleware): name = "X"

        with self.assertRaises(MiddlewareChainError):
            assemble_middleware_chain([A(), B()], extras=[X()])

    def test_tail_type(self):
        class A(Middleware): name = "A"
        class B(Middleware): name = "B"
        class Tail(Middleware): name = "Tail"
        chain = assemble_middleware_chain([Tail(), A(), B()], tail_type=Tail)
        self.assertEqual(chain[-1].name, "Tail")


if __name__ == "__main__":
    unittest.main()
