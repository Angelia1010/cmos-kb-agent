"""imports 模块测试 —— 动态导入解析器。"""

import unittest
from uniagent.imports.resolvers import resolve_variable, resolve_class, ResolveError


class TestResolveVariable(unittest.TestCase):
    """resolve_variable: 从 'pkg.mod:name' 字符串导入任意对象。"""

    def test_resolve_builtin(self):
        """导入标准库对象：json:dumps"""
        fn = resolve_variable("json:dumps")
        import json
        self.assertIs(fn, json.dumps)

    def test_resolve_with_type_check(self):
        """导入后做类型校验：os.path:sep 应为 str"""
        sep = resolve_variable("os.path:sep", str)
        self.assertIsInstance(sep, str)

    def test_resolve_type_mismatch(self):
        """类型不匹配时应抛 ResolveError"""
        with self.assertRaises(ResolveError):
            resolve_variable("os.path:sep", int)

    def test_missing_colon(self):
        """路径中缺少冒号时应抛 ResolveError"""
        with self.assertRaises(ResolveError):
            resolve_variable("json.dumps")

    def test_empty_attr_name(self):
        """冒号后为空时应抛 ResolveError"""
        with self.assertRaises(ResolveError):
            resolve_variable("json:")

    def test_nonexistent_module(self):
        """不存在的模块应抛 ResolveError"""
        with self.assertRaises(ResolveError):
            resolve_variable("nonexistent_module_xyz:foo")

    def test_nonexistent_attr(self):
        """模块存在但属性不存在应抛 ResolveError"""
        with self.assertRaises(ResolveError):
            resolve_variable("json:nonexistent_func_xyz")


class TestResolveClass(unittest.TestCase):
    """resolve_class: 从字符串导入类，可选基类校验。"""

    def test_resolve_class_basic(self):
        """导入一个类：collections:OrderedDict"""
        cls = resolve_class("collections:OrderedDict")
        from collections import OrderedDict
        self.assertIs(cls, OrderedDict)

    def test_resolve_class_with_base_check(self):
        """基类校验通过"""
        cls = resolve_class("collections:OrderedDict", base_class=dict)
        self.assertTrue(issubclass(cls, dict))

    def test_resolve_class_base_check_fail(self):
        """基类校验不通过应抛 ResolveError"""
        with self.assertRaises(ResolveError):
            resolve_class("collections:OrderedDict", base_class=list)

    def test_resolve_non_class_raises(self):
        """解析到的对象不是类时应抛 ResolveError"""
        with self.assertRaises(ResolveError):
            resolve_class("json:dumps")  # dumps 是函数，不是类


if __name__ == "__main__":
    unittest.main()
