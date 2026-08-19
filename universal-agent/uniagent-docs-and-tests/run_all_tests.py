"""一键运行 uniagent 全部模块测试。

用法:
    cd universal-agent
    PYTHONPATH=src python uniagent-docs-and-tests/run_all_tests.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path

# 确保 src 在 sys.path 中
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 发现并运行所有测试
def main():
    test_dir = Path(__file__).resolve().parent
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 按编号顺序查找所有 test_*.py
    test_files = sorted(test_dir.rglob("test_*.py"))

    for tf in test_files:
        # 将文件路径转换为模块式的 dotted name
        rel = tf.relative_to(test_dir)
        parts = list(rel.parts)
        parts[-1] = parts[-1].replace(".py", "")

        # 直接从文件加载
        try:
            spec = importlib.util.spec_from_file_location(
                ".".join(parts), str(tf)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            file_suite = loader.loadTestsFromModule(mod)
            suite.addTests(file_suite)
            print(f"  [OK] 已加载 {rel}")
        except Exception as exc:
            print(f"  [FAIL] 加载 {rel} 失败: {exc}")

    print(f"\n{'='*60}")
    print(f"共发现 {suite.countTestCases()} 个测试用例")
    print(f"{'='*60}\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    print(f"结果: {'全部通过' if result.wasSuccessful() else '存在失败'}")
    print(f"运行: {result.testsRun}  失败: {len(result.failures)}  错误: {len(result.errors)}")
    print(f"{'='*60}")

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
