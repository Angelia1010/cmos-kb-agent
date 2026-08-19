# uniagent 框架模块文档与测试

本目录为 `src/uniagent/` 框架的每个功能模块提供独立的**功能说明**和**测试脚本**。

## 目录结构

```
uniagent-docs-and-tests/
├── README.md                              ← 本文件
├── run_all_tests.py                       ← 一键运行全部测试
│
├── 01_imports/                            ← 动态导入解析器
│   ├── MODULE.md
│   └── test_imports.py
│
├── 02_config/                             ← 配置系统(YAML热重载/环境变量/ContextVar栈)
│   ├── MODULE.md
│   └── test_config.py
│
├── 03_state/                              ← 线程状态与归约器
│   ├── MODULE.md
│   └── test_state.py
│
├── 04_middleware/                          ← 中间件基类、洋葱链、排序装饰器
│   ├── MODULE.md
│   └── test_middleware.py
│
├── 05_middleware_builtins/                 ← 5个内置中间件
│   ├── MODULE.md
│   └── test_middleware_builtins.py
│
├── 06_runtime_signals_and_budget/          ← 循环信号、Budget预算管理
│   ├── MODULE.md
│   └── test_signals_and_budget.py
│
├── 07_runtime_hooks/                      ← 循环级钩子系统
│   ├── MODULE.md
│   └── test_hooks.py
│
├── 08_runtime_loop/                       ← GoalLoop/TurnLoop 循环引擎
│   ├── MODULE.md
│   └── test_loop.py
│
├── 09_runtime_context_and_checkpointer/   ← 用户上下文IoC + 检查点工厂
│   ├── MODULE.md
│   └── test_context_and_checkpointer.py
│
├── 10_verification/                       ← 验证器协议与内置验证器
│   ├── MODULE.md
│   └── test_verification.py
│
├── 11_skills/                             ← 技能子系统(清单/加载器/注入器/注册表)
│   ├── MODULE.md
│   └── test_skills.py
│
├── 12_tools/                              ← 工具注册表
│   ├── MODULE.md
│   └── test_tools.py
│
└── 13_agents/                             ← Agent工厂(create_agent/config_factory/features)
    ├── MODULE.md
    └── test_agents.py
```

## 运行方式

```bash
# 在项目根目录下运行全部测试
cd universal-agent
PYTHONPATH=src python uniagent-docs-and-tests/run_all_tests.py

# 运行单个模块测试
PYTHONPATH=src python -m pytest uniagent-docs-and-tests/01_imports/test_imports.py -v
# 或
PYTHONPATH=src python uniagent-docs-and-tests/01_imports/test_imports.py
```

## 依赖

与项目主体一致：`langgraph`, `langchain-core`, `pydantic`, `pyyaml`
