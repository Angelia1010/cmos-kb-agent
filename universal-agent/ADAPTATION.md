# uniagent 裁剪与适配说明(修订版)

## 定位原则(本次修订确立)

**主智能体是编排好的,子智能体是自主规划的。** 这决定了取舍标准:

- 主智能体(kbagent/main_agent.py):阶段顺序、缓存快速通道、降级兜底、trace
  全部固定,不含任何 LLM 决策 —— 生产链路的骨架必须可预测;
- 三个子智能体(kbagent/subagents.py):每个阶段内部,LLM 面对自己的工具集
  自主决定调用顺序与参数(uniagent create_agent → ReAct);
- 自主性有边界:轮次/时间上限(Budget)、充分性判定(Verifier)、DSL 白名单、
  锚定校验、保底流水线都是确定性护栏,刻意不交给 LLM。

## 保留清单(自主规划子智能体所需 + 明确要求保留)

| 保留模块 | 在本场景中的角色 |
|---|---|
| agents/(create_agent 工厂 + AgentFeatures) | 构建自主规划的 ReAct 子智能体 |
| **skills/ 技能子系统(明确要求保留)** | 业务技能包(taocan/kuandai/zhangdan/tousu):关键词触发 → 归一规则注入处理子智能体;新增业务类目零代码 |
| middleware/(5 个内置) | 为自主工具调用服务:SkillMiddleware 注入技能、ToolErrorHandling 工具异常不崩溃、LoopDetection 防子智能体死循环、DanglingToolCall 修补孤立调用、TokenUsage 用量统计 |
| runtime/(GoalLoop/Budget/信号/钩子) | 检索循环骨架:验证失败注入负例反馈、轮次与时间硬上限 |
| verification/(Verifier 协议 + llm/composite/always_pass) | 充分性检验以验证器挂载,生成器/评估器分离 |
| tools/registry + state/(ThreadState/reducers/backend) + config/ + imports/ | 工厂与技能子系统的依赖底座 |

## 删除清单(与"自主规划"无关的能力)

| 删除模块 | 理由 |
|---|---|
| sandbox/ + SandboxMiddleware | 子智能体的工具是检索/清洗函数,不执行任意代码 |
| tools/deferred/(MCP 延迟发现)+ mcp_metadata | 工具集固定 11 个,延迟发现零收益、多一次往返 |
| tools/builtins/clarification | 坐席场景不向最终用户反问 |
| state/feature_list、wip、progress_log、decision_log + runtime/wip_hook | WIP=1/功能状态机是长周期编码任务机制,单查询秒级链路无任务列表可管 |
| verification/command_verifier | 无 shell 命令可作为验证依据 |
| middleware/summarization | 单查询消息数达不到摘要阈值;做多轮坐席会话时可从上游恢复 |

## 框架兼容性修复(非裁剪,原样无法运行)

1. middleware/base.py 原继承的 `AgentMiddleware` 在 langgraph 0.6/1.x 均不存在 → 自包含基类;
2. 中间件链原只挂在 agent 对象上从未执行 → 循环引擎显式执行(before 正序/after 逆序);
3. 技能注入原依赖不存在的提示词管道 → 改为 SystemMessage 追加进消息流。

## 与方案 V2 的映射、生产接入清单

见 README.md(GoalLoop=检索循环、Verifier=充分性判据、技能包=业务个性化处理、
Budget=轮次与延迟预算;接入需替换 LLM/ESClient、缓存换 Redis、评测集校准阈值)。
