# 11 - skills 模块：技能子系统

## 文件

- `uniagent/skills/manifest.py` — `SkillManifest`、`TriggerRule`、`ReferenceEntry` 数据模型
- `uniagent/skills/loader.py` — `SkillLoader`、`SkillContent` 渐进式披露加载
- `uniagent/skills/injector.py` — `SkillInjector` 系统提示注入
- `uniagent/skills/registry.py` — `SkillRegistry` 目录扫描、触发器匹配

## 功能说明

### SkillManifest（技能清单）

从 `metadata.json` 解析，不可变数据类：

| 字段 | 说明 |
|------|------|
| `name` / `skill_id` | 名称 / 由名称派生的标识符 |
| `triggers` | 激活触发条件列表（keyword/prefix/regex/intent） |
| `references` | 渐进式披露文档列表 |
| `templates` / `scripts` | 模板/脚本文件 |
| `promoted_tools` | 激活时应提升的工具列表 |

### TriggerRule（触发规则）

4种类型：
- **keyword**：精确或子串匹配，按覆盖率打分
- **prefix**：消息前缀匹配（如 `/skill-name`）
- **regex**：正则表达式匹配，预编译缓存
- **intent**：语义意图（需外部分类器，当前未实现）

### SkillLoader（渐进式披露加载）

4个披露阶段：
1. `instruction` — 始终加载 SKILL.md
2. `eager_references` — `when="always"` 的参考立即加载
3. `loaded_references` — `load_reference()` 按需加载
4. `templates` — `load_template()` 按需加载

### SkillInjector（系统提示注入）

将激活的技能内容格式化为 `<!-- SKILL: id -->...<!-- /SKILL: id -->` 段落追加到系统提示。支持最多 N 个同时激活的技能（超出时驱逐最旧的）。

### SkillRegistry（注册表）

完整的技能生命周期管理：
1. `scan(*dirs)` — 扫描目录，找到含 `metadata.json` 的子目录并注册
2. `match(user_input)` — 将输入与所有触发器匹配，返回按得分降序排列的结果
3. `match_by_name(name)` — 按名称直接查找
4. `activate(match)` — 加载技能内容
5. `register/unregister` — 手动管理
