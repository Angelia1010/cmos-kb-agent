# test.html 关键词重写前后对比 — 精简版

## 目标
test.html 批量测试台新增 4 列：重写前关键词、重写后关键词、重写前检索id、重写后检索id。

## 修改文件（共 5 个）

1. **tools.py** — intergrate_all 覆盖前保存第一轮数据到 `ws.data["round1_keywords"]` / `ws.data["round1_ranked_kids"]`
2. **models.py**(service) — RetrievalResponseObject 和 BatchRetrievalItem 各加 `round1_keywords` + `round1_kids` 字段
3. **runner.py** — 从 ws.data 读取 round1 数据传入响应
4. **app.py** — batch stream 构造 BatchRetrievalItem 时传入 round1 字段；创建 GET `/test-recall/stream` SSE 端点
5. **test.html** — 全部明细表新增 4 列，渲染 round1_keywords / rewritten_keywords / round1_kids / recalled_kids
