(() => {
  "use strict";

  const JSON_PREVIEW = 240000;
  const VIEW_DEFINITIONS = [
    ["overview", "执行总览", "一次请求如何流到 Top3", "FLOW VIEW"],
    ["analyze", "Analyze", "候选规范化与质量统计", "STAGE 1"],
    ["filter", "Filter", "适用性判定与过滤原因", "STAGE 2"],
    ["build_markdown", "Build Markdown", "业务内容转换与对照", "STAGE 3"],
    ["rerank", "Rerank 回放", "Batch → Global → Top3", "STAGE 4"],
    ["final", "最终产物", "Top3、processed_chunks 与降级", "RESULT"],
    ["raw", "原始 Trace", "排查特殊字段时使用", "RAW JSON"],
  ];

  const REASON_LABELS = {
    inactive_status: "知识状态不可用",
    not_started: "尚未生效",
    expired: "已过有效期",
    region_not_applicable: "地区不适用",
    region_excluded: "地区被排除",
    channel_not_applicable: "渠道不适用",
    channel_excluded: "渠道被排除",
    empty_content: "没有可处理的业务内容",
    retrieval_rank_supplement: "模型结果不足，按程序内部 retrieval_rank 补位",
    incomplete_model_result: "模型返回结果不完整",
    batch_fallback_used: "上游批次曾发生降级或补位",
    global_model_failed: "全局模型调用未产生可用结果",
    rerank_wrong_count: "模型返回数量与期望不一致",
    rerank_invalid_json: "模型未返回严格 JSON",
    rerank_model_error: "模型调用异常",
    rerank_timeout: "模型调用超时",
    rerank_prompt_budget_exceeded: "Prompt 最小载荷仍超预算",
    insufficient_candidates: "有效候选不足 TopK",
  };

  const RERANK_MODE_HELP = {
    title_only: "Batch和Global都只发送标题，输入量最低，但可能损失正文语义",
    headings_and_intro: "Batch和Global发送H1-H3标题大纲，并附带简介/概述章节正文",
    title_then_content: "Batch只发送标题，Global发送标题和精简正文",
    title_and_content: "Batch和Global都发送正文，输入量和模型耗时可能更高",
  };

  const els = {
    search: document.getElementById("searchInput"),
    rerankMode: document.getElementById("rerankMode"),
    rerankModeHint: document.getElementById("rerankModeHint"),
    run: document.getElementById("runButton"),
    download: document.getElementById("downloadButton"),
    editor: document.getElementById("requestEditor"),
    requestError: document.getElementById("requestError"),
    format: document.getElementById("formatButton"),
    status: document.getElementById("status"),
    resultLayout: document.getElementById("resultLayout"),
    stageNav: document.getElementById("stageNav"),
    pipeline: document.getElementById("pipeline"),
    viewer: document.getElementById("traceViewer"),
    summary: document.getElementById("summary"),
    expand: document.getElementById("expandButton"),
    collapse: document.getElementById("collapseButton"),
    copyStage: document.getElementById("copyStageButton"),
    viewTitle: document.getElementById("viewTitle"),
    viewEyebrow: document.getElementById("viewEyebrow"),
    toast: document.getElementById("toast"),
  };

  const state = { response: null, view: "overview" };

  function make(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function append(parent, ...children) {
    children.flat().filter((item) => item !== null && item !== undefined).forEach((child) => {
      parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    });
    return parent;
  }

  function array(value) { return Array.isArray(value) ? value : []; }
  function object(value) { return value && typeof value === "object" && !Array.isArray(value) ? value : {}; }
  function stringify(value, space = 2) {
    try { return JSON.stringify(value, null, space); }
    catch (_) { return String(value); }
  }

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.classList.add("show");
    window.setTimeout(() => els.toast.classList.remove("show"), 1800);
  }

  function setStatus(kind, message) {
    els.status.className = `status ${kind || ""}`.trim();
    els.status.replaceChildren(make("span"), document.createTextNode(message));
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      showToast("已复制");
    } catch (_) {
      showToast("复制失败，请手动选择");
    }
  }

  function actionButton(label, handler, className = "small") {
    const button = make("button", className, label);
    button.type = "button";
    button.addEventListener("click", handler);
    return button;
  }

  function badge(text, type = "") { return make("span", `badge ${type}`.trim(), text); }
  function chip(text) { return make("span", "id-chip", text); }

  function panel(title, subtitle, content, extraClass = "") {
    const wrapper = make("section", `content-panel ${extraClass}`.trim());
    const heading = make("div", "panel-heading");
    const titleBox = make("div");
    titleBox.append(make("h3", "", title));
    if (subtitle) titleBox.append(make("p", "", subtitle));
    heading.append(titleBox);
    wrapper.append(heading);
    if (content) wrapper.append(content);
    return wrapper;
  }

  function statRow(items) {
    const row = make("div", "stat-row");
    items.forEach(([label, value]) => {
      const item = make("span", "stat-pill");
      item.append(document.createTextNode(`${label} `), make("b", "", value ?? "—"));
      row.append(item);
    });
    return row;
  }

  function dataTable(headers, rows, rowClass) {
    const wrap = make("div", "data-table-wrap");
    const table = make("table");
    const thead = make("thead");
    const headerRow = make("tr");
    headers.forEach((header) => headerRow.append(make("th", "", header)));
    thead.append(headerRow);
    const tbody = make("tbody");
    rows.forEach((row, index) => {
      const tr = make("tr", rowClass ? rowClass(row, index) : "");
      row.forEach((value) => {
        const td = make("td");
        td.append(value instanceof Node ? value : document.createTextNode(value == null ? "—" : String(value)));
        tr.append(td);
      });
      tbody.append(tr);
    });
    table.append(thead, tbody);
    wrap.append(table);
    return wrap;
  }

  function empty(message) { return make("div", "empty-state", message); }

  function jsonDrawer(label, value) {
    const details = make("details", "json-drawer");
    details.append(make("summary", "", label));
    const text = stringify(value);
    details.append(make("pre", "", text.length > JSON_PREVIEW
      ? `${text.slice(0, JSON_PREVIEW)}\n…[浏览器视图限长，下载 Trace 可查看更多]`
      : text));
    return details;
  }

  function lazyDetails(summaryContent, bodyFactory, open = false) {
    const details = make("details", "timeline-card");
    const summary = make("summary");
    summary.append(summaryContent);
    details.append(summary);
    let loaded = false;
    const load = () => {
      if (loaded) return;
      loaded = true;
      details.append(bodyFactory());
    };
    details._ensureLoaded = load;
    details.addEventListener("toggle", () => { if (details.open) load(); });
    if (open) { details.open = true; load(); }
    return details;
  }

  function codePanel(title, text) {
    const wrapper = make("div", "code-panel");
    const heading = make("div", "code-heading");
    heading.append(make("span", "", title));
    heading.append(actionButton("复制", (event) => {
      event.stopPropagation();
      copyText(text || "");
    }));
    wrapper.append(heading, make("pre", "", text || "（无）"));
    return wrapper;
  }

  function contentPreview(text, label = "查看内容") {
    if (!text) return make("span", "cell-sub", "未发送正文");
    const details = make("details");
    details.append(make("summary", "cell-sub", `${label}（${text.length} 字符）`));
    details.append(make("div", "content-preview", text));
    return details;
  }

  function currentTrace() { return state.response ? object(state.response.trace) : {}; }
  function currentStages() { return object(currentTrace().stages); }
  function stage(name) { return object(currentStages()[name]); }
  function stageOutput(name) { return object(stage(name).output); }

  function viewData(name) {
    if (name === "overview" || name === "raw") return currentTrace();
    if (name === "final") return object(currentTrace().final);
    return stage(name);
  }

  function queryText() {
    const request = object(currentTrace().request);
    return request.query || "";
  }

  function matches(value) {
    const keyword = els.search.value.trim().toLocaleLowerCase();
    return !keyword || stringify(value, 0).toLocaleLowerCase().includes(keyword);
  }

  function titleCell(title, subtitle) {
    const cell = make("div");
    cell.append(make("span", "cell-title", title || "（无标题）"));
    if (subtitle) cell.append(make("span", "cell-sub", subtitle));
    return cell;
  }

  function reasonText(reason) { return REASON_LABELS[reason] || reason; }

  function warningsView(warnings) {
    const values = array(warnings);
    if (!values.length) return make("div", "notice", "该阶段没有 warning。");
    const list = make("div", "warning-list");
    values.forEach((warning) => {
      const item = object(warning);
      const code = item.code || "warning";
      const text = item.message || reasonText(code);
      list.append(make("div", "warning-item", `${code}：${text}`));
    });
    return list;
  }

  function flowStory(steps) {
    const flow = make("div", "flow-story");
    steps.forEach((step, index) => {
      const box = make("div", "flow-step");
      box.append(make("span", "step-number", index + 1));
      box.append(make("strong", "", step[0]));
      box.append(make("p", "", step[1]));
      flow.append(box);
      if (index < steps.length - 1) flow.append(make("div", "flow-connector", "→"));
    });
    return flow;
  }

  function allWarnings() {
    const rows = [];
    ["analyze", "filter", "build_markdown", "rerank"].forEach((name) => {
      array(stageOutput(name).warnings).forEach((warning) => rows.push({ stage: name, ...object(warning) }));
    });
    return rows;
  }

  function pipelineMetrics() {
    const request = object(currentTrace().request);
    const analyze = stageOutput("analyze");
    const filter = stageOutput("filter");
    const build = stageOutput("build_markdown");
    const rerank = stageOutput("rerank");
    return [
      ["request", "原始候选", array(request.chunks).length, "chunks"],
      ["analyze", "Analyze", array(analyze.normalized_candidates).length, "已规范化"],
      ["filter", "Filter", array(filter.kept_candidates).length, `过滤 ${array(filter.filtered_decisions).length}`],
      ["build_markdown", "Markdown", array(build.processed_candidates).length, "已构建"],
      ["rerank", "Rerank", array(rerank.top3_candidates).length, "Top 候选"],
    ];
  }

  function renderSummary() {
    const response = state.response;
    const result = object(response.final_result);
    const request = object(response.trace && response.trace.request);
    const calls = array(stage("rerank").model_calls);
    const rerankMode = object(object(stageOutput("rerank")).rerank_details).input_mode
      || object(stage("rerank").details).input_mode
      || "—";
    const values = [
      ["输入候选", array(request.chunks).length, false],
      ["最终 Top", array(result.top3_candidates).length, false],
      ["模型调用", calls.length, false],
      ["重排模式", rerankMode, true],
      ["总耗时", `${response.trace.elapsed_ms || result.elapsed_ms || 0} ms`, false],
      ["执行结果", `${result.outcome || "—"}${result.degraded ? "（降级）" : ""}`, true],
    ];
    els.summary.replaceChildren();
    values.forEach(([label, value, small]) => {
      const card = make("article", "metric-card");
      card.append(make("span", "metric-label", label));
      card.append(make("strong", `metric-value${small ? " small-value" : ""}`, value));
      els.summary.append(card);
    });
  }

  function selectView(name) {
    state.view = name;
    renderNav();
    renderPipeline();
    renderCurrent();
  }

  function renderPipeline() {
    els.pipeline.replaceChildren();
    pipelineMetrics().forEach((item, index, values) => {
      const [view, label, count, suffix] = item;
      const button = make("button", `pipeline-node${state.view === view ? " active" : ""}`);
      button.type = "button";
      button.append(make("strong", "", label), make("span", "", count), make("small", "", suffix));
      button.addEventListener("click", () => selectView(view));
      els.pipeline.append(button);
      if (index < values.length - 1) els.pipeline.append(make("div", "pipeline-arrow", "→"));
    });
  }

  function renderNav() {
    els.stageNav.replaceChildren();
    VIEW_DEFINITIONS.forEach(([key, title, subtitle]) => {
      const button = make("button", state.view === key ? "active" : "");
      button.type = "button";
      button.append(make("strong", "", title), make("small", "", subtitle));
      button.addEventListener("click", () => selectView(key));
      els.stageNav.append(button);
    });
  }

  function renderOverview() {
    const container = make("div");
    const metrics = pipelineMetrics();
    container.append(flowStory([
      ["接收候选", `收到 ${metrics[0][2]} 条 Retrieval chunks，保留原始字段。`],
      ["确定性处理", `Analyze、Filter 和 Markdown 顺序执行，剩余 ${metrics[3][2]} 条可重排知识。`],
      ["两阶段重排", "先按批次粗排，再将入围池交给 Global 终排，模型故障时按程序规则补位。"],
      ["适配输出", `生成 ${array(object(currentTrace().final).processed_chunks).length} 条 processed_chunks，完整正文不受 Prompt 投影影响。`],
    ]));

    const stageRows = ["analyze", "filter", "build_markdown", "rerank"].map((name, index) => {
      const current = stage(name);
      const out = object(current.output);
      const counts = [
        array(out.normalized_candidates).length,
        array(out.kept_candidates).length,
        array(out.processed_candidates).length,
        array(out.top3_candidates).length,
      ];
      return [index + 1, VIEW_DEFINITIONS.find((item) => item[0] === name)[1], `${current.elapsed_ms || 0} ms`, counts[index], array(out.warnings).length];
    });
    container.append(panel("执行节点", "这里看的是顺序和变化，不是字段堆叠。", dataTable(
      ["顺序", "阶段", "耗时", "本阶段产物", "Warning"], stageRows,
    )));

    const finalMeta = object(object(currentTrace().final).processing_meta);
    const outcome = make("div");
    outcome.append(statRow([
      ["是否降级", finalMeta.degraded ? "是" : "否"],
      ["降级原因", array(finalMeta.degradation_reasons).length],
      ["全部 warning", allWarnings().length],
      ["Trace ID", currentTrace().trace_id || "—"],
    ]));
    if (array(finalMeta.degradation_reasons).length) {
      const reasons = make("div", "warning-list");
      array(finalMeta.degradation_reasons).forEach((reason) => reasons.append(make("div", "warning-item", reasonText(reason))));
      outcome.append(reasons);
    }
    container.append(panel("本次结果判定", "降级不等于失败；它表示部分结果由程序规则补位或候选不足。", outcome));
    return container;
  }

  function renderAnalyze() {
    const current = stage("analyze");
    const input = object(current.input);
    const output = object(current.output);
    const normalized = array(output.normalized_candidates).filter(matches);
    const analysis = object(output.analysis);
    const container = make("div");
    container.append(flowStory([
      ["读取 Retrieval chunks", `${array(input.chunks).length} 条原始 chunk 进入 Adapter。`],
      ["映射为知识候选", "建立 knowledge_id、title、atoms、matched_atom_ids、retrieval_rank 等内部字段。"],
      ["统计与质量检查", `得到 ${normalized.length} 条规范化候选，耗时 ${current.elapsed_ms || 0} ms。`],
    ]));
    const analysisStats = Object.entries(analysis)
      .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
      .slice(0, 12)
      .map(([key, value]) => [key, value]);
    if (analysisStats.length) container.append(panel("分析统计", "Analyze 生成的确定性质量指标。", statRow(analysisStats), "soft"));
    const rows = normalized.map((candidate) => {
      const item = object(candidate);
      return [
        item.source_index,
        item.knowledge_id,
        titleCell(item.name, `retrieval_rank=${item.retrieval_rank ?? "—"}`),
        array(item.atoms).length,
        object(item.applicability).status || item.status || "—",
      ];
    });
    container.append(panel("规范化后的候选", "搜索框可按 knowledge_id/chunk_id 筛选。", rows.length
      ? dataTable(["来源序号", "Knowledge ID", "标题 / 内部检索顺序", "Atoms", "状态"], rows)
      : empty("没有匹配当前搜索的候选。")));
    container.append(panel("Analyze warnings", "只展示安全的 code 和定位信息。", warningsView(output.warnings), "soft"));
    container.append(jsonDrawer("查看 Analyze 原始 Trace", current));
    return container;
  }

  function renderFilter() {
    const current = stage("filter");
    const input = object(current.input);
    const output = object(current.output);
    const kept = array(output.kept_decisions);
    const removed = array(output.filtered_decisions);
    const candidates = new Map(array(input.normalized_candidates).map((item) => [object(item).knowledge_id, object(item)]));
    const decisions = [...kept, ...removed].filter(matches);
    const container = make("div");
    container.append(flowStory([
      ["读取上下文", "地区、渠道、请求时间和受众从 processing_context 取值。"],
      ["执行适用性规则", "依次检查状态、生效时间、地区、渠道、排除条件和可渲染内容。"],
      ["分流", `保留 ${kept.length} 条，过滤 ${removed.length} 条，每条都保留决策原因。`],
    ]));
    container.append(panel("本次过滤上下文", "这些值决定适用性规则的分支。", statRow(Object.entries(object(input.processing_context)).map(([key, value]) => [key, value ?? "空"]))));
    const rows = decisions.map((decision) => {
      const item = object(decision);
      const candidate = candidates.get(item.knowledge_id) || {};
      const reasonBox = make("div");
      if (!array(item.reasons).length) reasonBox.append(badge("通过全部规则", "success"));
      else array(item.reasons).forEach((reason) => reasonBox.append(badge(reasonText(reason), "error"), document.createTextNode(" ")));
      return [
        item.accepted ? badge("保留", "success") : badge("过滤", "error"),
        item.knowledge_id,
        titleCell(candidate.name, `source_index=${item.source_index ?? "—"}`),
        reasonBox,
        `${item.kept_atom_count ?? 0} / ${item.filtered_atom_count ?? 0}`,
      ];
    });
    container.append(panel("逐条决策", "保留和过滤放在同一条时间线上，直接看到数据为什么流向下一步。", rows.length
      ? dataTable(["结果", "Knowledge ID", "知识", "原因", "保留/过滤 Atom"], rows, (row) => row[0].textContent === "保留" ? "selected-row" : "")
      : empty("没有匹配当前搜索的决策。")));
    container.append(panel("Filter warnings", "包含 Atom 级别和知识级别的适用性告警。", warningsView(output.warnings), "soft"));
    container.append(jsonDrawer("查看 Filter 原始 Trace", current));
    return container;
  }

  function renderBuildMarkdown() {
    const current = stage("build_markdown");
    const output = object(current.output);
    const comparisons = array(output.content_comparison).filter(matches);
    const container = make("div");
    container.append(flowStory([
      ["接收过滤后知识", `${array(object(current.input).filtered_candidates).length} 条知识进入内容构建。`],
      ["解析内容与 Atom", "处理 HTML、结构化内容、表格、Atom 适用性和注释可见性。"],
      ["生成完整 Markdown", `产生 ${array(output.processed_candidates).length} 条 content_md，作为重排的完整源内容。`],
    ]));
    const timeline = make("div", "timeline");
    comparisons.forEach((comparison, index) => {
      const item = object(comparison);
      const header = make("div", "timeline-title");
      header.append(make("strong", "", item.knowledge_name || item.knowledge_id));
      header.append(make("small", "", `${item.knowledge_id || "—"} · ${item.chunk_id || "—"}`));
      const summary = make("div");
      summary.append(header);
      const detail = lazyDetails(summary, () => {
        const body = make("div", "timeline-body");
        body.append(flowStory([
          ["原始内容", `${typeof item.source_content === "string" ? item.source_content.length : stringify(item.source_content, 0).length} 字符，${array(item.source_atoms).length} 个 Atom。`],
          ["确定性转换", "执行清洗、表格保护、标题建立和 Atom 排序。"],
          ["content_md", `${String(item.content_md || "").length} 字符，保留为最终 processed_chunk 正文。`],
        ]));
        const compare = make("div", "prompt-grid");
        compare.append(codePanel("原始 content / atoms", `${stringify(item.source_content)}\n\nAtoms:\n${stringify(item.source_atoms)}`));
        compare.append(codePanel("构建后 content_md", item.content_md || ""));
        body.append(compare);
        return body;
      }, index === 0);
      const itemWrap = make("div", "timeline-item");
      itemWrap.append(make("span", "timeline-dot"), detail);
      timeline.append(itemWrap);
    });
    container.append(panel("内容构建回放", "逐条对照原始数据和完整 Markdown，默认只展开第一条。", comparisons.length ? timeline : empty("没有匹配当前搜索的内容。")));
    container.append(panel("Build Markdown warnings", "渲染、注释和内容完整性告警。", warningsView(output.warnings), "soft"));
    container.append(jsonDrawer("查看 Build Markdown 原始 Trace", current));
    return container;
  }

  function rerankContext() {
    const current = stage("rerank");
    const output = object(current.output);
    const details = object(output.rerank_details);
    const evidenceMap = object(output.evidence_map);
    const processed = array(object(current.input).processed_candidates);
    const byKnowledge = new Map(processed.map((item) => [object(item).knowledge_id, object(item)]));
    const calls = array(current.model_calls);
    return { current, output, details, evidenceMap, processed, byKnowledge, calls };
  }

  function callForBatch(ctx, batch, ordinal) {
    return ctx.calls.find((call) => object(call).stage === "batch" && object(object(call).parsed_result).batch_index === batch.batch_index)
      || ctx.calls.filter((call) => object(call).stage === "batch")[ordinal]
      || null;
  }

  function callForGlobal(ctx) { return ctx.calls.find((call) => object(call).stage === "global") || null; }

  function evidenceInfo(ctx, evidenceId) {
    const knowledgeId = ctx.evidenceMap[evidenceId] || "";
    const candidate = ctx.byKnowledge.get(knowledgeId) || {};
    return { evidenceId, knowledgeId, candidate };
  }

  function rerankEvidenceMatches(ctx, evidenceId, sentPayload = {}) {
    const info = evidenceInfo(ctx, evidenceId);
    return matches({
      evidence_id: evidenceId,
      knowledge_id: info.knowledgeId,
      chunk_id: object(info.candidate).chunk_id,
      title: object(info.candidate).name,
      sent_payload: sentPayload,
    });
  }

  function payloadById(call) {
    return new Map(array(object(object(call).input_payload).candidates).map((candidate) => [object(candidate).evidence_id, object(candidate)]));
  }

  function promptDetailsById(detail) {
    return new Map(array(object(object(detail).prompt).candidates).map((candidate) => [object(candidate).evidence_id, object(candidate)]));
  }

  function rankList(title, ids, ctx, type = "info") {
    const wrapper = make("div", "content-panel soft");
    wrapper.append(make("h4", "", title));
    const list = make("div", "chip-list");
    array(ids).forEach((id, index) => {
      const info = evidenceInfo(ctx, id);
      const name = info.candidate.name || info.knowledgeId || id;
      list.append(badge(`${index + 1}. ${id} · ${name}`, type));
    });
    if (!array(ids).length) list.append(make("span", "cell-sub", "无"));
    wrapper.append(list);
    return wrapper;
  }

  function promptView(call) {
    if (!call) return make("div", "notice warn", "本阶段没有调用模型，通常是 Prompt 最小载荷超预算或候选数不需要重排。");
    const prompt = object(object(call).prompt);
    const grid = make("div", "prompt-grid");
    grid.append(codePanel(`System Prompt · ${prompt.system_chars || 0} 字符`, prompt.system || ""));
    grid.append(codePanel(`User Prompt · ${prompt.user_chars || 0} 字符`, prompt.user || ""));
    return grid;
  }

  function projectionDescription(prompt, isBatch) {
    const mode = object(prompt).input_mode;
    if (mode === "headings_and_intro") {
      return "从完整content_md按原顺序提取H1-H3；只为6类简介/概述标题附带完整正文单元，有效期仅保留标题。";
    }
    if (mode === "title_only" || (mode === "title_then_content" && isBatch)) {
      return "只投影Evidence ID和标题，不发送content_md。";
    }
    if (mode === "title_then_content") {
      return "按Atom精确命中、Query关键词相关度和原文顺序构建确定性精简正文。";
    }
    return "在字符预算内投影标题和正文，并只按完整Markdown单元缩减。";
  }

  function resultView(ctx, detail, call, nextStepText) {
    const wrapper = make("div");
    const modelIds = array(detail.model_ids);
    const selectedIds = array(detail.selected_ids);
    const supplements = selectedIds.filter((id) => !modelIds.includes(id));
    const process = make("div", "process-grid");
    [
      ["1 · LLM 原始输出", call ? "保留模型原始字符串，不改写。" : "未调用模型。"],
      ["2 · JSON 解析", "校验 ranked_ids 结构、Evidence ID、重复项和期望数量。"],
      ["3 · 确定性补位", supplements.length ? `${supplements.length} 条按程序内部 retrieval_rank 补入。` : "无补位，模型结果可直接使用。"],
      ["4 · 流向下一步", nextStepText],
    ].forEach(([label, text]) => {
      const box = make("div", "process-box");
      box.append(make("span", "", label), make("p", "", text));
      process.append(box);
    });
    wrapper.append(process);
    if (call) wrapper.append(codePanel(`LLM 原始输出 · ${object(call).elapsed_ms || 0} ms`, object(call).raw_output || ""));
    wrapper.append(rankList("模型解析后的有效排序", modelIds, ctx, "purple"));
    wrapper.append(rankList("程序补位后的实际结果", selectedIds, ctx, "success"));
    if (array(detail.fallback_reasons).length) {
      const reasons = make("div", "warning-list");
      array(detail.fallback_reasons).forEach((reason) => reasons.append(make("div", "warning-item", reasonText(reason))));
      wrapper.append(reasons);
    }
    return wrapper;
  }

  function batchBody(ctx, batch, call) {
    const body = make("div", "timeline-body");
    const payloadMap = payloadById(call);
    const detailMap = promptDetailsById(batch);
    body.append(flowStory([
      ["分批", `${array(batch.input_ids).length} 条 Evidence 被分到 Batch ${batch.batch_index}。`],
      ["Prompt 投影", projectionDescription(batch.prompt, true)],
      ["LLM 粗排", call ? `调用耗时 ${object(call).elapsed_ms || 0} ms，要求 Top ${object(object(call).input_payload).top_k || array(batch.selected_ids).length}。` : "Prompt 未进入模型，直接执行程序降级。"],
      ["入围全局池", `${array(batch.selected_ids).length} 条批次结果按批次顺序汇入全局候选池。`],
    ]));
    const rows = array(batch.input_ids).filter((id) => rerankEvidenceMatches(ctx, id, payloadMap.get(id))).map((id) => {
      const info = evidenceInfo(ctx, id);
      const sent = payloadMap.get(id) || {};
      const projection = detailMap.get(id) || {};
      const sentFields = [sent.title !== undefined ? "title" : null, sent.content_md !== undefined ? "content_md" : null].filter(Boolean).join(" + ") || "无";
      const treatment = make("div");
      treatment.append(make("span", "cell-title", projection.selection_method || "—"));
      treatment.append(make("span", "cell-sub", `正文 ${projection.source_content_chars || 0} → ${projection.sent_content_chars || 0} 字符；省略 ${projection.omitted_unit_count || 0} 块`));
      if (sent.content_md) treatment.append(contentPreview(sent.content_md, "查看发送正文"));
      return [
        chip(id),
        info.knowledgeId,
        titleCell(sent.title || info.candidate.name, `retrieval_rank=${info.candidate.retrieval_rank ?? "—"}（仅程序内部）`),
        sentFields,
        treatment,
      ];
    });
    body.append(panel("本批候选如何进入 Prompt", "retrieval_rank 仅用于程序降级/补位，不会出现在模型 Prompt 中。", dataTable(
      ["Evidence", "Knowledge ID", "知识标题", "实际发送字段", "正文处理"], rows,
      (row) => array(batch.selected_ids).includes(row[0].textContent) ? "selected-row" : "",
    )));
    body.append(panel("实际模型 Prompt", `序列化后 ${object(batch.prompt).sent_chars || 0} / ${object(batch.prompt).budget_chars || 0} 字符。`, promptView(call)));
    body.append(panel("模型输出与程序后处理", "从原始输出到最终入围 ID，每一步都可对照。", resultView(ctx, batch, call, "将实际 selected_ids 追加到 Global 候选池。")));
    return body;
  }

  function globalBody(ctx, globalDetail, call) {
    const body = make("div", "timeline-body");
    const prompt = object(globalDetail.prompt);
    const payloadMap = payloadById(call);
    const detailMap = promptDetailsById(globalDetail);
    body.append(flowStory([
      ["汇总入围池", `${array(globalDetail.pool_ids).length} 条批次入围 Evidence 进入 Global。`],
      ["分配正文预算", `扣除固定开销 ${prompt.fixed_chars || 0} 和安全余量 ${prompt.safety_margin_chars || 0}，可用正文预算 ${prompt.body_budget_chars || 0}。`],
      ["构建候选投影", projectionDescription(prompt, false)],
      ["Global LLM 终排", call ? `调用耗时 ${object(call).elapsed_ms || 0} ms，最终要求 Top ${object(object(call).input_payload).top_k || 3}。` : "未调用模型，按程序内部顺序降级。"],
    ]));
    body.append(panel("全局 Prompt 预算是怎么分的", "先公平初配，再轮询复用剩余预算，不让少数长文占满。", statRow([
      ["总上限", prompt.budget_chars || 0],
      ["固定开销", prompt.fixed_chars || 0],
      ["安全余量", prompt.safety_margin_chars || 0],
      ["正文总预算", prompt.body_budget_chars || 0],
      ["初始单条配额", prompt.initial_per_candidate_body_budget_chars || 0],
      ["实际正文占用", prompt.sent_body_serialized_chars || 0],
      ["未使用", prompt.unused_body_budget_chars || 0],
      ["最终 Prompt", prompt.sent_chars || 0],
    ])));
    const rows = array(globalDetail.pool_ids).filter((id) => rerankEvidenceMatches(ctx, id, payloadMap.get(id))).map((id) => {
      const info = evidenceInfo(ctx, id);
      const sent = payloadMap.get(id) || {};
      const projection = detailMap.get(id) || {};
      const bodyCell = make("div");
      bodyCell.append(make("span", "cell-title", `${projection.sent_content_chars || 0} / ${projection.source_content_chars || 0} 字符`));
      bodyCell.append(make("span", "cell-sub", `${projection.sent_unit_count || 0} 块已发送，${projection.omitted_unit_count || 0} 块省略`));
      if (sent.content_md) bodyCell.append(contentPreview(sent.content_md, "查看精简正文"));
      const method = make("div");
      method.append(make("span", "cell-title", projection.selection_method || "—"));
      if (array(projection.matched_atom_ids_used).length) method.append(make("span", "cell-sub", `Atom: ${array(projection.matched_atom_ids_used).join(", ")}`));
      return [chip(id), info.knowledgeId, sent.title || info.candidate.name || "—", method, bodyCell];
    });
    body.append(panel("全局候选的Prompt投影", "这些是真正送入 Global Prompt 的标题和content_md投影。", rows.length
      ? dataTable(["Evidence", "Knowledge ID", "标题", "选择方式", "实际发送content_md"], rows)
      : empty("全局候选池为空。")));
    body.append(panel("实际 Global Prompt", `序列化后 ${prompt.sent_chars || 0} / ${prompt.budget_chars || 0} 字符。`, promptView(call)));
    body.append(panel("Global 模型输出与 Top3 生成", "模型结果不足或异常时，保留有效结果后按程序规则补齐。", resultView(ctx, globalDetail, call, "selected_ids 转换为最终 Top3，并适配为 processed_chunks。")));
    return body;
  }

  function renderRerank() {
    const ctx = rerankContext();
    const batches = array(ctx.details.batches);
    const globalDetail = object(ctx.details.global);
    const pool = array(globalDetail.pool_ids);
    const container = make("div");
    const lane = make("div", "rerank-lane");
    [
      ["可重排候选", ctx.details.eligible_count || ctx.processed.length, `输入模式 ${ctx.details.input_mode || "—"}`],
      ["Batch 粗排", batches.length, "默认每批 20，每批选 Top5"],
      ["Global 候选池", pool.length, "最多 25 条"],
      ["最终 Top", array(globalDetail.selected_ids).length || array(ctx.output.top3_candidates).length, globalDetail.mode || "—"],
    ].forEach(([label, value, note], index, values) => {
      const item = make("div", "lane-node");
      item.append(make("small", "", label), make("b", "", value), make("small", "", note));
      lane.append(item);
      if (index < values.length - 1) lane.append(make("div", "lane-arrow", "→"));
    });
    container.append(lane);
    container.append(make("div", "notice", "下面按真实执行顺序回放。每个 Batch 可查看分批名单、Prompt 投影、实际 Prompt、LLM 原始输出、解析与补位；Global 另外展示正文预算分配。"));

    const timeline = make("div", "timeline");
    let visibleBatchCount = 0;
    batches.forEach((batchValue, index) => {
      const batch = object(batchValue);
      const call = callForBatch(ctx, batch, index);
      const payloadMap = payloadById(call);
      const batchMatches = matches({
        batch_index: batch.batch_index,
        input_ids: batch.input_ids,
        selected_ids: batch.selected_ids,
        candidates: array(batch.input_ids).map((id) => ({ id, ...evidenceInfo(ctx, id), sent: payloadMap.get(id) })),
      });
      if (!batchMatches) return;
      visibleBatchCount += 1;
      const summary = make("div");
      const title = make("div", "timeline-title");
      title.append(make("strong", "", `Batch ${batch.batch_index}`));
      title.append(make("small", "", `${array(batch.input_ids).length} 条输入 → ${array(batch.selected_ids).length} 条入围`));
      const meta = make("div", "timeline-meta");
      meta.append(badge(batch.complete ? "模型结果完整" : "含补位/降级", batch.complete ? "success" : "warn"));
      meta.append(badge(call ? `${object(call).elapsed_ms || 0} ms` : "未调模型", "info"));
      meta.append(badge(`${object(batch.prompt).sent_chars || 0}/${object(batch.prompt).budget_chars || 0} chars`, "purple"));
      summary.append(title, meta);
      const detailNode = lazyDetails(summary, () => batchBody(ctx, batch, call), index === 0);
      const itemWrap = make("div", "timeline-item");
      itemWrap.append(make("span", "timeline-dot"), detailNode);
      timeline.append(itemWrap);
    });

    if (Object.keys(globalDetail).length) {
      const call = callForGlobal(ctx);
      const summary = make("div");
      const title = make("div", "timeline-title");
      title.append(make("strong", "", "Global 终排"));
      title.append(make("small", "", `${pool.length} 条全局候选 → ${array(globalDetail.selected_ids).length} 条最终结果`));
      const meta = make("div", "timeline-meta");
      meta.append(badge(globalDetail.complete ? "模型结果完整" : "含补位/降级", globalDetail.complete ? "success" : "warn"));
      meta.append(badge(call ? `${object(call).elapsed_ms || 0} ms` : "未调模型", "info"));
      meta.append(badge(globalDetail.mode || "—", "purple"));
      summary.append(title, meta);
      const detailNode = lazyDetails(summary, () => globalBody(ctx, globalDetail, call), true);
      const itemWrap = make("div", "timeline-item");
      itemWrap.append(make("span", "timeline-dot"), detailNode);
      timeline.append(itemWrap);
    }
    if (!visibleBatchCount && !Object.keys(globalDetail).length) {
      timeline.append(empty("没有匹配当前搜索条件的重排记录。"));
    }
    container.append(panel("重排执行时间线", "Batch 顺序展开，最后进入 Global。默认展开第一批和 Global。", timeline));
    container.append(panel("Rerank warnings", "异常、非法输出、数量不足和 Prompt 超预算都会在这里汇总。", warningsView(ctx.output.warnings), "soft"));
    container.append(jsonDrawer("查看 Rerank 原始 Trace", ctx.current));
    return container;
  }

  function renderFinal() {
    const final = object(currentTrace().final);
    const candidates = array(final.top3_candidates).filter(matches);
    const chunks = new Map(array(final.processed_chunks).map((chunk) => [object(chunk).chunk_id, object(chunk)]));
    const meta = object(final.processing_meta);
    const container = make("div");
    container.append(flowStory([
      ["确定 Top3", `按 rerank_rank 得到 ${candidates.length} 条最终知识。`],
      ["适配 Chunk", "通过 chunk_id 找回原始 Retrieval chunk，回写完整 content_md。"],
      ["返回服务契约", `产生 ${array(final.processed_chunks).length} 条 processed_chunks，不携带调试 Trace 字段。`],
    ]));
    const rows = candidates.map((candidateValue) => {
      const candidate = object(candidateValue);
      const chunk = chunks.get(candidate.chunk_id) || {};
      return [
        badge(`#${candidate.rerank_rank || "—"}`, "success"),
        candidate.knowledge_id,
        titleCell(candidate.name || candidate.knowledge_name, candidate.chunk_id),
        candidate.retrieval_rank,
        String(candidate.content_md || "").length,
        contentPreview(chunk.content || candidate.content_md || "", "查看最终完整内容"),
      ];
    });
    container.append(panel("最终 Top3 与 processed_chunks", "Prompt 中的精简正文不会改变这里的完整输出。", rows.length
      ? dataTable(["排名", "Knowledge ID", "知识 / Chunk", "内部 retrieval_rank", "完整正文字符", "输出内容"], rows)
      : empty("没有匹配当前搜索的最终结果。")));
    const metaBox = make("div");
    metaBox.append(statRow([
      ["input", meta.input_count || 0], ["normalized", meta.normalized_count || 0],
      ["filtered", meta.filtered_count || 0], ["processed", meta.processed_count || 0],
      ["eligible", meta.rerank_eligible_count || 0], ["top", meta.top_count || 0],
      ["warnings", meta.warning_count || 0], ["degraded", meta.degraded ? "是" : "否"],
    ]));
    if (array(meta.degradation_reasons).length) {
      const warnings = make("div", "warning-list");
      array(meta.degradation_reasons).forEach((reason) => warnings.append(make("div", "warning-item", reasonText(reason))));
      metaBox.append(warnings);
    }
    container.append(panel("ProcessingMeta 与降级判定", "这是本次固定流水线的数量收敛记录。", metaBox, "soft"));
    container.append(panel("最终 warnings", "对外返回使用安全文案，Trace 中不包含完整异常。", warningsView(final.warnings)));
    container.append(jsonDrawer("查看最终原始 Trace", final));
    return container;
  }

  function renderRaw() {
    const container = make("div");
    container.append(make("div", "notice", "原始 Trace 仅用于排查专用视图未覆盖的特殊字段。主要执行过程请优先使用左侧阶段视图。"));
    const text = stringify(currentTrace());
    const raw = make("div", "code-panel");
    raw.append(make("div", "code-heading", `Trace JSON · ${text.length} 字符`));
    raw.append(make("pre", "", text.length > JSON_PREVIEW ? `${text.slice(0, JSON_PREVIEW)}\n…[界面限长，请下载完整 Trace]` : text));
    container.append(raw);
    return container;
  }

  function renderCurrent() {
    const definition = VIEW_DEFINITIONS.find((item) => item[0] === state.view) || VIEW_DEFINITIONS[0];
    els.viewTitle.textContent = definition[1];
    els.viewEyebrow.textContent = definition[3];
    els.viewer.replaceChildren();
    if (!state.response) {
      els.viewer.append(empty("请先执行一次 Processing。"));
      return;
    }
    const renderers = {
      overview: renderOverview,
      analyze: renderAnalyze,
      filter: renderFilter,
      build_markdown: renderBuildMarkdown,
      rerank: renderRerank,
      final: renderFinal,
      raw: renderRaw,
    };
    els.viewer.append((renderers[state.view] || renderOverview)());
  }

  async function run() {
    els.requestError.textContent = "";
    let payload;
    try { payload = JSON.parse(els.editor.value); }
    catch (error) {
      els.requestError.textContent = `JSON 解析失败：${error.message}`;
      return;
    }
    payload.rerank_input_mode = els.rerankMode.value;
    els.run.disabled = true;
    els.download.disabled = true;
    setStatus("running", "Processing 执行中");
    try {
      const uiPath = window.location.pathname.replace(/\/+$/, "");
      const response = await fetch(`${uiPath}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        credentials: "same-origin",
      });
      let body;
      try { body = await response.json(); } catch (_) { body = {}; }
      if (!response.ok) throw new Error(body.detail || body.rtnMsg || `HTTP ${response.status}`);
      state.response = body;
      state.view = "overview";
      els.resultLayout.hidden = false;
      els.download.disabled = false;
      const requestPanel = document.querySelector(".request-panel");
      if (requestPanel) requestPanel.open = false;
      renderSummary();
      renderPipeline();
      renderNav();
      renderCurrent();
      setStatus("success", body.final_result && body.final_result.degraded ? "已完成（存在降级）" : "已完成");
    } catch (error) {
      setStatus("error", "执行失败");
      els.requestError.textContent = error.message || String(error);
    } finally {
      els.run.disabled = false;
    }
  }

  els.run.addEventListener("click", run);
  els.rerankMode.addEventListener("change", () => {
    els.rerankModeHint.textContent = RERANK_MODE_HELP[els.rerankMode.value] || "";
  });
  els.format.addEventListener("click", () => {
    try {
      els.editor.value = stringify(JSON.parse(els.editor.value));
      els.requestError.textContent = "";
    } catch (error) { els.requestError.textContent = `JSON 解析失败：${error.message}`; }
  });
  document.querySelectorAll("[data-copy-target]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      copyText(document.getElementById(node.dataset.copyTarget).value);
    });
  });
  els.search.addEventListener("input", () => { if (state.response) renderCurrent(); });
  els.expand.addEventListener("click", () => {
    els.viewer.querySelectorAll("details").forEach((node) => {
      if (typeof node._ensureLoaded === "function") node._ensureLoaded();
      node.open = true;
    });
  });
  els.collapse.addEventListener("click", () => els.viewer.querySelectorAll("details").forEach((node) => { node.open = false; }));
  els.copyStage.addEventListener("click", () => copyText(stringify(viewData(state.view))));
  els.download.addEventListener("click", () => {
    if (!state.response || !state.response.trace) return;
    const blob = new Blob([stringify(state.response.trace)], { type: "application/json;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `processing-trace-${state.response.trace_id || "debug"}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  });
})();
