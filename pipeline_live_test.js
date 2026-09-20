// 检索链路 live 状态机测试:stub DOM 加载 index.html 内嵌脚本,
// 校验流式各时刻的阶段状态(进行中/等待中/已执行/跳过/降级)。
const fs = require('fs');
const html = fs.readFileSync(
  'D:/code/cmos-kb-agent/universal-agent/services/kbagent_service/static/index.html', 'utf8');
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];

// ── 最小 DOM stub ──
function fakeEl(id) {
  return { id, value: '', appendChild() {}, addEventListener() {},
           classList: { add() {}, remove() {} }, style: {},
           innerHTML: '', textContent: '', selected: false,
           scrollIntoView() {} };
}
const els = {};
global.document = {
  getElementById: id => els[id] || (els[id] = fakeEl(id)),
  createElement: () => fakeEl('tmp'),
  querySelectorAll: () => [],
  body: { appendChild() {}, removeChild() {} }
};
global.fetch = () => Promise.reject(new Error('offline'));
global.navigator = {};
global.window = {};
global.TextDecoder = require('util').TextDecoder;

// 1) 语法检查
new Function(js);
console.log('✓ JS 语法检查通过');

// 2) eval 并导出被测函数
const api = eval(js + ';({ renderPipeline, computeLiveKey, stageKeyOf, handleStreamMsg, setLoadingStage, doQuery, setQR(q, r) { lastQuestion = q; lastRegion = r; } })');

const flow = () => els['pipelineFlow'].innerHTML;
function segs() {
  // 按阶段切段,返回 [{name, state}]
  return flow().split('<div class="pipe-stage').slice(1).map(s => {
    const name = (s.match(/pipe-name">([^<]*)/) || [])[1] || '?';
    let state = 'done';
    if (s.includes('◐ 进行中')) state = 'running';
    else if (s.includes('○ 等待中')) state = 'pending';
    else if (s.includes('● 降级兜底')) state = 'bad';
    else if (s.includes('○')) state = 'skipped';
    return { name, state, seg: s };
  });
}
function expect(cond, msg) {
  if (!cond) { console.error('✗ FAIL:', msg, '\n  flow=', flow().slice(0, 800)); process.exitCode = 1; }
  else console.log('✓', msg);
}
const ev = (stage, event, payload) => ({ ts_ms: Date.now(), stage, event, payload: payload || {} });

// ── A. live 起始:无事件 → 阶段1进行中,其余等待 ──
api.setQR('异地能不能补办手机卡?', '350');
api.renderPipeline([], null, { live: true });
let st = segs();
expect(st.length === 6, 'A 渲染 6 个阶段');
expect(st[0].state === 'running', 'A 无事件时阶段① 进行中');
expect(st.slice(1).every(x => x.state === 'pending'), 'A 其余阶段全部等待中');
expect(st[0].seg.includes('异地能不能补办手机卡'), 'A 问题 chip 用 lastQuestion 兜底');
expect(st[0].seg.includes('350'), 'A 归属地 chip 用 lastRegion 兜底');

// ── B. run.start 后:仍在阶段① ──
let events = [ev('run', 'start', { query: '异地能不能补办手机卡?', region_code: '350' })];
api.renderPipeline(events, null, { live: true });
st = segs();
expect(st[0].state === 'running' && st[1].state === 'pending', 'B run.start → ①进行中 ②等待');

// ── C. cache.miss 后:完成事件前推 → ②已执行 ③进行中 ──
events.push(ev('cache', 'miss', {}));
api.renderPipeline(events, null, { live: true });
st = segs();
expect(st[0].state === 'done', 'C ①已执行');
expect(st[1].state === 'done', 'C cache.miss 后②已执行');
expect(st[2].state === 'running', 'C 前推:③检索进行中');
expect(st[3].state === 'pending' && st[5].state === 'pending', 'C ④⑥等待中');

// ── D. 检索中:round1.recall → ③进行中且有召回 chip ──
events.push(ev('retrieval.round1', 'recall', { channel: 'intergrate_all', titles: ['补卡手册', '套餐资费'], scores: [12.5, 8.1], region_code: '350' }));
api.renderPipeline(events, null, { live: true });
st = segs();
expect(st[2].state === 'running', 'D round1.recall → ③进行中');
expect(st[2].seg.includes('补卡手册'), 'D 召回明细已渲染');
expect(st[2].seg.includes('一体化流水线'), 'D 通道 chip 正确');

// ── E. retrieval.done → 前推 ④进行中 ──
events.push(ev('retrieval', 'done', { count: 12 }));
api.renderPipeline(events, null, { live: true });
st = segs();
expect(st[2].state === 'done' && st[2].seg.includes('✓ 检索完成'), 'E ③已执行+完成chip');
expect(st[3].state === 'running', 'E retrieval.done 前推 → ④进行中');

// ── F. processing 完成(processed_chunks_adapted)→ 前推 ⑤进行中 ──
events.push(ev('processing.knowledge', 'orchestrator_step', { step: 'analyze' }));
events.push(ev('processing.knowledge', 'processed_chunks_adapted', { count: 3 }));
api.renderPipeline(events, null, { live: true });
st = segs();
expect(st[3].state === 'done' && st[3].seg.includes('精选输出'), 'F ④已执行+精选chip');
expect(st[4].state === 'running', 'F adapted 前推 → ⑤进行中');

// ── G. answer 批量一致性校验 → ⑤进行中 ──
events.push(ev('answer', 'materials', { chunk_ids: ['kb_1#p1'] }));
events.push(ev('answer', 'consistency_check', { consistent: true, issue_count: 0, issues: [] }));
api.renderPipeline(events, null, { live: true });
st = segs();
expect(st[4].state === 'running', 'G consistency_check → ⑤仍进行中');
expect(st[4].seg.includes('话术一致性校验通过'), 'G 一致性通过 chip');
// 不一致时:warn chip 带待核实条数
const warnEvents = events.concat([ev('answer', 'consistency_check', { consistent: false, issue_count: 2, issues: ['资费无依据', '条件不符'] })]);
api.renderPipeline(warnEvents, null, { live: true });
st = segs();
expect(st[4].seg.includes('2') && st[4].seg.includes('需核实'), 'G 不一致 → 待核实条数 chip');

// ── H. finalize.done(live 末态)→ ⑥进行中(等 final 帧) ──
events.push(ev('finalize', 'done', { elapsed_ms: 45600 }));
api.renderPipeline(events, null, { live: true });
st = segs();
expect(st[5].state === 'running', 'H finalize.done(live)→ ⑥进行中');

// ── I. 非 live(最终帧):全部已执行,无进行中/等待中 ──
api.renderPipeline(events, { elapsedMs: 45600 });
st = segs();
expect(st.every(x => x.state === 'done'), 'I 非live 全阶段已执行');
expect(!flow().includes('进行中') && !flow().includes('等待中'), 'I 无进行中/等待中残留');
expect(st[5].seg.includes('45.6'), 'I 端到端耗时 chip');

// ── J. 缓存命中(live):③④⑤跳过,⑥进行中 ──
const cacheEvents = [ev('run', 'start', { query: 'q', region_code: '000' }), ev('cache', 'hit', {})];
api.renderPipeline(cacheEvents, null, { live: true });
st = segs();
expect(st[1].state === 'done', 'J cache.hit → ②已执行');
expect(st[2].state === 'skipped' && st[2].seg.includes('缓存命中,已跳过'), 'J ③跳过(缓存命中)');
expect(st[3].state === 'skipped' && st[4].state === 'skipped', 'J ④⑤跳过');
expect(st[5].state === 'running', 'J 前推 → ⑥进行中');
api.renderPipeline(cacheEvents, { elapsedMs: 300 });
st = segs();
expect(st[5].state === 'done' && st[5].seg.includes('缓存命中直接返回'), 'J 非live缓存命中 ⑥已执行+chip');

// ── K. 降级(live):⑥bad,③④⑤skipped(链路降级) ──
const degEvents = [ev('run', 'start', {}), ev('cache', 'miss', {}), ev('degrade', 'triggered', { error: 'timeout' }), ev('degrade', 'done', { reason: '处理超时', hit_count: 5 })];
api.renderPipeline(degEvents, null, { live: true });
st = segs();
expect(st[5].state === 'bad', 'K 降级 → ⑥降级兜底');
expect(st[5].seg.includes('处理超时') && st[5].seg.includes('5'), 'K 降级原因+兜底条数 chip');
expect(st[3].state === 'skipped' && st[3].seg.includes('链路降级,未执行'), 'K ④跳过(链路降级)');

// ── L. 非 live 且 trace 中断(如超时错误信封):后段 skipped ──
api.renderPipeline([ev('run', 'start', {}), ev('cache', 'miss', {})], null);
st = segs();
expect(st[2].state === 'skipped' && st[5].state === 'skipped', 'L 无数据阶段 skipped');
expect(st[5].seg.includes('未收到完成事件'), 'L ⑥注明未收到完成事件');

// ── M. handleStreamMsg:trace 帧累积渲染 + final 帧走 render ──
els['pipelineFlow'].innerHTML = '';
api.handleStreamMsg({ type: 'trace', ts_ms: 1, stage: 'run', event: 'start', payload: { query: '流式测试', region_code: '591' } });
expect(flow().includes('流式测试') && flow().includes('◐ 进行中'), 'M trace 帧驱动 live 渲染');
expect(els['traceTimeline'].innerHTML.includes('run') || els['traceCount'].textContent.includes('1'), 'M 时间线同步渲染');
api.handleStreamMsg({ type: 'final', response: { rtnCode: '50002', rtnMsg: 'timeout', object: { processTrace: events } } });
expect(els['bannerBox'].innerHTML.includes('50002'), 'M final 错误信封 → 错误横幅');
expect(flow().includes('✓ 检索完成'), 'M final 后用完整 trace 重渲染链路');

// ── N. setLoadingStage:阶段名归一化 ──
api.setLoadingStage('retrieval.round2');
expect(els['loadingText'].textContent.includes('检索'), 'N retrieval.round2 → 检索');
api.setLoadingStage('processing.knowledge');
expect(els['loadingText'].textContent.includes('知识处理'), 'N processing.knowledge → 知识处理');
api.setLoadingStage('answer');
expect(els['loadingText'].textContent.includes('答案生成'), 'N answer → 答案生成');

// ── N2. 检索关键信息:关键词提取 / 双路条数 / 查询改写 / 验证器 / 循环汇总 ──
const rtEvents = [
  ev('run', 'start', { query: 'q', region_code: '350' }),
  ev('cache', 'miss', {}),
  ev('retrieval.round1', 'recall', { channel: 'intergrate_all', region_code: '350', keywords: ['异地', '补卡', '手机卡'], keyword_count: 8, vector_count: 6, merged_count: 12, titles: ['补卡手册'], scores: [9.9] }),
  ev('retrieval.verify', 'done', { round: 1, status: 'failed', reason_codes: ['insufficient_evidence'], evidence_count: 0, summary: '候选不足以回答', top_titles: ['补卡手册'] }),
  ev('retrieval.verify', 'feedback', { round: 1, suggested_query: '异地补卡需要什么材料', suggested_keywords: ['异地补卡', '材料'], missing_aspects: ['办理材料'], retry_strategy: '改写关键词' }),
  ev('retrieval.round1', 'query_rewrite', { last_keywords: ['异地', '补卡'], rewritten_keywords: ['异地补卡', '材料'] }),
  ev('retrieval.round2', 'recall', { channel: 'intergrate_all', region_code: '350', keywords: ['异地补卡', '材料'], keyword_count: 5, vector_count: 5, merged_count: 9, titles: ['补卡材料清单'], scores: [11.2] }),
  ev('retrieval.verify', 'done', { round: 2, status: 'passed', reason_codes: [], evidence_count: 2, summary: 'Top3 可回答', top_titles: ['补卡材料清单', '补卡手册'] }),
  ev('retrieval', 'loop_result', { success: true, iterations: 2, reason: 'verified' }),
  ev('retrieval', 'done', { count: 9 })
];
api.renderPipeline(rtEvents, null, { live: true });
st = segs();
expect(st[2].seg.includes('异地 / 补卡 / 手机卡'), 'N2 第1轮提取关键词 chip');
expect(st[2].seg.includes('关键词 <b>8</b> 条 · 向量 <b>6</b> 条 → 去重 <b>12</b> 条'), 'N2 双路召回条数 chip');
expect(st[2].seg.includes('第 1 轮验证:⚠ 未通过'), 'N2 第1轮验证器未通过 chip');
expect(st[2].seg.includes('insufficient_evidence'), 'N2 验证器原因码');
expect(st[2].seg.includes('查询改写') && st[2].seg.includes('异地补卡 / 材料'), 'N2 查询改写 chip');
expect(st[2].seg.includes('第 2 轮验证:✓ 通过') && st[2].seg.includes('证据 <b>2</b> 篇'), 'N2 第2轮验证器通过 chip');
expect(st[2].seg.includes('共 <b>2</b> 轮'), 'N2 GoalLoop 汇总 chip');
expect(st[2].seg.includes('建议检索语句') && st[2].seg.includes('异地补卡需要什么材料'), 'N2 明细含验证器建议');
expect(st[2].seg.includes('缺失方面') && st[2].seg.includes('办理材料'), 'N2 明细含缺失方面');
expect(st[2].seg.includes('候选不足以回答') && st[2].seg.includes('Top3 可回答'), 'N2 明细含验证器 summary');
expect(st[2].seg.includes('被验证的 Top3') && st[2].seg.includes('《补卡材料清单》'), 'N2 明细含被验证 Top3 标题');
// 时间序:第1轮召回 → 第1轮验证 → 改写 → 第2轮召回 → 第2轮验证
const segOrder = ['第 <b>1</b> 轮召回', '第 1 轮验证:⚠ 未通过', '第 1 轮后查询改写', '第 <b>2</b> 轮召回', '第 2 轮验证:✓ 通过']
  .map(s => st[2].seg.indexOf(s));
expect(segOrder.every((x, i) => x >= 0 && (i === 0 || x > segOrder[i - 1])), 'N2 chips 按召回→验证→改写时间序排列');

// ── N3. 同参数重复召回去重:recall_cached → 缓存复用 chip ──
api.renderPipeline([
  ev('run', 'start', { query: '销户怎么办理', region_code: '福建' }),
  ev('cache', 'miss', {}),
  ev('retrieval.round1', 'recall', { channel: 'intergrate_all', region_code: '福建', keywords: ['销户'], keyword_count: 88, vector_count: 110, merged_count: 178, titles: ['销户办理指南'], scores: [9] }),
  ev('retrieval.round1', 'recall_cached', { channel: 'intergrate_all', region_code: '福建', keywords: ['销户'], recalled: 178 }),
  ev('retrieval.verify', 'done', { round: 1, status: 'passed', reason_codes: [], evidence_count: 1, summary: '可回答', top_titles: ['销户办理指南'] }),
  ev('retrieval', 'loop_result', { success: true, iterations: 1, reason: '目标验证通过' })
], null, { live: true });
st = segs();
expect(st[2].seg.includes('重复调用(同参数)') && st[2].seg.includes('复用缓存结果'), 'N3 同参数重复调用 → 缓存 chip');
expect(st[2].seg.includes('命中请求级缓存,复用上次 178 条结果'), 'N3 缓存明细含复用条数');
expect(st[2].seg.includes('第 1 轮验证:✓ 通过') && st[2].seg.includes('共 <b>1</b> 轮'), 'N3 单轮验证通过 + 循环汇总');
expect(st[2].seg.includes('《销户办理指南》'), 'N3 被验证 Top3 标题');

// ── O. doQuery 流式全流程:主区占位切换 + 右侧栏链路点亮 + final 渲染 ──
// 用假 fetch 返回一段 SSE 流,验证 welcome→searching→results 的显隐编排
const doQueryTest = (function testDoQueryStream() {
  const disp = id => document.getElementById(id).style.display;
  const gid = id => document.getElementById(id);
  // 重置显隐,模拟初始加载态
  gid('welcomeCard').style.display = '';
  gid('searchingCard').style.display = 'none';
  gid('results').style.display = 'none';
  gid('question').value = '异地能不能补办手机卡?';
  gid('province').value = '350';
  gid('city').value = '';

  const traceEvents = [
    { ts_ms: 1, stage: 'run', event: 'start', payload: { query: '异地能不能补办手机卡?', region_code: '350' } },
    { ts_ms: 2, stage: 'cache', event: 'miss', payload: {} },
    { ts_ms: 3, stage: 'retrieval.round1', event: 'recall', payload: { channel: 'intergrate_all', titles: ['补卡手册'], scores: [9.9] } },
    { ts_ms: 4, stage: 'retrieval', event: 'done', payload: { count: 1 } },
    { ts_ms: 4.5, stage: 'processing.knowledge', event: 'rerank', payload: { input_count: 3, top_count: 1, degraded: false, top_titles: ['补卡手册'] } },
    { ts_ms: 5, stage: 'processing.knowledge', event: 'processed_chunks_adapted', payload: { count: 1 } },
    { ts_ms: 6, stage: 'answer', event: 'materials', payload: { chunk_ids: ['kb_1#p1'] } },
    { ts_ms: 7, stage: 'answer', event: 'consistency_check', payload: { consistent: true, issue_count: 0, issues: [] } },
    { ts_ms: 8, stage: 'finalize', event: 'done', payload: { elapsed_ms: 4200 } }
  ];
  const sse = traceEvents.map(e =>
      'data: ' + JSON.stringify({ type: 'trace', ...e }) + '\n\n').join('') +
    'data: ' + JSON.stringify({ type: 'final', response: { rtnCode: '0', rtnMsg: 'success', object: { requestId: 'r1', traceId: 't1', elapsedMs: 4200, degraded: false, sources: [{ chunkId: 'kb_1#p1', docId: 'kb_1', docTitle: '补卡手册', relevance: 100, keyFragment: '异地可以补办手机卡', content: '异地可以补办手机卡,携带身份证件到营业厅办理。', updatedAt: '2026-06-10', stale: false }], usability: { level: 'directly_usable', reasons: [], uncovered: [] }, script: '您好,异地可以补办手机卡,携带身份证件即可。', handlingSuggestion: '引导客户到就近营业厅办理。', processTrace: traceEvents } } }) + '\n\n';

  function fakeStreamResp() {
    const chunks = [new TextEncoder().encode(sse)];
    let i = 0;
    return {
      ok: true, status: 200,
      headers: { get: () => 'text/event-stream' },
      body: { getReader: () => ({ read: () => Promise.resolve(i < chunks.length ? { done: false, value: chunks[i++] } : { done: true }) }) }
    };
  }
  const realFetch = global.fetch;
  global.fetch = (url) => /\/retrieve\/stream$/.test(url)
    ? Promise.resolve(fakeStreamResp())
    : Promise.reject(new Error('should not call plain'));

  const p = api.doQuery();
  // 同步阶段:进入"检索中"占位,结果区隐藏
  expect(disp('welcomeCard') === 'none', 'O doQuery → welcome 隐藏');
  expect(disp('searchingCard') === '', 'O doQuery → searching 占位显示');
  expect(disp('results') === 'none', 'O doQuery → 结果区先隐藏');
  expect(els['submitBtn'].disabled === true, 'O 查询期间按钮禁用');
  expect(flow().includes('◐ 进行中'), 'O 右侧链路图立即点亮进行中');

  return p.then(() => {
    // final 帧后:主区切到结果,占位退场,按钮恢复
    expect(disp('searchingCard') === 'none', 'O final 后 searching 退场');
    expect(disp('results') === 'block', 'O final 后结果区显示');
    expect(els['submitBtn'].disabled === false, 'O 完成后按钮恢复');
    expect(!flow().includes('进行中'), 'O final 后链路图无进行中残留');
    expect(flow().includes('✓ 检索完成'), 'O final 后链路图完整态');
    expect(flow().includes('Top3:《补卡手册》'), 'O ④阶段 Top3 标题 chip');
    expect(flow().includes('重排后 Top3 文档标题'), 'O ④阶段 Top3 标题明细');
    expect(els['statDocs'].textContent === 1, 'O 结果概览已渲染(1 篇文档)');
    expect(els['statTopRel'].textContent === '100%', 'O 最高相关度 100%');
  }).finally(() => { global.fetch = realFetch; });
})();

// ── P. 流式接口不可用(404)→ 自动回退普通 /retrieve ──
const fallbackTest = doQueryTest.then(() => {
  const gid = id => document.getElementById(id);
  gid('results').style.display = 'none';
  gid('question').value = '异地能不能补办手机卡?';
  const envelope = { rtnCode: '0', rtnMsg: 'success', object: { requestId: 'r2', traceId: 't2', elapsedMs: 900, degraded: false, sources: [], usability: { level: 'directly_usable', reasons: [], uncovered: [] }, script: '您好', handlingSuggestion: '', processTrace: [] } };
  const realFetch = global.fetch;
  let plainCalled = false;
  global.fetch = (url) => {
    if (/\/retrieve\/stream$/.test(url))
      return Promise.resolve({ ok: false, status: 404, headers: { get: () => 'text/plain' }, text: () => Promise.resolve('Not Found') });
    plainCalled = true;
    return Promise.resolve({ ok: true, headers: { get: () => 'application/json' }, json: () => Promise.resolve(envelope) });
  };
  return api.doQuery().then(() => {
    expect(plainCalled, 'P 流式 404 → 回退调用了普通 /retrieve');
    expect(gid('results').style.display === 'block', 'P 回退后结果区显示');
    expect(gid('searchingCard').style.display === 'none', 'P 回退后占位退场');
    expect(gid('bannerBox').innerHTML.includes('话术可直接使用'), 'P 回退结果正常渲染');
  }).finally(() => { global.fetch = realFetch; });
});

fallbackTest.then(() => {
  console.log(process.exitCode ? '\n存在失败用例' : '\n全部通过');
}).catch(e => {
  console.error('✗ O/P 用例异常:', e && e.message);
  console.log('\n存在失败用例');
  process.exitCode = 1;
});
