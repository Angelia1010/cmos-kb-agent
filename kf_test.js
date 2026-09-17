const fs = require('fs');
const html = fs.readFileSync('D:/code/cmos-kb-agent/universal-agent/services/kbagent_service/static/index.html', 'utf8');
const js = html.match(/<script>([\s\S]*)<\/script>/)[1];
// ── 最小 DOM stub,让脚本可加载 ──
function fakeEl() { return { value: '', appendChild() {}, addEventListener() {}, classList: { add() {}, remove() {} }, style: {}, innerHTML: '', textContent: '', selected: false }; }
const els = {};
global.document = { getElementById: id => els[id] || (els[id] = fakeEl()), createElement: fakeEl, querySelectorAll: () => [], body: { appendChild() {}, removeChild() {} } };
global.fetch = () => Promise.reject(new Error('offline'));
global.navigator = {};
global.window = {};
const api = eval(js + ';({ normalizeWithMap, analyzeSnippet, pickKeyFragment, clipRanges, findQuestionMatches, renderTwoColor, groupByDoc, setQ(q) { lastQuestion = q; } })');
const { normalizeWithMap, analyzeSnippet, pickKeyFragment, clipRanges, findQuestionMatches, renderTwoColor, groupByDoc } = api;

// ── 测试数据:整篇文档,关键答案埋在中部 ──
api.setQ('异地能不能补办手机卡?需要什么材料?');
const filler = '本章节介绍套餐资费的整体背景与历史沿革,与补卡业务无直接关系。'.repeat(6);
const doc = filler +
  '客户在异地可以补办手机卡,机主本人携带有效身份证件到自有营业厅即可办理。补卡费用为10元,当场领卡。' +
  filler;
const answerCorpus = '您好,异地是可以补办手机卡的,需要机主本人携带有效身份证件到自有营业厅办理,费用10元。';
const corpus = normalizeWithMap(answerCorpus).text;
const r = analyzeSnippet(doc, corpus);
const kf = pickKeyFragment(doc, r.ranges);
console.log('used:', r.ranges.length > 0, 'kf.fallback:', kf.fallback);
console.log('片段位置:', kf.start, '-', kf.end, '/ 全文', doc.length, '字');
console.log('⭐ 关键片段内容:', JSON.stringify(doc.slice(kf.start, kf.end)));
const kfText = doc.slice(kf.start, kf.end);
const rendered = renderTwoColor(kfText, clipRanges(r.ranges, kf.start, kf.end), findQuestionMatches(kfText).ranges);
console.log('渲染含黄色mark:', rendered.includes('<mark>'), '含蓝色mark:', rendered.includes('class="q"'));

// 短文档路径
const kf2 = pickKeyFragment('全文很短,直接展示即可。', []);
console.log('短文档 kf:', JSON.stringify(kf2));

// 完全无信号文档 → 文首摘要 fallback
const irrelevant = '天气预报显示明天有小雨,气温二十度左右,适合出行。'.repeat(5);
const kf3 = pickKeyFragment(irrelevant, analyzeSnippet(irrelevant, '').ranges);
console.log('无关文档 fallback:', kf3.fallback, 'len:', kf3.end - kf3.start);

// 分组排序:弱相关文档 vs 强相关文档
const analyzed = [
  { docTitle: 'A弱相关', snippet: irrelevant, used: false, kfScore: pickKeyFragment(irrelevant, []).score, updatedAt: '2026-01-01', stale: false },
  { docTitle: 'B强相关', snippet: doc, used: true, kfScore: kf.score, updatedAt: '2026-01-01', stale: false },
];
const groups = groupByDoc(analyzed.map(a => ({ ...a, ranges: [], quotes: [], kf: null })));
console.log('分组排序:', groups.map(g => g.title).join(' → '));

// 断言
const assert = require('assert');
assert.ok(r.ranges.length > 0, '答案引用区间应命中');
assert.ok(!kf.fallback, '有信号时不应走 fallback');
assert.ok(doc.slice(kf.start, kf.end).includes('补办手机卡'), '关键片段应含答案句');
assert.ok(rendered.includes('<mark>'), '渲染应含黄色高亮');
assert.strictEqual(groups[0].title, 'B强相关', '强相关文档应排前');
console.log('✓ 关键片段回归测试全部通过');
