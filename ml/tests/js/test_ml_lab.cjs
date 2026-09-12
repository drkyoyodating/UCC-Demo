'use strict';
// Unit tests for docs/assets/ml-lab.js (plan C Task 5). Run: node --test ml/tests/js/test_ml_lab.cjs
// No browser and no npm packages: a minimal fake DOM whose innerHTML setter throws proves every
// write goes through textContent, and a stub fetch proves which files the page requests.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const SOURCE = path.join(__dirname, '..', '..', '..', 'docs', 'assets', 'ml-lab.js');
const Lab = require(SOURCE);

const EVIL = '<img src=x onerror=alert(1)>';

// The real labels file holds two rounds labelled under DIFFERENT arrangements, so the sentence the
// documents publish is the pooled one and no single round's phrase is true of the file. The
// fixtures mirror that, because rendering it correctly is the thing that can go wrong.
const BY_ROUND = {
  main_v1: 'model-labelled, two independent blind passes, disagreements retained as unresolved',
  pilot_v1: 'model-labelled, founder-adjudicated'
};
const POOLED = 'mixed by round -- main_v1: ' + BY_ROUND.main_v1 + '; pilot_v1: ' + BY_ROUND.pilot_v1;

class FakeNode {
  constructor(doc, tag) {
    this.ownerDocument = doc;
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.parentNode = null;
    this.text = '';
    this.disabled = false;
    this.value = '';
  }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  removeChild(child) { this.children = this.children.filter((c) => c !== child); child.parentNode = null; return child; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn); }
  focus() { this.ownerDocument.activeElement = this; }
  set textContent(value) { this.children = []; this.text = String(value); }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(''); }
  set innerHTML(value) { throw new Error('innerHTML is forbidden'); }
  get innerHTML() { throw new Error('innerHTML is forbidden'); }
  insertAdjacentHTML() { throw new Error('insertAdjacentHTML is forbidden'); }
}

class FakeDocument {
  constructor() { this.activeElement = null; }
  createElement(tag) { return new FakeNode(this, tag); }
}

function walk(node, visit) { visit(node); node.children.forEach((child) => walk(child, visit)); }
function byRole(rootNode, role) { const out = []; walk(rootNode, (n) => { if (n.attributes['data-role'] === role) out.push(n); }); return out; }
function one(rootNode, role) {
  const found = byRole(rootNode, role);
  assert.equal(found.length, 1, `expected exactly one [data-role=${role}], found ${found.length}`);
  return found[0];
}
function byTag(rootNode, tag) { const out = []; walk(rootNode, (n) => { if (n.tagName === tag.toUpperCase()) out.push(n); }); return out; }
function flush() { return new Promise((resolve) => setTimeout(resolve, 0)).then(() => new Promise((resolve) => setTimeout(resolve, 0))); }
function respond(status, body) { return { status, ok: status >= 200 && status < 300, json: () => Promise.resolve(body) }; }

function ci(estimate, lower, upper) { return { estimate, lower, upper, n_resamples: 2000, n_failed: 0 }; }
function block(tp, fp, fn, tn, p, r, f) {
  return { tp, fp, fn, tn, weighted_precision: p, weighted_recall: r, weighted_f1: f,
    ci: { weighted_precision: ci(p, p - 0.05, p + 0.03), weighted_recall: ci(r, r - 0.06, r + 0.05), weighted_f1: ci(f, f - 0.05, f + 0.04) } };
}

function fixtures(apiUrl, preview) {
  const manifest = {
    schema_version: 1, generated_at: '2026-09-12T10:00:00Z', current: 'r1',
    releases: [{ release_id: 'r1', published_at: '2026-09-12T10:00:00Z', metrics_url: 'releases/r1/metrics.json',
      examples_url: 'releases/r1/examples.json', model_card_url: 'releases/r1/model-card.json', api_url: apiUrl }]
  };
  if (preview) manifest.preview = true;
  const stratum = (name) => ({ stratum: name, n: 150, positives: 60, model: block(50, 5, 10, 85, 0.9, 0.8, 0.85), rules: block(40, 2, 20, 88, 0.95, 0.66, 0.78), delta: {} });
  const metrics = {
    schema_version: 1, release_id: 'r1', generated_at: '2026-09-12T10:00:00Z', label_provenance: POOLED,
    label_provenance_by_round: BY_ROUND,
    test: {
      split: 'test', evaluated_at: '2026-09-11T08:00:00Z', forced: false,
      frozen: { threshold: 0.62, status: 'production', score_type: 'calibrated_probability' },
      evaluation: {
        n: 612, positives: 201, negatives: 411,
        weights: ['CO:accepted', 'CO:rejected', 'CT:accepted', 'CT:rejected'].map((s) => ({ stratum: s, N_h: 1234567, n_h: 150, w_h: 8230.4, unsampled: false })),
        model: block(180, 12, 21, 399, 0.912, 0.83, 0.87), rules: block(150, 3, 51, 408, 0.98, 0.61, 0.75),
        delta_model_minus_rules: { delta_weighted_precision: ci(-0.041, -0.09, -0.012), delta_weighted_recall: ci(0.22, 0.15, 0.3) },
        per_region: [{ region: 'CO', n: 300, positives: 100, model: block(1, 1, 1, 1, 0.9, 0.8, 0.85), rules: block(1, 1, 1, 1, 0.97, 0.6, 0.74), delta: {} },
          { region: 'CT', n: 312, positives: 101, model: block(1, 1, 1, 1, 0.92, 0.85, 0.88), rules: block(1, 1, 1, 1, 0.99, 0.62, 0.76), delta: {} }],
        per_stratum: ['CO:accepted', 'CO:rejected', 'CT:accepted', 'CT:rejected'].map(stratum),
        bootstrap: { n_resamples: 2000, seed: 20260912, level: 0.95, n_clusters: 540, straddling_groups: 0 }
      },
      calibration: { ece: 0.04, n_bins: 10 },
      unresolved: [{ region: 'CO', stratum: 'CO:accepted', n_labelled: 180, n_unresolved: 20, share_unresolved: 0.111 },
        { region: 'total', stratum: 'all', n_labelled: 700, n_unresolved: 88, share_unresolved: 0.1257 }],
      verdict: { text: 'model differs from rules' },
      labels: {
        disclosure: POOLED,
        disclosure_by_round: BY_ROUND,
        counts_by_status: { model_agreed: 1810, founder_confirmed: 80, founder_adjudicated: 310, blind_unresolved: 83, blind_repeat: 240 },
        counts_by_round: { pilot_v1: 220, main_v1: 2220 },
        pass_agreement: { pilot_v1: { n: 200, agreed: 171, rate: 0.855 }, main_v1: { n: 2000, agreed: 1720, rate: 0.86 } },
        founder_audit: { pilot_v1: { n_audited: 40, n_confirmed: 37, n_overturned: 3, agreement_rate: 0.925 } },
        repeat_consistency: { pass_a: { n: 120, consistent: 114, rate: 0.95 }, pass_b: { n: 120, consistent: 111, rate: 0.925 } }
      }
    }
  };
  const example = (name, label, decision, qualifies) => ({ case_id: 'c-' + name.length, region: 'CO', borrower_name: name, lender_names: ['WAGNER EQUIPMENT CO'],
    city: 'DENVER', state: 'CO', baseline_qualifies: qualifies, baseline_route: qualifies ? 'lender' : 'neither', score: 0.83, score_type: 'calibrated_probability',
    decision, threshold: 0.62, label, curated: true, precomputed: true });
  const examples = {
    schema_version: 1, release_id: 'r1', curated: true, precomputed: true, label_provenance: POOLED,
    label_provenance_by_round: BY_ROUND,
    examples: [example(EVIL, 'NOT_RELEVANT', 'suggest_relevant', false), example('SMITH EXCAVATING LLC', 'RELEVANT', 'suggest_relevant', true),
      example('ALPINE DENTAL PC', 'NOT_RELEVANT', 'review_needed', false)]
  };
  const card = { schema_version: 1, release_id: 'r1', label_provenance: POOLED,
    sections: [{ heading: 'Intended use', text: 'Screening.' }, { heading: 'Limitations', text: '- Names only.\n- Not a forecast.' }] };
  return { manifest, metrics, examples, card };
}

function routesFor(data) {
  return {
    'GET data/manifest.json': respond(200, data.manifest),
    'GET data/releases/r1/metrics.json': respond(200, data.metrics),
    'GET data/releases/r1/examples.json': respond(200, data.examples),
    'GET data/releases/r1/model-card.json': respond(200, data.card)
  };
}

function stubFetch(routes, log) {
  return (url, init) => {
    const method = (init && init.method) || 'GET';
    log.push({ url, method, body: init && init.body, headers: init && init.headers });
    const key = method + ' ' + url;
    if (!Object.prototype.hasOwnProperty.call(routes, key)) return Promise.resolve(respond(404, { detail: 'Not Found' }));
    const route = routes[key];
    return typeof route === 'function' ? route(url, init) : Promise.resolve(route);
  };
}

function mount(routes, extraConfig) {
  const doc = new FakeDocument();
  const container = doc.createElement('main');
  const log = [];
  const cleanup = Lab.mountMLLab(container, Object.assign({ dataBase: 'data/', fetch: stubFetch(routes, log) }, extraConfig || {}));
  return { doc, container, log, cleanup };
}

test('formatCell never separates years, ids, hashes or thresholds and always separates counts', () => {
  assert.equal(Lab.formatCell('loan_year', 1996), '1996');
  assert.equal(Lab.formatCell('year', '2014'), '2014');
  assert.equal(Lab.formatCell('file_id', 20131234567), '20131234567');
  assert.equal(Lab.formatCell('case_id', '0012345'), '0012345');
  assert.equal(Lab.formatCell('release_id', 3141592653), '3141592653');
  assert.equal(Lab.formatCell('input_hash', 'ab'.repeat(32)), 'ab'.repeat(32));
  assert.equal(Lab.formatCell('candidates_sha256', 1234567), '1234567');
  assert.equal(Lab.formatCell('threshold', 0.62), '0.6200');
  assert.equal(Lab.formatCell('threshold', 1234.5), '1234.5000');
  assert.equal(Lab.formatCell('n', 12345), '12,345');
  assert.equal(Lab.formatCell('N_h', 1234567), '1,234,567');
  assert.equal(Lab.formatCell('tp', 7), '7');
  assert.equal(Lab.formatCell('w_h', 12345.67), '12,345.7');
  assert.equal(Lab.formatCell('weighted_precision', 0.9123), '91.2%');
  assert.equal(Lab.formatCell('score', 0.83456), '0.835');
  assert.equal(Lab.formatCell('contribution', -0.13), '−0.130');
  assert.equal(Lab.formatCell('contribution', 0.4125), '+0.412');
  assert.equal(Lab.formatCell('elapsed_ms', 1234.56), '1,234.6 ms');
  assert.equal(Lab.formatCell('baseline_qualifies', false), 'rejected');
  assert.equal(Lab.formatCell('decision', 'suggest_relevant'), 'Suggest relevant');
  assert.equal(Lab.formatCell('lender_names', ['A BANK', 'B FINANCE']), 'A BANK; B FINANCE');
  assert.equal(Lab.formatCell('lender_names', []), '—');
  assert.equal(Lab.formatCell('n', null), '—');
  assert.equal(Lab.formatCell('score', Number.NaN), '—');
  assert.equal(Lab.formatCell('borrower_name', EVIL), EVIL);
});

test('estimates and paired deltas carry their interval and level', () => {
  assert.equal(Lab.formatEstimate({ estimate: 0.9123, lower: 0.86, upper: 0.951 }, 0.95), '91.2% (95% CI 86.0–95.1%)');
  assert.equal(Lab.formatEstimate({ estimate: 0.5, lower: null, upper: null }, 0.95), '50.0% (no interval)');
  assert.equal(Lab.formatEstimate({ estimate: null }, 0.95), '—');
  assert.equal(Lab.formatDelta({ estimate: 0.041, lower: -0.012, upper: 0.09 }, 0.9), '+4.1 pts (90% CI −1.2 to +9.0 pts)');
  assert.equal(Lab.formatDelta({ estimate: 0, lower: 0, upper: 0 }, 0.95), '0.0 pts (95% CI 0.0 to 0.0 pts)');
});

test('resolveApiUrl: explicit config wins, only https or loopback http is accepted', () => {
  const release = { api_url: 'https://ucc-ml-api-abc-uc.a.run.app/' };
  assert.equal(Lab.resolveApiUrl({}, release), 'https://ucc-ml-api-abc-uc.a.run.app');
  assert.equal(Lab.resolveApiUrl({ apiUrl: null }, release), null);
  assert.equal(Lab.resolveApiUrl({ apiUrl: 'http://127.0.0.1:8000' }, release), 'http://127.0.0.1:8000');
  assert.equal(Lab.resolveApiUrl({}, { api_url: 'http://api.example.com' }), null);
  assert.equal(Lab.resolveApiUrl({}, { api_url: 'https://api.example.com/v1/predict' }), null);
  assert.equal(Lab.resolveApiUrl({}, { api_url: null }), null);
});

test('selectErrorExamples keeps only resolved labels that disagree with the decision', () => {
  const data = fixtures(null, false);
  const extra = [{ label: 'INSUFFICIENT_EVIDENCE', decision: 'suggest_relevant' }, { label: 'RELEVANT', decision: 'review_needed', borrower_name: 'MISS' },
    { label: 'RELEVANT', decision: 'unknown' }, { decision: 'suggest_relevant' }];
  const errors = Lab.selectErrorExamples(data.examples.examples.concat(extra));
  assert.deepEqual(errors.map((e) => e.borrower_name), [EVIL, 'MISS']);
  assert.deepEqual(Lab.selectErrorExamples(undefined), []);
});

test('buildPredictBody enforces the API bounds before any request', () => {
  assert.deepEqual(Lab.buildPredictBody('  Example Excavation LLC ', 'A BANK\n\n B FINANCE \n', 'CT'),
    { body: { borrower_name: 'Example Excavation LLC', lender_names: ['A BANK', 'B FINANCE'], region: 'CT' } });
  assert.match(Lab.buildPredictBody('   ', '', 'CO').error, /borrower name/);
  assert.match(Lab.buildPredictBody('X'.repeat(301), '', 'CO').error, /300 characters/);
  assert.match(Lab.buildPredictBody('X', Array(21).fill('L').join('\n'), 'CO').error, /at most 20/);
  assert.match(Lab.buildPredictBody('X', 'L'.repeat(301), 'CO').error, /Lender name 1/);
});

test('missingPaths reports a renamed key, including inside every list element', () => {
  const data = fixtures(null, false);
  assert.deepEqual(Lab.missingPaths(data.metrics, Lab.METRICS_PATHS), []);
  assert.deepEqual(Lab.missingPaths(data.examples, Lab.EXAMPLES_PATHS), []);
  assert.deepEqual(Lab.missingPaths(data.card, Lab.MODEL_CARD_PATHS), []);
  assert.deepEqual(Lab.missingPaths(data.manifest, Lab.MANIFEST_PATHS), []);
  delete data.metrics.test.evaluation.per_region[1].model.ci.weighted_recall;
  delete data.examples.examples[2].precomputed;
  assert.deepEqual(Lab.missingPaths(data.metrics, Lab.METRICS_PATHS), ['test.evaluation.per_region[].model.ci.weighted_recall.estimate']);
  assert.deepEqual(Lab.missingPaths(data.examples, Lab.EXAMPLES_PATHS), ['examples[].precomputed']);
  assert.deepEqual(Lab.missingPaths({ examples: [] }, ['examples[].region']), ['examples[].region']);
});

test('the script has no HTML-parsing sinks', () => {
  const source = fs.readFileSync(SOURCE, 'utf8');
  for (const sink of ['innerHTML', 'outerHTML', 'insertAdjacentHTML', 'document.write', 'eval(', 'new Function']) {
    assert.equal(source.includes(sink), false, sink);
  }
});

test('static release without an API: manifest first, then exactly the three release files; form disabled; names escaped', async () => {
  const data = fixtures(null, false);
  const { container, log, cleanup } = mount(routesFor(data));
  await cleanup.ready;
  assert.deepEqual(log.map((r) => r.url), ['data/manifest.json', 'data/releases/r1/metrics.json', 'data/releases/r1/examples.json', 'data/releases/r1/model-card.json']);
  assert.deepEqual(byTag(container, 'h2').map((h) => h.textContent), [
    'What this screens', 'Try it: curated examples and the live form', 'Live result',
    'Rules versus model on held-out test data', 'Errors and limitations', 'Code, method and reproducibility']);
  assert.equal(byRole(container, 'schema-warning').length, 0);
  assert.equal(byRole(container, 'preview-banner').length, 0);
  assert.equal(one(container, 'predict-submit').disabled, true);
  assert.match(one(container, 'api-note').textContent, /^API unavailable: no live prediction service/);
  assert.equal(one(container, 'api-note').getAttribute('data-state'), 'unavailable');
  byRole(container, 'example-load').forEach((button) => assert.equal(button.disabled, true));
  const cells = [];
  walk(one(container, 'examples-table'), (n) => { if (n.tagName === 'TH' || n.tagName === 'TD') cells.push(n.textContent); });
  assert.ok(cells.includes(EVIL), 'the markup-looking borrower name is shown literally');
  assert.ok(cells.includes('precomputed'));
  const provenanceText = one(container, 'label-provenance').textContent;
  assert.ok(provenanceText.includes(POOLED), 'the page states the pooled sentence, not one round of it');
  Object.keys(BY_ROUND).forEach((round) => {
    assert.ok(provenanceText.includes(round) && provenanceText.includes(BY_ROUND[round]), round);
  });
  assert.equal(provenanceText.includes('the founder decided every disagreement'), false,
    'that is false of a round whose disagreements were retained as unresolved');
  assert.ok(one(container, 'sample-sizes').textContent.includes('612 resolved cases'));
  assert.ok(one(container, 'unresolved-summary').textContent.includes('88 of 700'));
  assert.equal(one(container, 'verdict').textContent, 'Verdict: model differs from rules');
  const strata = [];
  walk(one(container, 'stratum-table'), (n) => { if (n.tagName === 'TD') strata.push(n.textContent); });
  assert.ok(strata.includes('1,234,567') && strata.includes('8,230.4'));
  assert.ok(one(container, 'errors-table').textContent.includes(EVIL));
  assert.ok(one(container, 'limitations').textContent.includes('Not a forecast.'));
  assert.ok(one(container, 'labels-agreement-table').textContent.includes('85.5%'));
  cleanup();
  assert.equal(container.children.length, 0);
});

test('a synthetic preview manifest shows the SYNTHETIC banner', async () => {
  const { container, cleanup } = mount(routesFor(fixtures(null, true)));
  await cleanup.ready;
  assert.match(one(container, 'preview-banner').textContent, /^SYNTHETIC PREVIEW/);
  cleanup();
});

test('an unpublished Lab says so instead of failing', async () => {
  const { container, log, cleanup } = mount({});
  await cleanup.ready;
  assert.deepEqual(log.map((r) => r.url), ['data/manifest.json']);
  assert.match(one(container, 'load-error').textContent, /No model release has been published yet/);
  cleanup();
});

test('a renamed key is surfaced as a visible schema warning', async () => {
  const data = fixtures(null, false);
  delete data.metrics.test.verdict;
  const { container, cleanup } = mount(routesFor(data));
  await cleanup.ready;
  assert.match(one(container, 'schema-warning').textContent, /metrics test\.verdict\.text/);
  cleanup();
});

test('readyz not 200 keeps the form disabled and names the status', async () => {
  const api = 'https://ucc-ml-api-abc-uc.a.run.app';
  const routes = Object.assign(routesFor(fixtures(api, false)), { ['GET ' + api + '/readyz']: respond(503, { status: 'loading' }) });
  const { container, cleanup } = mount(routes);
  await cleanup.ready;
  assert.equal(one(container, 'predict-submit').disabled, true);
  assert.match(one(container, 'api-note').textContent, /readyz answered HTTP 503/);
  cleanup();
});

test('a ready API enables the form; a prediction renders as text with its release and timings', async () => {
  const api = 'https://ucc-ml-api-abc-uc.a.run.app';
  const prediction = { request_id: 'req-1', release_id: 'r1', input_hash: 'f'.repeat(64), score: 0.83, score_type: 'calibrated_probability',
    decision: 'suggest_relevant', threshold: 0.62, baseline_qualifies: true, baseline_route: 'lender', elapsed_ms: 3.21,
    top_feature_contributions: [{ block: 'borrower_word', feature: '<b>excavating</b>', contribution: 1.25 }] };
  const routes = Object.assign(routesFor(fixtures(api, false)), {
    ['GET ' + api + '/readyz']: respond(200, { status: 'ready', release_id: 'r1' }),
    ['POST ' + api + '/v1/predict']: respond(200, prediction)
  });
  const { container, log, cleanup } = mount(routes);
  await cleanup.ready;
  assert.equal(one(container, 'predict-submit').disabled, false);
  assert.match(one(container, 'api-note').textContent, /^Live API ready \(release r1\)/);
  const load = byRole(container, 'example-load')[1];
  assert.equal(load.disabled, false);
  load.listeners.click[0]();
  assert.equal(one(container, 'borrower-input').value, 'SMITH EXCAVATING LLC');
  assert.equal(one(container, 'lenders-input').value, 'WAGNER EQUIPMENT CO');
  one(container, 'predict-form').listeners.submit[0]({ preventDefault() {} });
  await flush();
  const post = log.find((r) => r.method === 'POST');
  assert.deepEqual(JSON.parse(post.body), { borrower_name: 'SMITH EXCAVATING LLC', lender_names: ['WAGNER EQUIPMENT CO'], region: 'CO' });
  assert.equal(post.headers['content-type'], 'application/json');
  const text = one(container, 'prediction').textContent;
  assert.ok(text.includes('Accepted (lender criterion)'));
  assert.ok(text.includes('0.830 (calibrated probability)'));
  assert.ok(text.includes('Suggest relevant at threshold 0.6200'));
  assert.ok(text.includes('release r1'));
  assert.ok(text.includes('3.2 ms in the model'));
  assert.ok(one(container, 'contributions-table').textContent.includes('<b>excavating</b>'));
  assert.ok(one(container, 'contributions-table').textContent.includes('+1.250'));
  cleanup();
});

test('a different API release is named, and a 422 shows the error without a stale result', async () => {
  const api = 'https://ucc-ml-api-abc-uc.a.run.app';
  const routes = Object.assign(routesFor(fixtures(api, false)), {
    ['GET ' + api + '/readyz']: respond(200, { status: 'ready', release_id: 'r2' }),
    ['POST ' + api + '/v1/predict']: respond(422, { detail: [{ msg: 'Input should be \'CO\' or \'CT\'' }] })
  });
  const { container, log, cleanup } = mount(routes);
  await cleanup.ready;
  assert.match(one(container, 'api-note').textContent, /serves release r2 while this report describes release r1/);
  one(container, 'borrower-input').value = 'Example Excavation LLC';
  one(container, 'predict-form').listeners.submit[0]({ preventDefault() {} });
  await flush();
  assert.match(one(container, 'predict-error').textContent, /The API rejected the input: Input should be/);
  assert.equal(byRole(container, 'prediction').length, 0);
  one(container, 'lenders-input').value = Array(21).fill('L').join('\n');
  const before = log.length;
  one(container, 'predict-form').listeners.submit[0]({ preventDefault() {} });
  await flush();
  assert.equal(log.length, before, 'an over-bound input never reaches the network');
  assert.match(one(container, 'predict-error').textContent, /at most 20/);
  cleanup();
});

test('cleanup removes the Lab, its listeners, and ignores responses that arrive later', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const data = fixtures(null, false);
  const routes = Object.assign(routesFor(data), { 'GET data/manifest.json': () => gate.then(() => respond(200, data.manifest)) });
  const { container, cleanup } = mount(routes);
  assert.equal(container.children.length, 1);
  cleanup();
  assert.equal(container.children.length, 0);
  release();
  await cleanup.ready;
  await flush();
  assert.equal(container.children.length, 0, 'nothing is rendered after cleanup');
  cleanup();
});

test('cleanup after an enabled form detaches the submit and example listeners', async () => {
  const api = 'https://ucc-ml-api-abc-uc.a.run.app';
  const routes = Object.assign(routesFor(fixtures(api, false)), { ['GET ' + api + '/readyz']: respond(200, { status: 'ready', release_id: 'r1' }) });
  const { container, cleanup } = mount(routes);
  await cleanup.ready;
  const form = one(container, 'predict-form');
  const buttons = byRole(container, 'example-load');
  assert.equal(form.listeners.submit.length, 1);
  cleanup();
  assert.equal(form.listeners.submit.length, 0);
  buttons.forEach((button) => assert.equal(button.listeners.click.length, 0));
});
