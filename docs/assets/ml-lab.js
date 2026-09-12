/*
 * UCC ML Lab -- docs/assets/ml-lab.js (plan C Task 5)
 *
 *   var cleanup = window.mountMLLab(container, config);   // cleanup() removes everything it added
 *
 * config: { dataBase: '../data/ml/', apiUrl: <string|null, optional override of the manifest>,
 *           repoUrl: 'https://github.com/drkyoyodating/UCC-Demo', mapUrl: '../' }
 *
 * The module fetches manifest.json first, then ONLY the current release's three files, renders the
 * six sections of the Codex plan section 10 in that order, and checks the live API last. The static
 * report is complete without the API. Every string that reaches the page is written with
 * textContent, never parsed as HTML, so borrower and lender names cannot inject markup.
 * The same file loads in node (module.exports) for ml/tests/js/test_ml_lab.cjs.
 */
(function (root, factory) {
  'use strict';
  var api = factory(root);
  if (typeof module === 'object' && module && module.exports) {
    module.exports = api;
  }
  if (root && typeof root.document !== 'undefined') {
    root.MLLab = api;
    root.mountMLLab = api.mountMLLab;
  }
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';

  // The disclosure of a labels file whose rounds all share one arrangement. It is a FALLBACK for a
  // document that states none -- never this page's claim about a release. A labels file may hold
  // rounds labelled under different arrangements, the real one does, and then the only true
  // sentence is the pooled one the document itself carries. Every render below reads the document.
  var LABEL_PROVENANCE = 'model-labelled, founder-adjudicated';
  var DASH = '—';
  var MINUS = '−';
  var EN_DASH = '–';
  var DEFAULT_REPO_URL = 'https://github.com/drkyoyodating/UCC-Demo';
  var READY_TIMEOUT_MS = 20000;
  var PREDICT_TIMEOUT_MS = 30000;
  var MAX_NAME_CHARS = 300;
  var MAX_LENDERS = 20;

  // The document paths this page reads. ml/tests/test_lab_page.py parses these four arrays out of
  // this file and checks the exported JSON against them, so a renamed key fails a test, not the page.
  var MANIFEST_PATHS = [
    'schema_version', 'current', 'releases[].release_id', 'releases[].metrics_url',
    'releases[].examples_url', 'releases[].model_card_url', 'releases[].api_url'
  ];
  var METRICS_PATHS = [
    'release_id', 'label_provenance',
    'test.split', 'test.evaluated_at', 'test.forced',
    'test.frozen.threshold', 'test.frozen.status', 'test.frozen.score_type',
    'test.evaluation.n', 'test.evaluation.positives', 'test.evaluation.negatives',
    'test.evaluation.model.tp', 'test.evaluation.model.fp', 'test.evaluation.model.fn', 'test.evaluation.model.tn',
    'test.evaluation.model.ci.weighted_precision.estimate', 'test.evaluation.model.ci.weighted_precision.lower',
    'test.evaluation.model.ci.weighted_precision.upper', 'test.evaluation.model.ci.weighted_recall.estimate',
    'test.evaluation.model.ci.weighted_recall.lower', 'test.evaluation.model.ci.weighted_recall.upper',
    'test.evaluation.model.ci.weighted_f1.estimate',
    'test.evaluation.rules.tp', 'test.evaluation.rules.fp', 'test.evaluation.rules.fn', 'test.evaluation.rules.tn',
    'test.evaluation.rules.ci.weighted_precision.estimate', 'test.evaluation.rules.ci.weighted_precision.lower',
    'test.evaluation.rules.ci.weighted_precision.upper', 'test.evaluation.rules.ci.weighted_recall.estimate',
    'test.evaluation.rules.ci.weighted_recall.lower', 'test.evaluation.rules.ci.weighted_recall.upper',
    'test.evaluation.rules.ci.weighted_f1.estimate',
    'test.evaluation.delta_model_minus_rules.delta_weighted_precision.estimate',
    'test.evaluation.delta_model_minus_rules.delta_weighted_precision.lower',
    'test.evaluation.delta_model_minus_rules.delta_weighted_precision.upper',
    'test.evaluation.delta_model_minus_rules.delta_weighted_recall.estimate',
    'test.evaluation.delta_model_minus_rules.delta_weighted_recall.lower',
    'test.evaluation.delta_model_minus_rules.delta_weighted_recall.upper',
    'test.evaluation.per_region[].region', 'test.evaluation.per_region[].n',
    'test.evaluation.per_region[].model.ci.weighted_precision.estimate',
    'test.evaluation.per_region[].model.ci.weighted_recall.estimate',
    'test.evaluation.per_region[].rules.ci.weighted_precision.estimate',
    'test.evaluation.per_region[].rules.ci.weighted_recall.estimate',
    'test.evaluation.per_stratum[].stratum', 'test.evaluation.per_stratum[].n',
    'test.evaluation.per_stratum[].model.ci.weighted_precision.estimate',
    'test.evaluation.per_stratum[].model.ci.weighted_recall.estimate',
    'test.evaluation.per_stratum[].rules.ci.weighted_precision.estimate',
    'test.evaluation.per_stratum[].rules.ci.weighted_recall.estimate',
    'test.evaluation.weights[].stratum', 'test.evaluation.weights[].N_h', 'test.evaluation.weights[].n_h',
    'test.evaluation.weights[].w_h',
    'test.evaluation.bootstrap.n_resamples', 'test.evaluation.bootstrap.level',
    'test.unresolved[].region', 'test.unresolved[].n_labelled', 'test.unresolved[].n_unresolved',
    'test.unresolved[].share_unresolved',
    'test.verdict.text',
    'test.labels.disclosure', 'test.labels.counts_by_status', 'test.labels.counts_by_round',
    'test.labels.pass_agreement', 'test.labels.founder_audit', 'test.labels.repeat_consistency'
  ];
  var EXAMPLES_PATHS = [
    'release_id', 'curated', 'precomputed', 'label_provenance',
    'examples[].region', 'examples[].borrower_name', 'examples[].lender_names',
    'examples[].baseline_qualifies', 'examples[].baseline_route', 'examples[].score',
    'examples[].decision', 'examples[].label', 'examples[].curated', 'examples[].precomputed'
  ];
  var MODEL_CARD_PATHS = [
    'release_id', 'label_provenance', 'sections[].heading', 'sections[].text'
  ];

  // ---- document helpers -------------------------------------------------------------------------

  function hasOwn(object, key) {
    return Object.prototype.hasOwnProperty.call(object, key);
  }

  function hasPath(value, parts) {
    if (parts.length === 0) {
      return true;
    }
    var head = parts[0];
    var rest = parts.slice(1);
    var isList = head.slice(-2) === '[]';
    var key = isList ? head.slice(0, -2) : head;
    if (value === null || typeof value !== 'object' || Array.isArray(value) || !hasOwn(value, key)) {
      return false;
    }
    var next = value[key];
    if (!isList) {
      return rest.length === 0 ? true : hasPath(next, rest);
    }
    if (!Array.isArray(next) || next.length === 0) {
      return false;
    }
    return next.every(function (item) { return hasPath(item, rest); });
  }

  function missingPaths(doc, paths) {
    return paths.filter(function (path) { return !hasPath(doc, path.split('.')); });
  }

  function get(object, path) {
    var parts = path.split('.');
    var current = object;
    for (var i = 0; i < parts.length; i += 1) {
      if (current === null || typeof current !== 'object') {
        return null;
      }
      current = current[parts[i]];
    }
    return current === undefined ? null : current;
  }

  // ---- formatting -------------------------------------------------------------------------------
  // Years, ids, hashes and thresholds are never thousands-separated; counts are. One formatter per
  // column name, so the same number is written the same way in every table.

  var COLUMN_KINDS = {
    loan_year: 'year', year: 'year',
    threshold: 'threshold',
    score: 'score', ece: 'score',
    contribution: 'contribution',
    weighted_precision: 'rate', weighted_recall: 'rate', weighted_f1: 'rate',
    precision: 'rate', recall: 'rate', f1: 'rate', rate: 'rate',
    agreement_rate: 'rate', share_unresolved: 'rate', level: 'rate',
    n: 'count', n_h: 'count', N_h: 'count', tp: 'count', fp: 'count', fn: 'count', tn: 'count',
    positives: 'count', negatives: 'count', predicted_positives: 'count',
    n_labelled: 'count', n_unresolved: 'count', rows: 'count', agreed: 'count', consistent: 'count',
    n_audited: 'count', n_confirmed: 'count', n_overturned: 'count', n_resamples: 'count', count: 'count',
    w_h: 'weight', weight: 'weight',
    elapsed_ms: 'ms', round_trip_ms: 'ms',
    baseline_qualifies: 'rules',
    decision: 'decision',
    lender_names: 'list',
    input_hash: 'hash'
  };
  var NUMERIC_KINDS = { count: true, rate: true, weight: true, score: true, threshold: true, contribution: true, ms: true };

  function columnKind(column) {
    var name = String(column);
    if (hasOwn(COLUMN_KINDS, name)) {
      return COLUMN_KINDS[name];
    }
    if (/sha256|_hash$/.test(name)) {
      return 'hash';
    }
    if (/(^|_)id$/.test(name)) {
      return 'id';
    }
    if (/year/.test(name)) {
      return 'year';
    }
    return 'text';
  }

  function isMissing(value) {
    return value === null || value === undefined || value === '' ||
      (typeof value === 'number' && !isFinite(value));
  }

  function asNumber(value) {
    var number = typeof value === 'number' ? value : Number(value);
    return isFinite(number) ? number : null;
  }

  function groupDigits(text) {
    return text.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function grouped(number, digits) {
    var text = Math.abs(number).toFixed(digits);
    var dot = text.indexOf('.');
    var whole = dot === -1 ? text : text.slice(0, dot);
    var fraction = dot === -1 ? '' : text.slice(dot);
    var sign = number < 0 && Number(text) !== 0 ? MINUS : '';
    return sign + groupDigits(whole) + fraction;
  }

  function decisionLabel(decision) {
    if (decision === 'suggest_relevant') {
      return 'Suggest relevant';
    }
    if (decision === 'review_needed') {
      return 'Review needed';
    }
    return isMissing(decision) ? DASH : String(decision);
  }

  function formatCell(column, value) {
    var kind = columnKind(column);
    var number;
    if (kind === 'list') {
      return Array.isArray(value) && value.length > 0 ? value.map(String).join('; ') : DASH;
    }
    if (isMissing(value)) {
      return DASH;
    }
    switch (kind) {
      case 'year':
        number = asNumber(value);
        return number === null ? String(value) : String(Math.trunc(number));
      case 'id':
      case 'hash':
        return String(value);
      case 'threshold':
        number = asNumber(value);
        return number === null ? String(value) : number.toFixed(4);
      case 'score':
        number = asNumber(value);
        return number === null ? String(value) : number.toFixed(3);
      case 'contribution':
        number = asNumber(value);
        return number === null ? String(value) : (number < 0 ? MINUS : '+') + Math.abs(number).toFixed(3);
      case 'count':
        number = asNumber(value);
        return number === null ? String(value) : grouped(Math.round(number), 0);
      case 'rate':
        number = asNumber(value);
        return number === null ? String(value) : (number * 100).toFixed(1) + '%';
      case 'weight':
        number = asNumber(value);
        return number === null ? String(value) : grouped(number, 1);
      case 'ms':
        number = asNumber(value);
        return number === null ? String(value) : grouped(number, 1) + ' ms';
      case 'rules':
        if (value === true) {
          return 'accepted';
        }
        return value === false ? 'rejected' : String(value);
      case 'decision':
        return decisionLabel(value);
      default:
        return String(value);
    }
  }

  function routeLabel(route) {
    var labels = { lender: 'lender criterion', borrower: 'borrower criterion', both: 'both criteria', neither: 'no criterion' };
    if (hasOwn(labels, route)) {
      return labels[route];
    }
    return isMissing(route) ? DASH : String(route);
  }

  function rulesOutcome(qualifies, route) {
    if (qualifies === true) {
      return 'Accepted (' + routeLabel(route) + ')';
    }
    return qualifies === false ? 'Rejected' : DASH;
  }

  function levelText(level) {
    var number = asNumber(level);
    return (number === null ? 95 : Math.round(number * 100)) + '% CI';
  }

  function formatEstimate(block, level) {
    if (!block || isMissing(block.estimate)) {
      return DASH;
    }
    var head = formatCell('rate', block.estimate);
    if (isMissing(block.lower) || isMissing(block.upper)) {
      return head + ' (no interval)';
    }
    return head + ' (' + levelText(level) + ' ' + (block.lower * 100).toFixed(1) + EN_DASH +
      (block.upper * 100).toFixed(1) + '%)';
  }

  function signedPoints(value) {
    var points = value * 100;
    var text = Math.abs(points).toFixed(1);
    if (Number(text) === 0) {
      return '0.0';
    }
    return (points < 0 ? MINUS : '+') + text;
  }

  function formatDelta(block, level) {
    if (!block || isMissing(block.estimate)) {
      return DASH;
    }
    var head = signedPoints(block.estimate) + ' pts';
    if (isMissing(block.lower) || isMissing(block.upper)) {
      return head + ' (no interval)';
    }
    return head + ' (' + levelText(level) + ' ' + signedPoints(block.lower) + ' to ' +
      signedPoints(block.upper) + ' pts)';
  }

  // ---- data rules -------------------------------------------------------------------------------

  function cleanApiUrl(value) {
    if (typeof value !== 'string') {
      return null;
    }
    var url = value.trim().replace(/\/+$/, '');
    if (/^https:\/\/[^\s/?#]+$/.test(url)) {
      return url;
    }
    if (/^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(url)) {
      return url;
    }
    return null;
  }

  // An explicit config.apiUrl (including null) wins over the manifest; otherwise the release's
  // api_url is used. Anything that is not https (or a loopback http URL) is treated as no API.
  function resolveApiUrl(config, release) {
    if (config && hasOwn(config, 'apiUrl')) {
      return cleanApiUrl(config.apiUrl);
    }
    return cleanApiUrl(release ? release.api_url : null);
  }

  // Curated examples whose resolved label disagrees with the model decision. They illustrate
  // failure modes; they are not an error rate (the list is curated).
  function selectErrorExamples(examples) {
    return (Array.isArray(examples) ? examples : []).filter(function (example) {
      if (!example || (example.label !== 'RELEVANT' && example.label !== 'NOT_RELEVANT')) {
        return false;
      }
      if (example.decision !== 'suggest_relevant' && example.decision !== 'review_needed') {
        return false;
      }
      return (example.decision === 'suggest_relevant') !== (example.label === 'RELEVANT');
    });
  }

  function currentRelease(manifest) {
    var releases = manifest && Array.isArray(manifest.releases) ? manifest.releases : [];
    for (var i = 0; i < releases.length; i += 1) {
      if (releases[i] && releases[i].release_id === manifest.current) {
        return releases[i];
      }
    }
    return null;
  }

  function isRelativeDataUrl(url) {
    return typeof url === 'string' && url !== '' && !/^[a-z][a-z0-9+.-]*:/i.test(url) &&
      url.charAt(0) !== '/' && url.split('/').indexOf('..') === -1;
  }

  // Returns {body: {...}} or {error: '...'}; the API accepts one case per call.
  function buildPredictBody(borrowerText, lendersText, region) {
    var name = String(borrowerText === null || borrowerText === undefined ? '' : borrowerText).trim();
    var lenders = String(lendersText === null || lendersText === undefined ? '' : lendersText)
      .split(/\r?\n/)
      .map(function (line) { return line.trim(); })
      .filter(function (line) { return line !== ''; });
    if (name === '') {
      return { error: 'Enter a borrower name.' };
    }
    if (name.length > MAX_NAME_CHARS) {
      return { error: 'The borrower name is longer than ' + MAX_NAME_CHARS + ' characters. The service sends such names to review instead of truncating them.' };
    }
    if (lenders.length > MAX_LENDERS) {
      return { error: 'Enter at most ' + MAX_LENDERS + ' lender names (one per line).' };
    }
    for (var i = 0; i < lenders.length; i += 1) {
      if (lenders[i].length > MAX_NAME_CHARS) {
        return { error: 'Lender name ' + (i + 1) + ' is longer than ' + MAX_NAME_CHARS + ' characters.' };
      }
    }
    return { body: { borrower_name: name, lender_names: lenders, region: region === 'CT' ? 'CT' : 'CO' } };
  }

  function readReport(metrics) {
    var test = (metrics && metrics.test) || {};
    var evaluation = test.evaluation || {};
    var bootstrap = evaluation.bootstrap || {};
    var unresolved = Array.isArray(test.unresolved) ? test.unresolved : [];
    var total = null;
    unresolved.forEach(function (row) {
      if (row && row.region === 'total') {
        total = row;
      }
    });
    return {
      releaseId: get(metrics, 'release_id'),
      provenance: get(metrics, 'label_provenance'),
      split: test.split || null,
      evaluatedAt: test.evaluated_at || null,
      forced: test.forced === true,
      threshold: get(test, 'frozen.threshold'),
      thresholdStatus: get(test, 'frozen.status'),
      scoreType: get(test, 'frozen.score_type'),
      n: get(evaluation, 'n'),
      positives: get(evaluation, 'positives'),
      negatives: get(evaluation, 'negatives'),
      level: get(bootstrap, 'level'),
      resamples: get(bootstrap, 'n_resamples'),
      model: evaluation.model || {},
      rules: evaluation.rules || {},
      delta: evaluation.delta_model_minus_rules || {},
      perRegion: Array.isArray(evaluation.per_region) ? evaluation.per_region : [],
      perStratum: Array.isArray(evaluation.per_stratum) ? evaluation.per_stratum : [],
      weights: Array.isArray(evaluation.weights) ? evaluation.weights : [],
      unresolved: unresolved.filter(function (row) { return row && row.region !== 'total'; }),
      unresolvedTotal: total,
      labels: test.labels || {},
      verdict: get(test, 'verdict.text')
    };
  }

  // ---- DOM (textContent and setAttribute only) ----------------------------------------------------

  function el(doc, tag, attrs, text) {
    var node = doc.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (name) {
        var value = attrs[name];
        if (value !== null && value !== undefined && value !== false) {
          node.setAttribute(name, value === true ? '' : String(value));
        }
      });
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function add(parent, child) {
    parent.appendChild(child);
    return child;
  }

  function para(doc, parent, text, attrs) {
    return add(parent, el(doc, 'p', attrs || null, text));
  }

  function section(doc, parent, id, title) {
    var node = add(parent, el(doc, 'section', { 'class': 'mll-section', 'aria-labelledby': id, 'data-role': 'section' }));
    add(node, el(doc, 'h2', { id: id }, title));
    return node;
  }

  function numericClass(column) {
    return NUMERIC_KINDS[columnKind(column || '')] ? 'num' : null;
  }

  // spec: {label, role, columns: [{label, key, format, text(row), cell(doc, node, row)}], rows}
  function dataTable(doc, spec) {
    var wrap = el(doc, 'div', { 'class': 'mll-scroll', tabindex: '0', role: 'region', 'aria-label': spec.label, 'data-role': spec.role || null });
    var table = add(wrap, el(doc, 'table', { 'class': 'mll-table' }));
    add(table, el(doc, 'caption', null, spec.label));
    var headRow = add(add(table, el(doc, 'thead')), el(doc, 'tr'));
    spec.columns.forEach(function (column) {
      add(headRow, el(doc, 'th', { scope: 'col', 'class': numericClass(column.format || column.key) }, column.label));
    });
    var body = add(table, el(doc, 'tbody'));
    spec.rows.forEach(function (row) {
      var tr = add(body, el(doc, 'tr'));
      spec.columns.forEach(function (column, index) {
        var attrs = index === 0 ? { scope: 'row' } : { 'class': numericClass(column.format || column.key) };
        var cell = add(tr, el(doc, index === 0 ? 'th' : 'td', attrs));
        if (column.cell) {
          column.cell(doc, cell, row);
        } else if (column.text) {
          cell.textContent = column.text(row);
        } else {
          cell.textContent = formatCell(column.format || column.key, row[column.key]);
        }
      });
    });
    return wrap;
  }

  function keyedRows(object, keyName) {
    var source = object && typeof object === 'object' ? object : {};
    return Object.keys(source).sort().map(function (key) {
      var value = source[key];
      var row = value && typeof value === 'object' ? Object.assign({}, value) : { n: value };
      row[keyName] = key;
      return row;
    });
  }

  // ---- sections (Codex plan section 10, in order) -------------------------------------------------

  function renderHeader(doc, parent, manifest, releaseId, provenance) {
    var header = add(parent, el(doc, 'header', { 'class': 'mll-header' }));
    if (manifest && manifest.preview === true) {
      add(header, el(doc, 'p', { 'class': 'mll-banner', role: 'note', 'data-role': 'preview-banner' },
        'SYNTHETIC PREVIEW — fictitious names and numbers from the test fixtures. Not a model result; never publish this build.'));
    }
    add(header, el(doc, 'h1', null, 'UCC ML Lab'));
    add(header, el(doc, 'p', { 'class': 'mll-sub', 'data-role': 'release-line' },
      'Release ' + formatCell('release_id', releaseId) + ' · labels ' + (provenance || LABEL_PROVENANCE)));
  }

  function renderIntro(doc, parent, config, view) {
    var node = section(doc, parent, 'mll-intro', 'What this screens');
    para(doc, node, 'The Lab reads one UCC filing at a time: the borrower name, the lender names on the same filing and the state. It asks whether those names alone show that the borrower works in the heavy-construction-equipment finance market.');
    para(doc, node, 'The model is a review queue. It suggests cases the frozen rules rejected and gives a second opinion on cases they accepted. It never replaces the rules and never changes the map or its counts.');
    para(doc, node, 'A score is evidence of relevance under the written screening policy. It does not establish the collateral, equipment ownership, loan amount, active debt or creditworthiness.');
    // Rendered from the document, never asserted. The sentence this replaced also claimed "the
    // founder decided every disagreement and audited a sample of agreements", which is false of a
    // round whose disagreements were retained as unresolved -- 2,880 of the real file's 3,120 rows.
    var provenance = add(node, el(doc, 'div', { 'class': 'mll-provenance', 'data-role': 'label-provenance' }));
    var provenanceLine = add(provenance, el(doc, 'p'));
    add(provenanceLine, el(doc, 'strong', null, 'Labels: '));
    add(provenanceLine, el(doc, 'span', null, (view && view.provenance) || LABEL_PROVENANCE));
    add(provenanceLine, el(doc, 'span', null, '. Two independent blind Claude passes labelled each case from its names and city/state only, with no lookups.'));
    var introByRound = view && view.labels && view.labels.disclosure_by_round;
    if (introByRound && typeof introByRound === 'object') {
      para(doc, provenance, 'The rounds were not labelled under one arrangement, so each is stated separately:', { 'class': 'mll-note' });
      var introRounds = add(provenance, el(doc, 'ul', { 'class': 'mll-list', 'data-role': 'label-provenance-rounds' }));
      Object.keys(introByRound).sort().forEach(function (round) {
        add(introRounds, el(doc, 'li', null, round + ': ' + introByRound[round]));
      });
    }
    var back = add(node, el(doc, 'p', { 'class': 'mll-back' }));
    add(back, el(doc, 'a', { href: config.mapUrl }, 'Back to the map'));
  }

  function renderTry(doc, parent, examplesDoc, state) {
    var node = section(doc, parent, 'mll-try', 'Try it: curated examples and the live form');
    var examples = examplesDoc && Array.isArray(examplesDoc.examples) ? examplesDoc.examples : [];
    add(node, el(doc, 'h3', null, 'Curated examples'));
    para(doc, node, 'Curated and precomputed with release ' + formatCell('release_id', examplesDoc && examplesDoc.release_id) +
      '. A written rule chose them to explain the model, so they are never a benchmark: the measured rates are in the comparison section below.',
    { 'class': 'mll-note', 'data-role': 'examples-note' });
    if (examples.length === 0) {
      para(doc, node, 'This release publishes no curated examples.');
    } else {
      add(node, dataTable(doc, {
        label: 'Curated, precomputed examples', role: 'examples-table', rows: examples,
        columns: [
          { label: 'Borrower', key: 'borrower_name', format: 'text' },
          { label: 'Lenders', key: 'lender_names' },
          { label: 'State', key: 'region', format: 'text' },
          { label: 'Rules', text: function (row) { return rulesOutcome(row.baseline_qualifies, row.baseline_route); } },
          { label: 'Score', key: 'score' },
          { label: 'Decision', key: 'decision' },
          { label: 'Label', key: 'label', format: 'text' },
          { label: 'Source', text: function (row) { return row.precomputed === true ? 'precomputed' : DASH; } },
          { label: 'Try it', cell: function (d, cell, row) {
            var button = add(cell, el(d, 'button', { type: 'button', 'class': 'mll-small', 'data-role': 'example-load' }, 'Load into form'));
            button.disabled = true;
            state.exampleButtons.push({ button: button, example: row });
          } }
        ]
      }));
    }
    renderForm(doc, node, state);
  }

  function field(doc, parent, id, label) {
    var wrap = add(parent, el(doc, 'div', { 'class': 'mll-field' }));
    add(wrap, el(doc, 'label', { 'for': id }, label));
    return wrap;
  }

  function renderForm(doc, parent, state) {
    add(parent, el(doc, 'h3', null, 'Live form'));
    var form = add(parent, el(doc, 'form', { 'class': 'mll-form', 'data-role': 'predict-form', novalidate: true }));
    var fieldset = add(form, el(doc, 'fieldset', { 'data-role': 'predict-fieldset' }));
    add(fieldset, el(doc, 'legend', null, 'Score one borrower with the live API'));
    var name = add(field(doc, fieldset, 'mll-borrower', 'Borrower name'),
      el(doc, 'input', { id: 'mll-borrower', type: 'text', maxlength: MAX_NAME_CHARS, autocomplete: 'off', spellcheck: 'false', 'data-role': 'borrower-input' }));
    var lenders = add(field(doc, fieldset, 'mll-lenders', 'Lender names, one per line (up to 20)'),
      el(doc, 'textarea', { id: 'mll-lenders', rows: 3, spellcheck: 'false', 'data-role': 'lenders-input' }));
    var region = add(field(doc, fieldset, 'mll-region', 'State'), el(doc, 'select', { id: 'mll-region', 'data-role': 'region-input' }));
    add(region, el(doc, 'option', { value: 'CO' }, 'Colorado (CO)'));
    add(region, el(doc, 'option', { value: 'CT' }, 'Connecticut (CT)'));
    region.value = 'CO';
    var submit = add(fieldset, el(doc, 'button', { type: 'submit', 'data-role': 'predict-submit' }, 'Score with the live API'));
    var note = add(form, el(doc, 'p', { 'class': 'mll-note', role: 'status', 'aria-live': 'polite', 'data-role': 'api-note' }, 'Checking the live API…'));
    fieldset.disabled = true;
    submit.disabled = true;
    state.form = { form: form, fieldset: fieldset, name: name, lenders: lenders, region: region, submit: submit, note: note };
  }

  function renderResultSection(doc, parent, state) {
    var node = section(doc, parent, 'mll-result', 'Live result');
    state.result = add(node, el(doc, 'div', { 'aria-live': 'polite', 'data-role': 'result' }));
    para(doc, state.result, 'No live prediction yet. Results appear here only when the API answers; a curated example is never shown as a live result.', { 'class': 'mll-note' });
  }

  function showPrediction(doc, state, prediction, roundTripMs) {
    var box = state.result;
    box.textContent = '';
    var list = add(box, el(doc, 'dl', { 'class': 'mll-result', 'data-role': 'prediction' }));
    function row(term, value) {
      add(list, el(doc, 'dt', null, term));
      add(list, el(doc, 'dd', null, value));
    }
    var sameRelease = prediction.release_id === state.releaseId;
    row('Frozen rules', rulesOutcome(prediction.baseline_qualifies, prediction.baseline_route));
    row('Model score', formatCell('score', prediction.score) +
      (prediction.score_type === 'calibrated_probability' ? ' (calibrated probability)' : ' (a score, not a probability)'));
    row('Review decision', decisionLabel(prediction.decision) + ' at threshold ' + formatCell('threshold', prediction.threshold));
    row('Model version', 'release ' + formatCell('release_id', prediction.release_id) +
      (sameRelease ? '' : ' (this report describes release ' + formatCell('release_id', state.releaseId) + ')'));
    row('Elapsed', formatCell('elapsed_ms', prediction.elapsed_ms) + ' in the model, ' + formatCell('round_trip_ms', roundTripMs) + ' round trip');
    row('Request', formatCell('request_id', prediction.request_id));
    var contributions = Array.isArray(prediction.top_feature_contributions) ? prediction.top_feature_contributions : [];
    if (contributions.length > 0) {
      add(box, dataTable(doc, {
        label: 'Largest feature contributions', role: 'contributions-table', rows: contributions,
        columns: [
          { label: 'Feature', key: 'feature', format: 'text' },
          { label: 'Block', key: 'block', format: 'text' },
          { label: 'Contribution', key: 'contribution' }
        ]
      }));
    }
    para(doc, box, 'Linear feature contributions describe the model’s calculation, not independent evidence about the business.', { 'class': 'mll-note' });
  }

  function showPredictError(doc, state, message) {
    state.result.textContent = '';
    para(doc, state.result, message, { 'class': 'mll-error', role: 'alert', 'data-role': 'predict-error' });
  }

  function estimateColumn(label, metric, level) {
    return { label: label, text: function (row) { return formatEstimate(get(row.block, 'ci.' + metric), level); } };
  }

  function renderComparison(doc, parent, view) {
    var node = section(doc, parent, 'mll-compare', 'Rules versus model on held-out test data');
    var level = view.level;
    para(doc, node, 'Measured once on the ' + (view.split || 'test') + ' split: ' + formatCell('n', view.n) + ' resolved cases (' +
      formatCell('positives', view.positives) + ' relevant, ' + formatCell('negatives', view.negatives) + ' not relevant), evaluated ' +
      (view.evaluatedAt || DASH) + '. Rates are design-weighted (w_h = N_h / n_h); intervals come from a ' +
      formatCell('n_resamples', view.resamples) + '-resample stratified group bootstrap.', { 'data-role': 'sample-sizes' });
    if (view.unresolvedTotal) {
      para(doc, node, 'Unresolved: ' + formatCell('n_unresolved', view.unresolvedTotal.n_unresolved) + ' of ' +
        formatCell('n_labelled', view.unresolvedTotal.n_labelled) + ' labelled test cases (' +
        formatCell('share_unresolved', view.unresolvedTotal.share_unresolved) +
        ') were INSUFFICIENT_EVIDENCE and are excluded from every rate on this page.', { 'data-role': 'unresolved-summary' });
    }
    para(doc, node, 'Threshold ' + formatCell('threshold', view.threshold) + ', chosen on validation before the test was scored; status ' +
      (view.thresholdStatus || DASH) + '; ' + (view.scoreType === 'calibrated_probability' ? 'scores are calibrated probabilities.' : 'scores are not calibrated probabilities.'));
    if (view.thresholdStatus === 'experimental') {
      para(doc, node, 'Experimental: no validation threshold met the pre-registered weighted-precision target, so the rules stay authoritative.', { 'class': 'mll-warn', 'data-role': 'experimental-note' });
    }
    if (view.forced) {
      para(doc, node, 'This test result was re-run after the first evaluation, so it is not a clean held-out measurement.', { 'class': 'mll-warn', 'data-role': 'forced-note' });
    }
    add(node, el(doc, 'p', { 'class': 'mll-verdict', 'data-role': 'verdict' }, 'Verdict: ' + (view.verdict || DASH)));
    add(node, dataTable(doc, {
      label: 'Rules and model on the same test cases', role: 'headline-table',
      rows: [{ name: 'Frozen rules', block: view.rules }, { name: 'Model', block: view.model }],
      columns: [
        { label: 'Classifier', text: function (row) { return row.name; } },
        estimateColumn('Weighted precision', 'weighted_precision', level),
        estimateColumn('Weighted recall', 'weighted_recall', level),
        estimateColumn('Weighted F1', 'weighted_f1', level),
        { label: 'TP', format: 'tp', text: function (row) { return formatCell('tp', row.block.tp); } },
        { label: 'FP', format: 'fp', text: function (row) { return formatCell('fp', row.block.fp); } },
        { label: 'FN', format: 'fn', text: function (row) { return formatCell('fn', row.block.fn); } },
        { label: 'TN', format: 'tn', text: function (row) { return formatCell('tn', row.block.tn); } }
      ]
    }));
    add(node, dataTable(doc, {
      label: 'Paired difference, model minus rules', role: 'delta-table',
      rows: [
        { name: 'Weighted precision', block: view.delta.delta_weighted_precision },
        { name: 'Weighted recall', block: view.delta.delta_weighted_recall }
      ],
      columns: [
        { label: 'Metric', text: function (row) { return row.name; } },
        { label: 'Model − rules', text: function (row) { return formatDelta(row.block, level); } }
      ]
    }));
    add(node, dataTable(doc, {
      label: 'By state', role: 'region-table', rows: view.perRegion,
      columns: [
        { label: 'State', key: 'region', format: 'text' },
        { label: 'n', key: 'n' },
        { label: 'Rules precision', text: function (row) { return formatEstimate(get(row, 'rules.ci.weighted_precision'), level); } },
        { label: 'Model precision', text: function (row) { return formatEstimate(get(row, 'model.ci.weighted_precision'), level); } },
        { label: 'Rules recall', text: function (row) { return formatEstimate(get(row, 'rules.ci.weighted_recall'), level); } },
        { label: 'Model recall', text: function (row) { return formatEstimate(get(row, 'model.ci.weighted_recall'), level); } }
      ]
    }));
    var weights = {};
    view.weights.forEach(function (row) {
      if (row && typeof row.stratum === 'string') {
        weights[row.stratum] = row;
      }
    });
    add(node, dataTable(doc, {
      label: 'By state and rule outcome (the sampling strata)', role: 'stratum-table', rows: view.perStratum,
      columns: [
        { label: 'Stratum', key: 'stratum', format: 'text' },
        { label: 'Population N_h', format: 'N_h', text: function (row) { return formatCell('N_h', get(weights[row.stratum], 'N_h')); } },
        { label: 'Labelled n_h', format: 'n_h', text: function (row) { return formatCell('n_h', get(weights[row.stratum], 'n_h')); } },
        { label: 'Weight w_h', format: 'w_h', text: function (row) { return formatCell('w_h', get(weights[row.stratum], 'w_h')); } },
        { label: 'Resolved n', key: 'n' },
        { label: 'Rules precision', text: function (row) { return formatEstimate(get(row, 'rules.ci.weighted_precision'), level); } },
        { label: 'Model precision', text: function (row) { return formatEstimate(get(row, 'model.ci.weighted_precision'), level); } },
        { label: 'Rules recall', text: function (row) { return formatEstimate(get(row, 'rules.ci.weighted_recall'), level); } },
        { label: 'Model recall', text: function (row) { return formatEstimate(get(row, 'model.ci.weighted_recall'), level); } }
      ]
    }));
    renderLabelStats(doc, node, view.labels);
  }

  var STATUS_MEANING = {
    model_agreed: 'both blind passes gave the same label',
    founder_confirmed: 'founder re-read an agreed label and kept it',
    founder_adjudicated: 'founder decided a disagreement or overturned an agreed label',
    blind_unresolved: 'the two passes disagreed and nobody adjudicated it; retained as INSUFFICIENT_EVIDENCE, never fitted or evaluated',
    blind_repeat: 'hidden repeat for consistency; never fitted or evaluated'
  };

  function renderLabelStats(doc, parent, labels) {
    add(parent, el(doc, 'h3', null, 'How the labels were made'));
    para(doc, parent, 'Disclosure: ' + (labels.disclosure || LABEL_PROVENANCE) + '. Nothing was labelled by looking anything up.', { 'data-role': 'label-disclosure' });
    var statsByRound = labels.disclosure_by_round;
    if (statsByRound && typeof statsByRound === 'object') {
      var statsRounds = add(parent, el(doc, 'ul', { 'class': 'mll-list', 'data-role': 'labels-round-disclosure' }));
      Object.keys(statsByRound).sort().forEach(function (round) {
        add(statsRounds, el(doc, 'li', null, round + ': ' + statsByRound[round]));
      });
    }
    add(parent, dataTable(doc, {
      label: 'Label rows by status', role: 'labels-status-table', rows: keyedRows(labels.counts_by_status, 'status'),
      columns: [
        { label: 'Status', key: 'status', format: 'text' },
        { label: 'Meaning', text: function (row) { return hasOwn(STATUS_MEANING, row.status) ? STATUS_MEANING[row.status] : DASH; } },
        { label: 'Rows', key: 'n' }
      ]
    }));
    add(parent, dataTable(doc, {
      label: 'Label rows by round', role: 'labels-round-table', rows: keyedRows(labels.counts_by_round, 'round'),
      columns: [{ label: 'Round', key: 'round', format: 'text' }, { label: 'Rows', key: 'n' }]
    }));
    add(parent, dataTable(doc, {
      label: 'Agreement between the two blind passes', role: 'labels-agreement-table', rows: keyedRows(labels.pass_agreement, 'round'),
      columns: [{ label: 'Round', key: 'round', format: 'text' }, { label: 'Cases', key: 'n' }, { label: 'Agreed', key: 'agreed' }, { label: 'Rate', key: 'rate' }]
    }));
    add(parent, dataTable(doc, {
      label: 'Founder audit of agreed labels', role: 'labels-audit-table', rows: keyedRows(labels.founder_audit, 'round'),
      columns: [
        { label: 'Round', key: 'round', format: 'text' }, { label: 'Audited', key: 'n_audited' },
        { label: 'Confirmed', key: 'n_confirmed' }, { label: 'Overturned', key: 'n_overturned' },
        { label: 'Agreement', key: 'agreement_rate' }
      ]
    }));
    add(parent, dataTable(doc, {
      label: 'Blind-repeat consistency', role: 'labels-repeat-table', rows: keyedRows(labels.repeat_consistency, 'pass'),
      columns: [{ label: 'Pass', key: 'pass', format: 'text' }, { label: 'Repeats', key: 'n' }, { label: 'Consistent', key: 'consistent' }, { label: 'Rate', key: 'rate' }]
    }));
  }

  function renderErrors(doc, parent, examplesDoc, card, view) {
    var node = section(doc, parent, 'mll-errors', 'Errors and limitations');
    var errors = selectErrorExamples(examplesDoc && examplesDoc.examples);
    para(doc, node, 'Error examples from the curated list, where the adjudicated label and the model decision disagree. They illustrate failure modes; they are not an error rate.', { 'class': 'mll-note' });
    if (errors.length === 0) {
      para(doc, node, 'None of this release’s curated examples is a model error.', { 'data-role': 'no-errors' });
    } else {
      add(node, dataTable(doc, {
        label: 'Curated error examples', role: 'errors-table', rows: errors,
        columns: [
          { label: 'Borrower', key: 'borrower_name', format: 'text' },
          { label: 'Lenders', key: 'lender_names' },
          { label: 'State', key: 'region', format: 'text' },
          { label: 'Label', key: 'label', format: 'text' },
          { label: 'Decision', key: 'decision' },
          { label: 'Score', key: 'score' },
          { label: 'Rules', text: function (row) { return rulesOutcome(row.baseline_qualifies, row.baseline_route); } }
        ]
      }));
    }
    add(node, el(doc, 'h3', null, 'Limitations'));
    var list = add(node, el(doc, 'ul', { 'class': 'mll-list', 'data-role': 'limitations' }));
    add(list, el(doc, 'li', null, 'Names only: Colorado’s collateral field is a short category list and Connecticut publishes none, so the model never sees collateral text.'));
    if (view.unresolvedTotal) {
      add(list, el(doc, 'li', null, 'Unresolved labels: ' + formatCell('share_unresolved', view.unresolvedTotal.share_unresolved) +
        ' of labelled test cases were INSUFFICIENT_EVIDENCE; every rate describes resolved cases only.'));
    }
    var sections = card && Array.isArray(card.sections) ? card.sections : [];
    sections.forEach(function (item) {
      if (item && typeof item.heading === 'string' && item.heading.trim().toLowerCase() === 'limitations') {
        String(item.text || '').split(/\r?\n/).forEach(function (line) {
          var text = line.replace(/^\s*[-*]\s+/, '').trim();
          if (text !== '') {
            add(list, el(doc, 'li', null, text));
          }
        });
      }
    });
  }

  function renderLinks(doc, parent, config, release) {
    var node = section(doc, parent, 'mll-links', 'Code, method and reproducibility');
    var list = add(node, el(doc, 'ul', { 'class': 'mll-list', 'data-role': 'links' }));
    var links = [
      ['Source code', config.repoUrl],
      ['Label policy', config.repoUrl + '/blob/main/ml/specs/label_policy_v1.md'],
      ['Evaluation protocol (pre-registered)', config.repoUrl + '/blob/main/ml/specs/evaluation_protocol_v1.md'],
      ['Deployment and rollback notes', config.repoUrl + '/blob/main/ml/DEPLOY.md']
    ];
    if (release && isRelativeDataUrl(release.model_card_url)) {
      links.push(['This release’s model card (JSON)', config.dataBase + release.model_card_url]);
    }
    links.forEach(function (link) {
      add(add(list, el(doc, 'li')), el(doc, 'a', { href: link[1] }, link[0]));
    });
    para(doc, node, 'Reproduce the checks on synthetic fixtures from a clone of the repository:');
    add(add(node, el(doc, 'pre')), el(doc, 'code', null,
      './.venv-ml/bin/python -m pip install -r ml/requirements-ml.lock.txt\n' +
      './.venv-ml/bin/python -m pip install --no-deps -e ml\n' +
      './.venv-ml/bin/python -m pytest ml/tests -q'));
    para(doc, node, 'Each release bundle is checked against its SHA256SUMS before it is loaded, and its release id is computed from its manifests, so the id on this page names exactly one set of model files.');
  }

  // ---- mount -------------------------------------------------------------------------------------

  function normaliseConfig(config) {
    var raw = config || {};
    var base = typeof raw.dataBase === 'string' && raw.dataBase !== '' ? raw.dataBase : '../data/ml/';
    return {
      raw: raw,
      dataBase: base.charAt(base.length - 1) === '/' ? base : base + '/',
      repoUrl: typeof raw.repoUrl === 'string' && /^https:\/\//.test(raw.repoUrl) ? raw.repoUrl.replace(/\/+$/, '') : DEFAULT_REPO_URL,
      mapUrl: typeof raw.mapUrl === 'string' && raw.mapUrl !== '' ? raw.mapUrl : '../',
      fetch: typeof raw.fetch === 'function' ? raw.fetch : null
    };
  }

  function loadError(file, status) {
    var error = new Error(file + ' answered HTTP ' + status);
    error.file = file;
    error.status = status;
    return error;
  }

  function predictErrorText(response) {
    var detail = response.body && response.body.detail;
    if (response.status === 422) {
      var first = Array.isArray(detail) && detail.length > 0 ? detail[0] : null;
      return 'The API rejected the input' + (first && first.msg ? ': ' + first.msg : '.');
    }
    if (response.status === 413) {
      return 'The request is larger than the 16 KB the API accepts.';
    }
    if (response.status === 429) {
      return 'Too many requests from this address; wait a few seconds and try again.';
    }
    if (response.status === 503) {
      return 'API unavailable: the model is not loaded (HTTP 503).';
    }
    return 'Unexpected response from the API (HTTP ' + response.status + ').';
  }

  function mountMLLab(container, config) {
    var cfg = normaliseConfig(config);
    var doc = container.ownerDocument || root.document;
    var fetchImpl = cfg.fetch || (typeof root.fetch === 'function' ? root.fetch.bind(root) : null);
    var state = {
      disposed: false, controllers: [], timers: [], listeners: [], exampleButtons: [],
      releaseId: null, apiUrl: null, form: null, result: null
    };
    var host = add(container, el(doc, 'div', { 'class': 'mll', 'data-role': 'lab' }));
    para(doc, host, 'Loading the ML Lab…', { 'class': 'mll-note', role: 'status', 'data-role': 'load-status' });

    function now() {
      return root.performance && typeof root.performance.now === 'function' ? root.performance.now() : Date.now();
    }

    function listen(target, type, handler) {
      target.addEventListener(type, handler);
      state.listeners.push([target, type, handler]);
    }

    // Resolves {status, ok, body}; body is null when the response is not JSON.
    function request(url, init, timeoutMs) {
      var controller = typeof AbortController === 'function' ? new AbortController() : null;
      var options = Object.assign({ cache: 'no-store', credentials: 'omit' }, init || {});
      var timer = null;
      if (controller) {
        state.controllers.push(controller);
        options.signal = controller.signal;
        if (timeoutMs) {
          timer = setTimeout(function () { controller.abort(); }, timeoutMs);
          state.timers.push(timer);
        }
      }
      function settle() {
        if (timer !== null) {
          clearTimeout(timer);
        }
      }
      if (!fetchImpl) {
        return Promise.reject(new Error('this browser has no fetch()'));
      }
      return Promise.resolve()
        .then(function () { return fetchImpl(url, options); })
        .then(function (response) {
          return Promise.resolve()
            .then(function () { return response.json(); })
            .then(function (body) { return body; }, function () { return null; })
            .then(function (body) {
              settle();
              return { status: response.status, ok: response.ok === true, body: body };
            });
        }, function (error) {
          settle();
          throw error;
        });
    }

    function setNote(text, available) {
      state.form.note.textContent = text;
      state.form.note.setAttribute('data-state', available ? 'available' : 'unavailable');
    }

    function setFormEnabled(enabled) {
      state.form.fieldset.disabled = !enabled;
      state.form.submit.disabled = !enabled;
      state.exampleButtons.forEach(function (item) { item.button.disabled = !enabled; });
    }

    function disableForm(text) {
      setFormEnabled(false);
      setNote(text, false);
    }

    function enableForm(text) {
      setFormEnabled(true);
      setNote(text, true);
      state.exampleButtons.forEach(function (item) {
        listen(item.button, 'click', function () {
          state.form.name.value = String(item.example.borrower_name || '');
          state.form.lenders.value = (Array.isArray(item.example.lender_names) ? item.example.lender_names : []).join('\n');
          state.form.region.value = item.example.region === 'CT' ? 'CT' : 'CO';
          if (typeof state.form.name.focus === 'function') {
            state.form.name.focus();
          }
        });
      });
      listen(state.form.form, 'submit', onSubmit);
    }

    function onSubmit(event) {
      if (event && typeof event.preventDefault === 'function') {
        event.preventDefault();
      }
      if (state.disposed || !state.apiUrl || state.form.submit.disabled) {
        return;
      }
      var input = buildPredictBody(state.form.name.value, state.form.lenders.value, state.form.region.value);
      if (input.error) {
        showPredictError(doc, state, input.error);
        return;
      }
      state.result.textContent = '';
      para(doc, state.result, 'Scoring…', { 'class': 'mll-note', 'data-role': 'predict-pending' });
      state.form.submit.disabled = true;
      var started = now();
      request(state.apiUrl + '/v1/predict', {
        method: 'POST', mode: 'cors', headers: { 'content-type': 'application/json' }, body: JSON.stringify(input.body)
      }, PREDICT_TIMEOUT_MS).then(function (response) {
        if (state.disposed) {
          return;
        }
        state.form.submit.disabled = false;
        if (response.status === 200 && response.body) {
          showPrediction(doc, state, response.body, now() - started);
        } else {
          showPredictError(doc, state, predictErrorText(response));
        }
      }, function (error) {
        if (state.disposed) {
          return;
        }
        state.form.submit.disabled = false;
        showPredictError(doc, state, 'API unavailable: the request did not complete' +
          (error && error.name === 'AbortError' ? ' within ' + (PREDICT_TIMEOUT_MS / 1000) + ' s' : '') +
          '. No result is shown rather than a stale one.');
      });
    }

    function checkApi(release) {
      var apiUrl = resolveApiUrl(cfg.raw, release);
      state.apiUrl = apiUrl;
      if (!apiUrl) {
        disableForm('API unavailable: no live prediction service is configured for this release. The report on this page is static and complete without it.');
        return null;
      }
      setNote('Checking the live API at ' + apiUrl + '… (a cold start can take several seconds)', false);
      return request(apiUrl + '/readyz', { method: 'GET', mode: 'cors' }, READY_TIMEOUT_MS).then(function (response) {
        if (state.disposed) {
          return;
        }
        if (response.status !== 200 || !response.body || response.body.status !== 'ready') {
          disableForm('API unavailable: ' + apiUrl + '/readyz answered HTTP ' + response.status + '. The static report is unaffected.');
          return;
        }
        var apiRelease = response.body.release_id;
        if (apiRelease !== state.releaseId) {
          enableForm('Live API ready, but it serves release ' + formatCell('release_id', apiRelease) + ' while this report describes release ' +
            formatCell('release_id', state.releaseId) + '. Live results name the release that produced them.');
        } else {
          enableForm('Live API ready (release ' + formatCell('release_id', apiRelease) + '). Visitor inputs are not saved as feedback.');
        }
      }, function (error) {
        if (state.disposed) {
          return;
        }
        disableForm('API unavailable: ' + apiUrl + ' did not answer' +
          (error && error.name === 'AbortError' ? ' within ' + (READY_TIMEOUT_MS / 1000) + ' s' : '') + '. The static report is unaffected.');
      });
    }

    function renderAll(data) {
      host.textContent = '';
      var view = readReport(data.metrics);
      state.releaseId = data.release.release_id;
      renderHeader(doc, host, data.manifest, state.releaseId, view.provenance);
      var warnings = []
        .concat(missingPaths(data.manifest, MANIFEST_PATHS).map(function (p) { return 'manifest ' + p; }))
        .concat(missingPaths(data.metrics, METRICS_PATHS).map(function (p) { return 'metrics ' + p; }))
        .concat(missingPaths(data.examples, EXAMPLES_PATHS).map(function (p) { return 'examples ' + p; }))
        .concat(missingPaths(data.card, MODEL_CARD_PATHS).map(function (p) { return 'model card ' + p; }));
      [data.metrics, data.examples, data.card].forEach(function (doc2, index) {
        var id = doc2 && doc2.release_id;
        if (id !== state.releaseId) {
          warnings.push(['metrics', 'examples', 'model card'][index] + ' release_id ' + formatCell('release_id', id) + ' is not ' + formatCell('release_id', state.releaseId));
        }
      });
      // The page does not know which sentence is right -- the release does. What it CAN check is
      // that one is stated, and that a stated round map is reflected in it. Pinning the single
      // literal here would raise a visible warning on every honest mixed-round release.
      if (typeof view.provenance !== 'string' || view.provenance === '') {
        warnings.push('metrics label_provenance is missing');
      } else if (view.labels && view.labels.disclosure_by_round) {
        Object.keys(view.labels.disclosure_by_round).forEach(function (round) {
          if (view.provenance.indexOf(view.labels.disclosure_by_round[round]) === -1) {
            warnings.push('metrics label_provenance does not state round ' + round);
          }
        });
      }
      if (warnings.length > 0) {
        para(doc, host, 'This release’s files do not match what the page expects: ' + warnings.join('; ') + '.', { 'class': 'mll-warn', role: 'note', 'data-role': 'schema-warning' });
      }
      renderIntro(doc, host, cfg, view);
      renderTry(doc, host, data.examples, state);
      renderResultSection(doc, host, state);
      renderComparison(doc, host, view);
      renderErrors(doc, host, data.examples, data.card, view);
      renderLinks(doc, host, cfg, data.release);
    }

    function renderLoadError(error) {
      host.textContent = '';
      add(host, el(doc, 'h1', null, 'UCC ML Lab'));
      if (error && error.file === 'manifest.json' && error.status === 404) {
        para(doc, host, 'No model release has been published yet. This page shows its report once the first release is exported.', { role: 'status', 'data-role': 'load-error' });
        return;
      }
      para(doc, host, 'The Lab data could not be loaded: ' + (error && error.message ? error.message : 'unknown error') + '.', { 'class': 'mll-error', role: 'alert', 'data-role': 'load-error' });
    }

    var ready = request(cfg.dataBase + 'manifest.json').then(function (response) {
      if (state.disposed) {
        return null;
      }
      if (!response.ok || !response.body) {
        throw loadError('manifest.json', response.status);
      }
      var manifest = response.body;
      var release = currentRelease(manifest);
      if (!release) {
        throw new Error('manifest.json names no current release');
      }
      var urls = [release.metrics_url, release.examples_url, release.model_card_url];
      urls.forEach(function (url) {
        if (!isRelativeDataUrl(url)) {
          throw new Error('manifest.json lists a file outside the data directory: ' + String(url));
        }
      });
      return Promise.all(urls.map(function (url) { return request(cfg.dataBase + url); })).then(function (files) {
        files.forEach(function (file, index) {
          if (!file.ok || !file.body) {
            throw loadError(urls[index], file.status);
          }
        });
        return { manifest: manifest, release: release, metrics: files[0].body, examples: files[1].body, card: files[2].body };
      });
    }).then(function (data) {
      if (!data || state.disposed) {
        return null;
      }
      renderAll(data);
      return checkApi(data.release);
    }).catch(function (error) {
      if (!state.disposed) {
        renderLoadError(error);
      }
    });

    function cleanup() {
      if (state.disposed) {
        return;
      }
      state.disposed = true;
      state.controllers.forEach(function (controller) { controller.abort(); });
      state.timers.forEach(function (timer) { clearTimeout(timer); });
      state.listeners.forEach(function (item) { item[0].removeEventListener(item[1], item[2]); });
      state.listeners = [];
      if (host.parentNode) {
        host.parentNode.removeChild(host);
      }
    }
    cleanup.ready = ready;
    return cleanup;
  }

  return {
    LABEL_PROVENANCE: LABEL_PROVENANCE,
    MANIFEST_PATHS: MANIFEST_PATHS,
    METRICS_PATHS: METRICS_PATHS,
    EXAMPLES_PATHS: EXAMPLES_PATHS,
    MODEL_CARD_PATHS: MODEL_CARD_PATHS,
    mountMLLab: mountMLLab,
    formatCell: formatCell,
    formatEstimate: formatEstimate,
    formatDelta: formatDelta,
    decisionLabel: decisionLabel,
    rulesOutcome: rulesOutcome,
    resolveApiUrl: resolveApiUrl,
    selectErrorExamples: selectErrorExamples,
    missingPaths: missingPaths,
    readReport: readReport,
    buildPredictBody: buildPredictBody,
    currentRelease: currentRelease
  };
});
