/* The ML map: where the model and the frozen rules agree and diverge, over every scored filing.
 *
 *   window.mountMLMap(container, { dataUrl: '../data/ml/map_zips.json', reliefBase: '../data/' });
 *
 * It renders into its OWN container, deliberately not into the Lab's. ml-lab.js publishes an exact
 * list of six section headings that its tests assert on, so a seventh section rendered there would
 * fail them; and this file draws SVG, which that script has no need of.
 *
 * This file never assigns an HTML string into the document, for the same reason ml-lab.js never
 * does: every value here comes from a published JSON file and is written with textContent or
 * setAttribute. ml/tests/test_ml_map_asset.py enforces that rather than trusting this sentence.
 */
(function (root) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var PAD = 14;

  // bbox, size and ring copied from docs/index.html's STATE, so a postcode lands in the same
  // place on both maps and the relief images register against the same projection.
  var STATE = {
    co: { bbox: [-109.0448, -102.0415, 36.9930, 41.0034], w: 430, h: 300, img: 'relief_co.png',
          label: 'Colorado',
          ring: [[-109.0448, 41.0034], [-102.0415, 41.0034], [-102.0415, 36.9930], [-109.0448, 36.9930]] },
    ct: { bbox: [-73.7278, -71.7870, 40.9700, 42.0500], w: 300, h: 220, img: 'relief_ct.png',
          label: 'Connecticut',
          ring: [[-73.4877, 42.0500], [-71.7870, 42.0250], [-71.7870, 41.6400], [-71.8400, 41.4150],
                 [-71.8300, 41.3200], [-71.9600, 41.3050], [-72.0900, 41.2800], [-72.2000, 41.2600],
                 [-72.3400, 41.2550], [-72.4700, 41.2450], [-72.5800, 41.2500], [-72.6900, 41.2400],
                 [-72.8100, 41.2450], [-72.9000, 41.2300], [-73.0300, 41.1500], [-73.1100, 41.1400],
                 [-73.1800, 41.1200], [-73.2800, 41.1000], [-73.4000, 41.0300], [-73.5200, 40.9900],
                 [-73.6570, 40.9700], [-73.7278, 41.1005], [-73.4820, 41.2130]] }
  };

  var MODES = [
    { key: 'n',   label: 'All filings',  colour: '#6b6b6b', says: 'all filings scored' },
    { key: 'b',   label: 'Both agree',   colour: '#6b3fa0', says: 'filings both the rules and the model flag' },
    { key: 'om',  label: 'Model only',   colour: '#2f7d4f', says: 'filings the model flags that the rules reject' },
    { key: 'orl', label: 'Rules only',   colour: '#c0392b', says: 'filings the rules keep that the model does not flag' }
  ];

  function grouped(value) {
    var text = String(Math.round(Number(value) || 0)), out = '', i;
    for (i = 0; i < text.length; i += 1) {
      if (i > 0 && (text.length - i) % 3 === 0) { out += ','; }
      out += text.charAt(i);
    }
    return out;
  }

  function el(doc, tag, attrs, text) {
    var node = doc.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (name) {
        if (attrs[name] !== null && attrs[name] !== undefined) {
          node.setAttribute(name, String(attrs[name]));
        }
      });
    }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function svgEl(doc, tag, attrs, text) {
    var node = doc.createElementNS(NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (name) {
        if (attrs[name] !== null && attrs[name] !== undefined) {
          node.setAttribute(name, String(attrs[name]));
        }
      });
    }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function add(parent, child) { parent.appendChild(child); return child; }

  // The projection docs/index.html uses, unchanged.
  function projector(key) {
    var s = STATE[key], w = s.bbox[0], e = s.bbox[1], so = s.bbox[2], no = s.bbox[3];
    var kx = Math.cos((so + no) / 2 * Math.PI / 180), dx = (e - w) * kx, dy = no - so;
    var sc = Math.min((s.w - 2 * PAD) / dx, (s.h - 2 * PAD) / dy);
    var ox = (s.w - dx * sc) / 2, oy = (s.h - dy * sc) / 2;
    return function (lon, lat) { return [ox + (lon - w) * kx * sc, oy + (no - lat) * sc]; };
  }

  function ringPath(key, project) {
    return STATE[key].ring.map(function (c, i) {
      return (i ? 'L' : 'M') + project(c[0], c[1]).map(function (v) { return v.toFixed(1); }).join(' ');
    }).join('') + 'Z';
  }

  function drawPanel(doc, svg, key, zips, mode, reliefBase) {
    var s = STATE[key], project = projector(key), region = key.toUpperCase();
    while (svg.firstChild) { svg.removeChild(svg.firstChild); }
    var path = ringPath(key, project);

    var defs = add(svg, svgEl(doc, 'defs'));
    var clip = add(defs, svgEl(doc, 'clipPath', { id: 'mlm-clip-' + key }));
    add(clip, svgEl(doc, 'path', { d: path }));
    add(svg, svgEl(doc, 'path', { 'class': 'mlm-shape', d: path }));

    var tl = project(s.bbox[0], s.bbox[3]), br = project(s.bbox[1], s.bbox[2]);
    add(svg, svgEl(doc, 'image', {
      'class': 'mlm-relief', 'clip-path': 'url(#mlm-clip-' + key + ')',
      preserveAspectRatio: 'none', href: reliefBase + s.img,
      x: tl[0].toFixed(1), y: tl[1].toFixed(1),
      width: (br[0] - tl[0]).toFixed(1), height: (br[1] - tl[1]).toFixed(1)
    }));

    var here = zips.filter(function (z) { return z.r === region && z[mode.key] > 0; });
    var max = here.reduce(function (a, z) { return z[mode.key] > a ? z[mode.key] : a; }, 0) || 1;
    here.sort(function (a, b) { return b[mode.key] - a[mode.key]; }).forEach(function (z) {
      var xy = project(z.lon, z.lat);
      var r = 1.6 + 11 * Math.sqrt(z[mode.key]) / Math.sqrt(max);
      var dot = add(svg, svgEl(doc, 'circle', {
        cx: xy[0].toFixed(1), cy: xy[1].toFixed(1), r: r.toFixed(1),
        fill: mode.colour, 'fill-opacity': '.42', stroke: mode.colour, 'stroke-width': '.7'
      }));
      add(dot, svgEl(doc, 'title', null, z.c + ' ' + z.z + ' — ' + grouped(z[mode.key])));
    });
    return here.length;
  }

  function tile(doc, parent, value, label) {
    var box = add(parent, el(doc, 'div', { 'class': 'mlm-stat' }));
    add(box, el(doc, 'b', null, value));
    add(box, el(doc, 'span', null, label));
  }

  function table(doc, parent, head, body) {
    var wrap = add(parent, el(doc, 'div', { 'class': 'mlm-scroll' }));
    var node = add(wrap, el(doc, 'table', { 'data-role': 'map-table' }));
    var thead = add(node, el(doc, 'thead'));
    var hrow = add(thead, el(doc, 'tr'));
    head.forEach(function (c) { add(hrow, el(doc, 'th', c.num ? { 'class': 'num' } : null, c.t)); });
    var tbody = add(node, el(doc, 'tbody'));
    body.forEach(function (row) {
      var tr = add(tbody, el(doc, 'tr'));
      row.forEach(function (cell, i) {
        add(tr, el(doc, 'td', head[i].num ? { 'class': 'num' } : null, cell));
      });
    });
  }

  function mountMLMap(container, config) {
    var cfg = config || {};
    var doc = container.ownerDocument || root.document;
    var dataUrl = cfg.dataUrl || '../data/ml/map_zips.json';
    var reliefBase = cfg.reliefBase || '../data/';
    var fetchImpl = cfg.fetch || (typeof root.fetch === 'function' ? root.fetch.bind(root) : null);
    // SVG needs createElementNS. Where it is absent (a minimal test document), the map is simply
    // not drawn rather than throwing and taking the rest of the page down with it.
    if (!doc.createElementNS || !fetchImpl) { return null; }

    var wrap = add(container, el(doc, 'div', { 'class': 'mlm' }));
    add(wrap, el(doc, 'h2', { id: 'ml-map-heading' },
      'Where the model and the rules disagree, on the map'));
    var lead = add(wrap, el(doc, 'p', { 'class': 'mlm-note', 'data-role': 'map-lead' }, 'Loading the map…'));
    var tiles = add(wrap, el(doc, 'div', { 'class': 'mlm-tiles' }));
    var chips = add(wrap, el(doc, 'div', { 'class': 'mlm-chips', role: 'group',
      'aria-label': 'What to plot' }));
    var legend = add(wrap, el(doc, 'div', { 'class': 'mlm-legend' }));
    var maps = add(wrap, el(doc, 'div', { 'class': 'mlm-maps' }));
    var note = add(wrap, el(doc, 'p', { 'class': 'mlm-note', 'data-role': 'map-note' }));
    var tableHost = add(wrap, el(doc, 'div'));

    var panels = {};
    ['co', 'ct'].forEach(function (key) {
      var panel = add(maps, el(doc, 'div', { 'class': 'mlm-panel' }));
      var svg = add(panel, svgEl(doc, 'svg', {
        viewBox: '0 0 ' + STATE[key].w + ' ' + STATE[key].h, role: 'img',
        'data-role': 'map-' + key,
        'aria-label': STATE[key].label + ' filings by postcode'
      }));
      add(panel, el(doc, 'div', { 'class': 'mlm-cap' }, STATE[key].label));
      panels[key] = svg;
    });

    [['#6b3fa0', 'both agree — the model confirms the rules'],
     ['#2f7d4f', 'model only — flagged by the model, rejected by the rules'],
     ['#c0392b', 'rules only — kept by the rules, not flagged by the model']
    ].forEach(function (pair) {
      var item = add(legend, el(doc, 'span'));
      add(item, el(doc, 'i', { style: 'background:' + pair[0] }));
      add(item, el(doc, 'span', null, pair[1]));
    });

    var state = { mode: MODES[0], data: null, listeners: [] };

    function redraw() {
      if (!state.data) { return; }
      var zips = state.data.zips;
      ['co', 'ct'].forEach(function (key) {
        drawPanel(doc, panels[key], key, zips, state.mode, reliefBase);
      });
      var shown = zips.reduce(function (a, z) { return a + (z[state.mode.key] || 0); }, 0);
      var places = zips.filter(function (z) { return z[state.mode.key] > 0; }).length;
      note.textContent = 'Showing ' + grouped(shown) + ' ' + state.mode.says + ', across '
        + grouped(places) + ' postcodes.';
      while (tableHost.firstChild) { tableHost.removeChild(tableHost.firstChild); }
      var top = zips.slice().filter(function (z) { return z[state.mode.key] > 0; })
        .sort(function (a, b) { return b[state.mode.key] - a[state.mode.key]; }).slice(0, 10);
      table(doc, tableHost,
        [{ t: 'postcode' }, { t: 'place' }, { t: 'region' }, { t: 'filings', num: true },
         { t: 'rules keep', num: true }, { t: 'model flags', num: true },
         { t: 'both', num: true }, { t: 'model only', num: true }],
        top.map(function (z) {
          return [z.z, z.c, z.r, grouped(z.n), grouped(z.a), grouped(z.s), grouped(z.b), grouped(z.om)];
        }));
    }

    MODES.forEach(function (mode) {
      var chip = add(chips, el(doc, 'button', {
        type: 'button', 'class': 'mlm-chip', 'data-role': 'map-chip-' + mode.key,
        'aria-pressed': mode.key === state.mode.key ? 'true' : 'false'
      }, mode.label));
      function onClick() {
        state.mode = mode;
        MODES.forEach(function (other, i) {
          chips.childNodes[i].setAttribute('aria-pressed', other.key === mode.key ? 'true' : 'false');
        });
        redraw();
      }
      if (chip.addEventListener) {
        chip.addEventListener('click', onClick);
        state.listeners.push([chip, onClick]);
      }
    });

    var ready = fetchImpl(dataUrl).then(function (response) {
      if (!response || !response.ok) { throw new Error('HTTP ' + (response && response.status)); }
      return response.json();
    }).then(function (data) {
      state.data = data;
      var totals = data.totals || {};
      lead.textContent = 'Every scored filing that carries a postcode is plotted at that postcode: '
        + grouped(totals.filings) + ' filings across ' + grouped(totals.zips)
        + ' postcodes. A bubble’s area is how many filings sit there. This is the whole '
        + 'population, not a sample — the rules and the model were both run over all of it.';
      tile(doc, tiles, grouped(totals.both), 'both agree');
      tile(doc, tiles, grouped(totals.model_only), 'model only — rules rejected these');
      tile(doc, tiles, grouped(totals.rules_only), 'rules only — model did not flag these');
      tile(doc, tiles, grouped(totals.filings), 'filings mapped');
      redraw();
      return data;
    }).catch(function (error) {
      lead.textContent = 'The map data could not be loaded: '
        + (error && error.message ? error.message : 'unknown error') + '.';
      return null;
    });

    function cleanup() {
      state.listeners.forEach(function (pair) {
        if (pair[0].removeEventListener) { pair[0].removeEventListener('click', pair[1]); }
      });
      state.listeners = [];
      while (container.firstChild) { container.removeChild(container.firstChild); }
    }
    cleanup.ready = ready;
    return cleanup;
  }

  var api = { mountMLMap: mountMLMap, projector: projector, ringPath: ringPath, STATE: STATE, MODES: MODES };
  if (typeof module === 'object' && module && module.exports) {
    module.exports = api;
  } else {
    root.mountMLMap = mountMLMap;
    root.MLMap = api;
  }
}(typeof window !== 'undefined' ? window : this));
