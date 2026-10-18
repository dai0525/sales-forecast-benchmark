/* 销量预测工作台 · LightGBM 交互讲解组件
   纯原生 JS + 内联 SVG，无框架 / 无 CDN / 无外链。
   数据：运行时 ctx.load('lgbm.json')；拿不到就用内置示意数据，并在图下标注。 */
(function () {
  'use strict';

  var ID = 'lightgbm';

  /* ---------- 设计系统色板（与 css/app.css 一致，唯一强调色 #1D4ED8） ---------- */
  var C = {
    blue: '#1D4ED8', blue2: '#3B82F6', blue3: '#60A5FA', blue4: '#BFD2F6',
    ink: '#0F1B2D', ink2: '#5A6B85', grey: '#8A97AB',
    hair: '#E6EBF3', line: '#D6E2FB', wash: '#EFF4FF'
  };
  var RAMP = ['#1D4ED8', '#2563EB', '#3B82F6', '#4F8BF7', '#60A5FA', '#82B4FA', '#A6C6F7', '#BFD2F6'];
  var FSLBL = ['仅日历', '+ lag', '+ rolling'];

  /* ---------- 小工具 ---------- */
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function arr(v) { return Array.isArray(v) ? v.filter(isNum) : []; }
  function r1(v) { return Math.round(v * 10) / 10; }
  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function lin(d0, d1, r0, r1v) {
    var k = d1 === d0 ? 0 : (r1v - r0) / (d1 - d0);
    return function (v) { return r0 + (v - d0) * k; };
  }
  function pct(v, d) { return isNum(v) ? (v * 100).toFixed(d == null ? 1 : d) + '%' : '–'; }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function pathOf(pts) {
    if (!pts.length) return '';
    return pts.map(function (p, i) { return (i ? 'L' : 'M') + r1(p[0]) + ' ' + r1(p[1]); }).join(' ');
  }
  function ptsOf(vals, xf, yf, maxN) {
    maxN = maxN || 240;
    var step = Math.max(1, Math.ceil(vals.length / maxN)), out = [], i;
    for (i = 0; i < vals.length; i += step) out.push([xf(i), yf(vals[i])]);
    var last = vals.length - 1;
    if (last > 0 && last % step) out.push([xf(last), yf(vals[last])]);
    return out;
  }
  function svgWrap(w, h, inner, label) {
    return '<svg class="sfw-plot" viewBox="0 0 ' + w + ' ' + h + '" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="' + esc(label) + '">' + inner + '</svg>';
  }
  function gridY(x1, x2, y0, y1, mn, mx) {
    var out = [], i, v, yy;
    for (i = 0; i <= 2; i++) {
      v = mn + (mx - mn) * i / 2;
      yy = lin(mn, mx, y1, y0)(v);
      out.push('<line class="grid" x1="' + x1 + '" y1="' + r1(yy) + '" x2="' + x2 + '" y2="' + r1(yy) + '"/>');
      out.push('<text x="' + (x1 - 4) + '" y="' + r1(yy + 3.5) + '" text-anchor="end">' + Math.round(v) + '</text>');
    }
    return out.join('');
  }
  function el(tag, cls) { var n = document.createElement(tag); if (cls) n.className = cls; return n; }
  function html(tag, cls, inner) { var n = el(tag, cls); n.innerHTML = inner; return n; }

  /* ---------- 样式（全部 .sfw-lightgbm- 前缀） ---------- */
  function injectStyle() {
    if (document.getElementById('sfw-lightgbm-css')) return;
    var s = document.createElement('style');
    s.id = 'sfw-lightgbm-css';
    s.textContent = [
      '.sfw-lightgbm-card{border:1px solid var(--hair);border-radius:14px;padding:12px;margin:0 0 14px;background:#fff;min-width:0}',
      '.sfw-lightgbm-hd{display:flex;align-items:baseline;justify-content:space-between;gap:8px;flex-wrap:wrap;margin:0 0 8px}',
      '.sfw-lightgbm-hd b{font-size:14px;font-weight:700;color:var(--ink)}',
      '.sfw-lightgbm-hd span{font-size:12px;color:var(--ink-2)}',
      '.sfw-lightgbm-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:2px 0 4px}',
      '.sfw-lightgbm-lab{font-size:13px;font-weight:600;color:var(--ink);white-space:nowrap}',
      '.sfw-lightgbm-range{flex:1 1 120px;min-width:110px;height:40px;margin:0;touch-action:pan-y}',
      '.sfw-lightgbm-hval{font:700 15px/1 ui-monospace,monospace;color:var(--blue);min-width:30px;text-align:right}',
      '.sfw-lightgbm-metricrow{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:2px 0 6px}',
      '.sfw-lightgbm-sub{font-size:12px;color:var(--ink-2);line-height:1.45;flex:1 1 150px;min-width:0}',
      '.sfw-lightgbm-demo{margin-top:5px;background:var(--wash);border-radius:7px;padding:5px 8px;color:#3E5A8A}',
      '.sfw-lightgbm-chips{margin:0 0 8px}',
      '.sfw-lightgbm-chips .sfw-chip{min-height:42px;padding:9px 13px}',
      '.sfw-lightgbm-why{margin:8px 0 0;font-size:13px;line-height:1.55;color:var(--ink);background:var(--wash);border-left:3px solid var(--blue);padding:8px 10px;border-radius:0 8px 8px 0}',
      '.sfw-lightgbm-why b{color:var(--blue)}',
      '.sfw-lightgbm-cards{display:grid;grid-template-columns:1fr;gap:0;min-width:0}',
      '.sfw-lightgbm-plot{margin:6px 0 0;min-width:0;max-width:480px}',
      '@media (min-width:720px){.sfw-lightgbm-cards{grid-template-columns:1fr 1fr;gap:14px;align-items:start}.sfw-lightgbm-cards>.sfw-lightgbm-card{margin:0}}'
    ].join('');
    (document.head || document.documentElement).appendChild(s);
  }

  /* ---------- 示意数据（确定性，每次渲染一致） ---------- */
  function rng(seed) {
    var s = seed >>> 0;
    return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  function demoData() {
    var r = rng(20261017);
    var dow = [0.93, 1.01, 1.0, 1.05, 1.13, 1.25, 1.17]; // 周一 … 周日
    var clean = [], noisy = [], i, v;
    for (i = 0; i < 134; i++) {
      v = (128 + i * 0.62 + 8 * Math.sin(i / 12.5) + 6 * Math.cos(i / 29)) * dow[i % 7];
      if (i % 41 === 17) v *= 1.24;                      // 档期促销脉冲
      clean.push(v);
      noisy.push(v * (1 + (r() - 0.5) * 0.1));
    }
    var ytr = noisy.slice(0, 120).map(function (x) { return Math.max(1, Math.round(x)); });
    var yte = noisy.slice(120).map(function (x) { return Math.max(1, Math.round(x)); });
    var pR = [], pD = [];
    for (i = 0; i < 14; i++) {
      // 递归：单步很准，但每步的偏差被喂回下一步，误差随 h 放大
      pR.push(Math.max(1, Math.round(yte[i] + 1.4 + i * 2.1 + (r() - 0.5) * 3)));
      // 对照预测：一次输出 14 个值，误差不随 h 累积
      pD.push(Math.max(1, Math.round(yte[i] * 1.006 + (r() - 0.5) * 14)));
    }
    var fi = [
      { name: 'lag_7', value: 0.194 }, { name: 'rolling_mean_28', value: 0.173 },
      { name: 'lag_14', value: 0.131 }, { name: 'rolling_std_7', value: 0.112 },
      { name: 'lag_1', value: 0.097 }, { name: 'dow', value: 0.081 },
      { name: 'rolling_mean_7', value: 0.066 }, { name: 'month', value: 0.048 },
      { name: 'promo_flag', value: 0.041 }, { name: 'lag_28', value: 0.032 }
    ];
    var fset = [
      { name: 'calendar', wape: 0.138 },
      { name: 'calendar+lag', wape: 0.101 },
      { name: 'calendar+lag+rolling', wape: 0.086 }
    ];
    return {
      series_id: null, demo: true,
      y_train: ytr, y_test: yte, pred_recursive: pR, pred_direct: pD,
      feature_importance: fi, feature_sets: fset,
      horizon_curve: pR.map(function (_, k) { return { h: k + 1, wape: wapeAt(pR, yte, k + 1) }; })
    };
  }

  function normalize(d) {
    return {
      series_id: d.series_id || null,
      demo: false,
      y_train: arr(d.y_train),
      y_test: arr(d.y_test),
      pred_recursive: arr(d.pred_recursive),
      pred_direct: arr(d.pred_direct),
      feature_importance: (Array.isArray(d.feature_importance) ? d.feature_importance : [])
        .filter(function (x) { return x && isNum(x.value) && x.name != null; }),
      feature_sets: (Array.isArray(d.feature_sets) ? d.feature_sets : [])
        .filter(function (x) { return x && isNum(x.wape); }),
      horizon_curve: (Array.isArray(d.horizon_curve) ? d.horizon_curve : [])
        .filter(function (x) { return x && isNum(x.wape) && isNum(x.h); })
    };
  }

  function wapeAt(pred, truth, h) {
    var a = 0, b = 0, n = Math.min(h, pred.length, truth.length), i;
    for (i = 0; i < n; i++) { a += Math.abs(pred[i] - truth[i]); b += Math.abs(truth[i]); }
    return b > 0 ? a / b : null;
  }

  /* ---------- 图 1：历史 + 真值 + 递归 / 对照预测 ---------- */
  function seriesChart(D) {
    var W = 332, H = 196, pl = 30, pr = 8, pt = 12, pb = 26;
    var tr = arr(D.y_train), te = arr(D.y_test);
    var prd = arr(D.pred_recursive), pdd = arr(D.pred_direct);
    var N = tr.length + te.length;
    if (!N) return '';
    var cut = tr.length;
    // 历史 120 点占左 55%，未来 14 步放大到右 45%（切分处横轴压缩，两侧各自标注）
    var plotW = (W - pr) - pl;
    var histW = cut > 1 ? (te.length ? plotW * 0.55 : plotW) : 0;
    var futW = plotW - histW;
    function x(i) {
      if (i <= cut - 1) return pl + (cut > 1 ? histW * (i / (cut - 1)) : 0);
      return pl + histW + futW * ((i - (cut - 1)) / Math.max(1, N - 1 - (cut - 1)));
    }
    var all = tr.concat(te, prd, pdd);
    var mn = Math.min.apply(null, all), mx = Math.max.apply(null, all);
    var pad = Math.max(10, (mx - mn) * 0.12);
    mn -= pad; mx += pad;
    var y = lin(mn, mx, H - pb, pt);
    var o = [];

    o.push(gridY(pl, W - pr, pt, H - pb, mn, mx));
    if (cut > 0) {
      o.push('<line x1="' + r1(x(cut - 1)) + '" y1="' + pt + '" x2="' + r1(x(cut - 1)) + '" y2="' + (H - pb) + '" stroke="' + C.line + '" stroke-width="1" stroke-dasharray="3 3"/>');
    }
    // 历史：灰实线；超过 240 点时按原始索引抽稀（契约要求 >300 点降采样）
    o.push('<path d="' + pathOf(ptsOf(tr, x, y)) + '" fill="none" stroke="' + C.ink2 + '" stroke-width="1.2" stroke-linejoin="round"/>');

    // 未来真值：灰虚线（从切分点连过去，避免断口）
    var tp = cut ? [[x(cut - 1), y(tr[cut - 1])]] : [];
    te.forEach(function (v, i) { tp.push([x(cut + i), y(v)]); });
    o.push('<path d="' + pathOf(tp) + '" fill="none" stroke="' + C.grey + '" stroke-width="1.7" stroke-dasharray="4 3" stroke-linejoin="round"/>');

    // 对照预测：浅蓝
    if (pdd.length) {
      var dp = cut ? [[x(cut - 1), y(tr[cut - 1])]] : [];
      pdd.forEach(function (v, i) { dp.push([x(cut + i), y(v)]); });
      o.push('<path d="' + pathOf(dp) + '" fill="none" stroke="' + C.blue3 + '" stroke-width="1.8" stroke-linejoin="round"/>');
    }
    // 递归：强调蓝
    if (prd.length) {
      var rp = cut ? [[x(cut - 1), y(tr[cut - 1])]] : [];
      prd.forEach(function (v, i) { rp.push([x(cut + i), y(v)]); });
      o.push('<path d="' + pathOf(rp) + '" fill="none" stroke="' + C.blue + '" stroke-width="2.2" stroke-linejoin="round"/>');
      var last = rp[rp.length - 1];
      o.push('<circle cx="' + r1(last[0]) + '" cy="' + r1(last[1]) + '" r="2.8" fill="' + C.blue + '"/>');
    }

    o.push('<text x="' + r1(pl + histW / 2) + '" y="' + (H - 8) + '" text-anchor="middle">' + cut + ' 天历史</text>');
    o.push('<text x="' + r1(pl + histW + futW / 2) + '" y="' + (H - 8) + '" text-anchor="middle">未来 ' + te.length + ' 步</text>');
    return svgWrap(W, H, o.join(''), 'LightGBM 递归多步与对照预测预测对比');
  }

  /* ---------- 图 2：h 步误差累积（上：放大预测；下：每步绝对误差） ---------- */
  function horizonChart(D, h) {
    var W = 332, H = 244, pl = 30, pr = 8;
    var tr = arr(D.y_train), te = arr(D.y_test);
    var prd = arr(D.pred_recursive), pdd = arr(D.pred_direct);
    var nH = Math.min(14, te.length);
    if (!nH) return '';

    var tail = tr.slice(-16);
    var N = tail.length + nH;
    var x = lin(0, Math.max(1, N - 1), pl, W - pr);
    var topY0 = 12, topY1 = 138, barY0 = 158, barY1 = 214;

    var hist = (tail.length ? tail : te.slice(0, 1)).concat(te.slice(0, nH), prd.slice(0, nH), pdd.slice(0, nH));
    var mn = Math.min.apply(null, hist), mx = Math.max.apply(null, hist);
    var pad = Math.max(8, (mx - mn) * 0.14);
    mn -= pad; mx += pad;
    var y = lin(mn, mx, topY1, topY0);

    var o = [];
    o.push(gridY(pl, W - pr, topY0, topY1, mn, mx));

    // 预测起点
    if (tail.length) {
      o.push('<line x1="' + r1(x(tail.length - 1)) + '" y1="' + topY0 + '" x2="' + r1(x(tail.length - 1)) + '" y2="' + topY1 + '" stroke="' + C.line + '" stroke-width="1" stroke-dasharray="3 3"/>');
      o.push('<path d="' + pathOf(tail.map(function (v, i) { return [x(i), y(v)]; })) + '" fill="none" stroke="' + C.ink2 + '" stroke-width="1.2" stroke-linejoin="round"/>');
    }
    // 真值
    var tp = [];
    if (tail.length) tp.push([x(tail.length - 1), y(tail[tail.length - 1])]);
    te.slice(0, nH).forEach(function (v, i) { tp.push([x(tail.length + i), y(v)]); });
    o.push('<path d="' + pathOf(tp) + '" fill="none" stroke="' + C.grey + '" stroke-width="1.7" stroke-dasharray="4 3" stroke-linejoin="round"/>');

    // 对照预测（前 h 步）
    if (pdd.length) {
      var dp = tail.length ? [[x(tail.length - 1), y(tail[tail.length - 1])]] : [];
      for (var d = 0; d < Math.min(h, nH, pdd.length); d++) dp.push([x(tail.length + d), y(pdd[d])]);
      if (dp.length > 1) o.push('<path d="' + pathOf(dp) + '" fill="none" stroke="' + C.blue3 + '" stroke-width="1.6" stroke-linejoin="round" opacity="0.9"/>');
    }
    // 递归（只画前 h 步）
    var rp = tail.length ? [[x(tail.length - 1), y(tail[tail.length - 1])]] : [];
    var hi = Math.min(h, nH, prd.length);
    for (var i2 = 0; i2 < hi; i2++) rp.push([x(tail.length + i2), y(prd[i2])]);
    if (rp.length > 1) {
      o.push('<path d="' + pathOf(rp) + '" fill="none" stroke="' + C.blue + '" stroke-width="2.2" stroke-linejoin="round"/>');
      rp.slice(1).forEach(function (p) {
        o.push('<circle cx="' + r1(p[0]) + '" cy="' + r1(p[1]) + '" r="2.2" fill="#fff" stroke="' + C.blue + '" stroke-width="1.6"/>');
      });
    }

    // ---- 误差柱 ----
    var errs = [], k, maxE = 1;
    for (k = 0; k < nH; k++) {
      var pv = isNum(prd[k]) ? prd[k] : te[k];
      errs.push(Math.abs(pv - te[k]));
      if (errs[k] > maxE) maxE = errs[k];
    }
    var bw = clamp((x(1) - x(0)) * 0.58, 3, 9);
    o.push('<line x1="' + pl + '" y1="' + barY1 + '" x2="' + (W - pr) + '" y2="' + barY1 + '" stroke="' + C.line + '" stroke-width="1"/>');
    var tops = [];
    for (k = 0; k < nH; k++) {
      var bh = Math.max(1.5, (errs[k] / maxE) * (barY1 - barY0));
      var cx = x(tail.length + k), by = barY1 - bh;
      tops.push([cx, by]);
      o.push('<rect x="' + r1(cx - bw / 2) + '" y="' + r1(by) + '" width="' + r1(bw) + '" height="' + r1(bh) + '" rx="2" fill="' + (k < h ? C.blue : C.line) + '"/>');
    }
    o.push('<path d="' + pathOf(tops) + '" fill="none" stroke="' + C.blue3 + '" stroke-width="1.3" stroke-linejoin="round"/>');
    // 当前步标记
    var cur = tops[clamp(h, 1, nH) - 1];
    if (cur) {
      o.push('<line x1="' + r1(cur[0]) + '" y1="' + r1(cur[1]) + '" x2="' + r1(cur[0]) + '" y2="' + barY1 + '" stroke="' + C.blue + '" stroke-width="1" stroke-dasharray="2 2"/>');
      o.push('<circle cx="' + r1(cur[0]) + '" cy="' + r1(cur[1]) + '" r="2.6" fill="' + C.blue + '"/>');
    }
    // 轴标注
    o.push('<text x="' + (pl - 4) + '" y="' + (barY1 - 2) + '" text-anchor="end">0</text>');
    o.push('<text x="' + (pl - 4) + '" y="' + (barY0 + 4) + '" text-anchor="end">' + Math.round(maxE) + '</text>');
    [0, 3, 6, 9, nH - 1].forEach(function (idx) {
      if (idx >= nH) return;
      var an = idx === 0 ? 'start' : (idx === nH - 1 ? 'end' : 'middle');
      o.push('<text x="' + r1(x(tail.length + idx)) + '" y="' + (H - 10) + '" text-anchor="' + an + '">h=' + (idx + 1) + '</text>');
    });
    o.push('<text x="' + pl + '" y="' + (barY0 - 8) + '" text-anchor="start">每步绝对误差（件）</text>');
    return svgWrap(W, H, o.join(''), '递归预测的误差随步长累积');
  }

  /* ---------- 图 3：特征集 WAPE 对比 ---------- */
  function fsChart(sets, act) {
    var W = 332, rowH = 32, H = sets.length * rowH + 8;
    var pl = 66, pr = 44, i;
    var max = 1e-9;
    for (i = 0; i < sets.length; i++) if (sets[i].wape > max) max = sets[i].wape;
    var bw = W - pl - pr;
    var o = [];
    o.push('<line x1="' + pl + '" y1="4" x2="' + pl + '" y2="' + (H - 8) + '" stroke="' + C.hair + '" stroke-width="1"/>');
    sets.forEach(function (s, k) {
      var cy = 8 + k * rowH + rowH / 2 - 4;
      var w = Math.max(2, bw * (s.wape / max));
      var on = k === act;
      var lab = String(FSLBL[k] || s.name || ('集 ' + (k + 1)));
      if (lab.length > 8) lab = lab.slice(0, 7) + '…';
      o.push('<text x="' + (pl - 6) + '" y="' + r1(cy + 3.5) + '" text-anchor="end" fill="' + (on ? C.blue : C.ink2) + '" font-weight="' + (on ? '700' : '400') + '">' + esc(lab) + '</text>');
      o.push('<rect x="' + pl + '" y="' + r1(cy - 7) + '" width="' + r1(w) + '" height="14" rx="4" fill="' + (on ? C.blue : C.blue4) + '"/>');
      o.push('<text x="' + r1(pl + w + 6) + '" y="' + r1(cy + 3.5) + '" text-anchor="start" fill="' + (on ? C.blue : C.ink2) + '" font-weight="' + (on ? '700' : '400') + '">' + pct(s.wape) + '</text>');
    });
    return svgWrap(W, H, o.join(''), '三种特征集的 WAPE 对比');
  }

  /* ---------- 图 4：特征重要性 Top 8 ---------- */
  function impChart(list) {
    var rows = list.slice(0, 8);
    if (!rows.length) return '';
    var W = 332, rowH = 26, H = rows.length * rowH + 6;
    var pl = 108, pr = 40;
    var max = 1e-9, tot = 1e-9, i;
    for (i = 0; i < rows.length; i++) {
      if (rows[i].value > max) max = rows[i].value;
      tot += rows[i].value;
    }
    var bw = W - pl - pr;
    var o = [];
    rows.forEach(function (it, k) {
      var cy = 6 + k * rowH + rowH / 2 - 4;
      var w = Math.max(2, bw * (it.value / max));
      var nm = String(it.name);
      if (nm.length > 15) nm = nm.slice(0, 14) + '…';
      o.push('<text x="' + (pl - 6) + '" y="' + r1(cy + 3.5) + '" text-anchor="end">' + esc(nm) + '</text>');
      o.push('<rect x="' + pl + '" y="' + r1(cy - 6) + '" width="' + r1(w) + '" height="12" rx="3" fill="' + RAMP[k % RAMP.length] + '"/>');
      o.push('<text x="' + (W - pr + 4) + '" y="' + r1(cy + 3.5) + '" text-anchor="start">' + (it.value / tot * 100).toFixed(1) + '</text>');
    });
    return svgWrap(W, H, o.join(''), '特征重要性排名（Top 8）');
  }

  /* ---------- 挂载 ---------- */
  async function mount(root, ctx) {
    injectStyle();

    var raw = (ctx && ctx.data && (ctx.data.lgbm || ctx.data[ID])) || null;
    if (!raw && ctx && typeof ctx.load === 'function') {
      try { raw = await ctx.load('lgbm.json'); } catch (e) { raw = null; }
    }
    var real = !!(raw && arr(raw.y_train).length >= 2 && arr(raw.y_test).length >= 1);
    var D = real ? normalize(raw) : demoData();
    var dd = demoData();
    var demoFs = false, demoFi = false;
    if (!D.feature_sets.length) { D.feature_sets = dd.feature_sets; demoFs = true; }
    if (!D.feature_importance.length) { D.feature_importance = dd.feature_importance; demoFi = true; }
    D.feature_importance = D.feature_importance.slice().sort(function (a, b) { return b.value - a.value; });

    var nH = Math.min(14, arr(D.y_test).length || 14);
    var isDemo = D.demo || demoFs || demoFi;
    var demoCapText = '示意数据，真实结果生成后自动替换。';

    /* 导语 */
    root.appendChild(html('p', 'algo-lead',
      'LightGBM 把「预测销量」改写成<b>表格回归</b>：每行是一个 (SKU, 日期)，特征是 lag_7、28 天滚动均值、星期几，标签是当天销量，然后让梯度提升树拟合这张大表。' +
      '它的杀手锏是 leaf-wise 生长 + 直方图分桶，几百万行也能几十秒训完；而一旦把 <b>lag / rolling</b> 特征喂进去，树能自动学到「上周同日卖得好，这周大概也会好」。' +
      '代价是：递归多步要把上一步的<b>预测值</b>塞回特征里，步长越长越飘——第二个例子就是让你亲手拖出这条误差曲线。'));

    /* ---------- 例子 1：真实序列 ---------- */
    var cards = el('div', 'sfw-lightgbm-cards');
    var c1 = el('div', 'sfw-lightgbm-card');
    c1.appendChild(html('div', 'sfw-lightgbm-hd',
      '<b>例子 1 · ' + (D.demo ? '一条日销量序列（示意）' : '一条真实日销量序列') + '</b><span>' + (D.series_id ? esc(D.series_id) : 'M4 · Daily') + ' · 训练 ' + arr(D.y_train).length + ' 天 / 预测 ' + arr(D.y_test).length + ' 天</span>'));
    c1.appendChild(html('div', 'sfw-legend',
      '<span><i style="background:' + C.ink2 + '"></i>历史（灰实线）</span>' +
      '<span><i style="background:' + C.grey + '"></i>未来真值（灰虚线）</span>' +
      '<span><i style="background:' + C.blue + '"></i>递归预测</span>' +
      '<span><i style="background:' + C.blue3 + '"></i>对照预测</span>'));
    var sHost = el('div', 'sfw-lightgbm-plot');
    sHost.innerHTML = seriesChart(D);
    c1.appendChild(sHost);
    c1.appendChild(html('p', 'sfw-cap',
      '蓝线为递归预测；本仓库真实数据中的浅蓝线为 SeasonalNaive 对照，共 ' + arr(D.y_test).length + ' 个值。' +
      '前几步两条线几乎重合，越往后差距越大——这就是下一张图要放大的东西。'));
    if (D.demo) c1.appendChild(html('p', 'sfw-cap sfw-lightgbm-demo', demoCapText));
    cards.appendChild(c1);

    /* ---------- 例子 2：h 步误差累积 ---------- */
    var c2 = el('div', 'sfw-lightgbm-card');
    c2.appendChild(html('div', 'sfw-lightgbm-hd', '<b>例子 2 · 递归多步：误差怎么被一步步放大</b><span>拖动滑块看 h 步</span>'));

    var row = el('div', 'sfw-lightgbm-row');
    row.appendChild(html('label', 'sfw-lightgbm-lab', '预测步数 h'));
    var hIn = el('input', 'sfw-slider sfw-lightgbm-range');
    hIn.type = 'range'; hIn.min = '1'; hIn.max = String(nH); hIn.step = '1';
    hIn.value = String(Math.min(7, nH));
    hIn.setAttribute('aria-label', '预测步数 h');
    row.appendChild(hIn);
    var hVal = el('span', 'sfw-lightgbm-hval');
    row.appendChild(hVal);
    c2.appendChild(row);

    var mrow = el('div', 'sfw-lightgbm-metricrow');
    var big = el('span', 'sfw-metric');
    mrow.appendChild(big);
    var sub = el('span', 'sfw-lightgbm-sub');
    mrow.appendChild(sub);
    c2.appendChild(mrow);

    var hHost = el('div', 'sfw-lightgbm-plot');
    c2.appendChild(hHost);

    c2.appendChild(html('p', 'sfw-cap',
      '柱高 = 每一步的绝对误差（件），深蓝是你当前选的前 h 步。递归每走一步都把上一步的预测喂回模型，误差被反覆复述、逐渐放大；' +
      '对照预测一次输出 14 个值，误差不随 h 累积——这正是生产里长步长预测更常用「对照预测」或「每步一个模型」的原因。'));
    if (D.demo) c2.appendChild(html('p', 'sfw-cap sfw-lightgbm-demo', demoCapText));
    cards.appendChild(c2);

    function curWape(h) {
      var c = D.horizon_curve, i;
      for (i = 0; i < c.length; i++) if (Math.round(c[i].h) === h) return c[i].wape;
      return wapeAt(arr(D.pred_recursive), arr(D.y_test), h);
    }
    function avgErr(h) {
      var prd = arr(D.pred_recursive), te = arr(D.y_test);
      var n = Math.min(h, prd.length, te.length), s = 0, i;
      if (!n) return null;
      for (i = 0; i < n; i++) s += Math.abs(prd[i] - te[i]);
      return s / n;
    }
    function updateH(h) {
      h = clamp(Math.round(h), 1, nH);
      hVal.textContent = String(h);
      var w = curWape(h), wd = wapeAt(arr(D.pred_direct), arr(D.y_test), h), ae = avgErr(h);
      big.innerHTML = (isNum(w) ? (w * 100).toFixed(1) : '–') + '<span class="sfw-metric-u">% WAPE · 前 ' + h + ' 步</span>';
      sub.textContent = '对照预测同 h：' + pct(wd) + ' · 平均单步误差 ' + (isNum(ae) ? ae.toFixed(1) + ' 件' : '–');
      hHost.innerHTML = horizonChart(D, h);
    }
    hIn.addEventListener('input', function () { updateH(+hIn.value); });
    updateH(+hIn.value);

    /* ---------- 例子 3：特征集开关 ---------- */
    var c3 = el('div', 'sfw-lightgbm-card');
    c3.appendChild(html('div', 'sfw-lightgbm-hd', '<b>例子 3 · 特征集：收益几乎全来自 lag / rolling</b><span>点 chip 切换</span>'));
    var chips = html('div', 'sfw-chips sfw-lightgbm-chips', '');
    var chipBtns = [];
    D.feature_sets.forEach(function (s, k) {
      var b = el('button', 'sfw-chip');
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.textContent = FSLBL[k] || s.name || ('特征集 ' + (k + 1));
      b.addEventListener('click', function () { setFs(k); });
      chips.appendChild(b);
      chipBtns.push(b);
    });
    c3.appendChild(chips);

    var m3 = el('div', 'sfw-lightgbm-metricrow');
    var fsBig = el('span', 'sfw-metric');
    m3.appendChild(fsBig);
    var fsSub = el('span', 'sfw-lightgbm-sub');
    m3.appendChild(fsSub);
    c3.appendChild(m3);

    var fsHost = el('div', 'sfw-lightgbm-plot');
    c3.appendChild(fsHost);
    c3.appendChild(html('div', 'sfw-lightgbm-why',
      '为什么 lag / rolling 是全部收益的来源：<b>日历特征</b>只能告诉模型「今天是周六」，但周六卖多少取决于「上周六卖了多少、最近四周日均多少」。' +
      'lag 与滚动统计把历史值搬到了当前行，树才有的可学；只给日历，等于让模型盲猜这条序列的水平。'));
    if (demoFs) c3.appendChild(html('p', 'sfw-cap sfw-lightgbm-demo', demoCapText));
    cards.appendChild(c3);

    function setFs(k) {
      k = clamp(k | 0, 0, D.feature_sets.length - 1);
      var s = D.feature_sets[k], base = D.feature_sets[0];
      chipBtns.forEach(function (b, j) {
        b.classList.toggle('is-on', j === k);
        b.setAttribute('aria-pressed', String(j === k));
      });
      fsBig.innerHTML = pct(s.wape).replace('%', '') + '<span class="sfw-metric-u">% WAPE</span>';
      var diff = (s.wape - base.wape) * 100;
      fsSub.textContent = k === 0
        ? '基线：只用星期几 / 月份 / 节假日'
        : '相对「仅日历」' + (diff > 0 ? '+' : '−') + Math.abs(diff).toFixed(1) + 'pp' +
          (k === 1 ? ' · 加了 lag_1/7/14/28' : ' · 再加 28 天滚动均值与波动');
      fsHost.innerHTML = fsChart(D.feature_sets, k);
    }
    setFs(D.feature_sets.length - 1);

    /* ---------- 图 4：特征重要性 ---------- */
    var c4 = el('div', 'sfw-lightgbm-card');
    c4.appendChild(html('div', 'sfw-lightgbm-hd', '<b>特征重要性 · Top 8</b><span>导出 split 计数在 Top 8 内的占比 %</span>'));
    var impHost = el('div', 'sfw-lightgbm-plot');
    impHost.innerHTML = impChart(D.feature_importance);
    c4.appendChild(impHost);
    var top2 = 0;
    D.feature_importance.slice(0, 2).forEach(function (x) { top2 += x.value; });
    var tot = 0;
    D.feature_importance.forEach(function (x) { tot += x.value; });
    c4.appendChild(html('p', 'sfw-cap',
      '前两项 ' + esc(String(D.feature_importance[0].name)) + ' 与 ' + esc(String(D.feature_importance[1].name)) +
      ' 就占了 ' + (tot > 0 ? Math.round(top2 / tot * 100) : 0) + '% 的重要性。树模型判断明天卖多少，几乎只看「过去卖了多少」；' +
      '日历特征只用来做微调。'));
    if (D.demo || demoFi) c4.appendChild(html('p', 'sfw-cap sfw-lightgbm-demo', demoCapText));
    cards.appendChild(c4);

    /* ---------- 事实与结论 ---------- */
    var full = D.feature_sets[D.feature_sets.length - 1];
    root.appendChild(cards);

    root.appendChild(html('dl', 'algo-facts',
      '<div><dt>论文</dt><dd>Ke et al. 2017, <i>LightGBM: A Highly Efficient Gradient Boosting Decision Tree</i> · NeurIPS · ' +
      '<a href="https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree" target="_blank" rel="noopener">论文原文 ↗</a></dd></div>' +
      '<div><dt>开源</dt><dd><a href="https://github.com/microsoft/LightGBM" target="_blank" rel="noopener">github.com/microsoft/LightGBM</a> · MIT</dd></div>' +
      '<div><dt>M5 战绩</dt><dd>M5 冠军方案的主力模型之一：全部 SKU 合成一张表做全局训练</dd></div>' +
      '<div><dt>本机复现</dt><dd>WAPE ' + pct(full.wape) + '（教学面板；完整基准见实测成绩）' + (D.demo || demoFs ? ' · 示意' : '') + '</dd></div>'));

    root.appendChild(html('p', 'algo-note',
      '该用：几十到几万个 SKU、有稳定 lag/rolling 特征、要秒级重训和特征重要性归因时，它通常就是性价比最高的第一选择。' +
      '不该用：需要概率区间时可用 LightGBM 分位数目标；本实现只训练点预测、序列极短或大量新品冷启动（没有 lag 可学）时，换 N-HiTS / DeepAR 更合适。'));
  }

  window.SF = window.SF || {};
  window.SF.algo = window.SF.algo || {};
  window.SF.algo[ID] = {
    id: ID,
    title: 'LightGBM',
    tagline: '为什么 M5 冠军用的就是它',
    mount: mount,
    _test: {
      demo: demoData, normalize: normalize, wapeAt: wapeAt,
      series: seriesChart, horizon: horizonChart, fs: fsChart, imp: impChart
    }
  };
})();
