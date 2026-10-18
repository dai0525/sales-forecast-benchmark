/* 销量预测工作台 · N-HiTS（N-BEATS 家族）交互讲解组件
   纯原生 JS + 内联 SVG，无框架 / 无 CDN / 无外链。
   数据：运行时 ctx.load('nhits.json')；拿不到就用内置示意数据，并在图下标注。 */
(function () {
  'use strict';

  var ID = 'nhits';

  var C = {
    blue: '#1D4ED8', blue2: '#3B82F6', blue3: '#60A5FA', blue4: '#BFD2F6',
    ink: '#0F1B2D', ink2: '#5A6B85', grey: '#8A97AB',
    hair: '#E6EBF3', line: '#D6E2FB', wash: '#EFF4FF'
  };
  /* 三个 block 的堆叠带：同一蓝色阶，从下到上由浅到深 */
  var B_FILL = ['#DCE7FD', '#AFC9F7', '#6E9CEC'];
  var B_STROKE = ['#1D4ED8', '#2563EB', '#1D4ED8'];

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
  function areaTo(pts, baseY) {
    if (!pts.length) return '';
    var d = pathOf(pts);
    return d + ' L' + r1(pts[pts.length - 1][0]) + ' ' + r1(baseY) + ' L' + r1(pts[0][0]) + ' ' + r1(baseY) + ' Z';
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

  /* ---------- 样式（全部 .sfw-nhits- 前缀） ---------- */
  function injectStyle() {
    if (document.getElementById('sfw-nhits-css')) return;
    var s = document.createElement('style');
    s.id = 'sfw-nhits-css';
    s.textContent = [
      '.sfw-nhits-card{border:1px solid var(--hair);border-radius:14px;padding:12px;margin:0 0 14px;background:#fff;min-width:0}',
      '.sfw-nhits-hd{display:flex;align-items:baseline;justify-content:space-between;gap:8px;flex-wrap:wrap;margin:0 0 8px}',
      '.sfw-nhits-hd b{font-size:14px;font-weight:700;color:var(--ink)}',
      '.sfw-nhits-hd span{font-size:12px;color:var(--ink-2)}',
      '.sfw-nhits-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:2px 0 4px}',
      '.sfw-nhits-lab{font-size:13px;font-weight:600;color:var(--ink);white-space:nowrap}',
      '.sfw-nhits-range{flex:1 1 120px;min-width:110px;height:40px;margin:0;touch-action:pan-y}',
      '.sfw-nhits-hval{font:700 15px/1 ui-monospace,monospace;color:var(--blue);min-width:42px;text-align:right}',
      '.sfw-nhits-metricrow{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:2px 0 6px}',
      '.sfw-nhits-sub{font-size:12px;color:var(--ink-2);line-height:1.45;flex:1 1 150px;min-width:0}',
      '.sfw-nhits-demo{margin-top:5px;background:var(--wash);border-radius:7px;padding:5px 8px;color:#3E5A8A}',
      '.sfw-nhits-human{margin:8px 0 0;font-size:13px;line-height:1.55;color:var(--ink);background:var(--wash);border-left:3px solid var(--blue);padding:8px 10px;border-radius:0 8px 8px 0}',
      '.sfw-nhits-human b{color:var(--blue)}',
      '.sfw-nhits-rates{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0 8px}',
      '.sfw-nhits-rate{border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:#fff;min-width:0}',
      '.sfw-nhits-rate.is-on{background:var(--wash);border-color:var(--blue)}',
      '.sfw-nhits-ratev{display:block;font:800 21px/1.15 ui-monospace,monospace;color:#9AA9BE;font-variant-numeric:tabular-nums}',
      '.sfw-nhits-rate.is-on .sfw-nhits-ratev{color:var(--blue)}',
      '.sfw-nhits-ratel{display:block;font-size:11.5px;line-height:1.35;color:var(--ink-2);margin-top:3px}',
      '.sfw-nhits-chips{margin:0 0 8px}',
      '.sfw-nhits-chips .sfw-chip{min-height:42px;padding:9px 13px}',
      '.sfw-nhits-cards{display:grid;grid-template-columns:1fr;gap:0;min-width:0}',
      '.sfw-nhits-plot{margin:6px 0 0;min-width:0;max-width:480px}',
      '@media (min-width:720px){.sfw-nhits-cards{grid-template-columns:1fr 1fr;gap:14px;align-items:start}.sfw-nhits-cards>.sfw-nhits-card{margin:0}}'
    ].join('');
    (document.head || document.documentElement).appendChild(s);
  }

  /* ---------- 示意数据 ---------- */
  function rng(seed) {
    var s = seed >>> 0;
    return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  function demoData() {
    var r = rng(7741);
    var dow = [0.93, 1.0, 0.98, 1.05, 1.13, 1.24, 1.16];
    var ytr = [], i, v;
    for (i = 0; i < 120; i++) {
      v = (176 + i * 0.42 + 9 * Math.sin(i / 9.5) + 5 * Math.cos(i / 23)) * dow[i % 7];
      ytr.push(Math.max(1, Math.round(v * (1 + (r() - 0.5) * 0.07))));
    }
    // 未来 14 步 = 三个 block 的输出之和
    var lvl0 = 176 + 119 * 0.42;              // 训练末尾的水平
    var mean = dow.reduce(function (a, b) { return a + b; }, 0) / 7;
    var t1 = [], s2 = [], s3 = [];
    for (i = 0; i < 14; i++) {
      t1.push(lvl0 * mean * (1 + 0.012 * i));                 // block1：水平 + 缓慢斜率
      s2.push((dow[(120 + i) % 7] / mean - 1) * lvl0 * mean);  // block2：周内周期
      s3.push(9.5 * Math.sin((120 + i) / 1.8 + 0.7));          // block3：短周期波动
    }
    var pred = [], truth = [];
    for (i = 0; i < 14; i++) {
      pred.push(t1[i] + s2[i] + s3[i]);
      truth.push(Math.max(1, Math.round(pred[i] + (r() - 0.5) * 44)));
    }
    var predR = pred.map(function (x) { return Math.round(x); });
    var w1 = wapeAt(predR, truth, 14);
    return {
      series_id: null, demo: true,
      y_train: ytr, y_test: truth, pred: predR,
      blocks: [
        { name: 'block 1 · 水平/长期趋势', trend: t1, season: s2.map(function () { return 0; }) },
        { name: 'block 2 · 周周期', trend: s2.map(function () { return 0; }), season: s2 },
        { name: 'block 3 · 短波动', trend: s3.map(function () { return 0; }), season: s3 }
      ],
      multi_rate: [
        { n_pool: 1, wape: w1 },
        { n_pool: 2, wape: w1 * 1.19 + 0.004 }
      ]
    };
  }

  function normalize(d) {
    var blocks = (Array.isArray(d.blocks) ? d.blocks : []).filter(function (b) {
      return b && (arr(b.trend).length || arr(b.season).length);
    }).slice(0, 3);
    return {
      series_id: d.series_id || null,
      demo: false,
      y_train: arr(d.y_train),
      y_test: arr(d.y_test),
      pred: arr(d.pred),
      blocks: blocks,
      multi_rate: (Array.isArray(d.multi_rate) ? d.multi_rate : [])
        .filter(function (x) { return x && isNum(x.wape) && isNum(x.n_pool); })
        .sort(function (a, b) { return a.n_pool - b.n_pool; })
    };
  }

  function wapeAt(pred, truth, h) {
    var a = 0, b = 0, n = Math.min(h, pred.length, truth.length), i;
    for (i = 0; i < n; i++) { a += Math.abs(pred[i] - truth[i]); b += Math.abs(truth[i]); }
    return b > 0 ? a / b : null;
  }

  function blockTotal(b, L) {
    var tr = arr(b && b.trend), se = arr(b && b.season), out = [], i;
    for (i = 0; i < L; i++) out.push((tr[i] || 0) + (se[i] || 0));
    return out;
  }

  function horizonOf(D) {
    var n = arr(D.pred).length;
    (D.blocks || []).forEach(function (b) {
      n = Math.max(n, arr(b.trend).length, arr(b.season).length);
    });
    return Math.min(Math.max(n, 1), 60);
  }

  /* ---------- 图 1：历史 + 真值 + N-HiTS 预测 ---------- */
  function seriesChart(D) {
    var W = 332, H = 190, pl = 30, pr = 8, pt = 12, pb = 24;
    var tr = arr(D.y_train), te = arr(D.y_test), pd = arr(D.pred);
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
    var all = tr.concat(te, pd);
    var mn = Math.min.apply(null, all), mx = Math.max.apply(null, all);
    var pad = Math.max(10, (mx - mn) * 0.12);
    mn -= pad; mx += pad;
    var y = lin(mn, mx, H - pb, pt);
    var o = [];

    o.push(gridY(pl, W - pr, pt, H - pb, mn, mx));
    if (cut > 0) {
      o.push('<line x1="' + r1(x(cut - 1)) + '" y1="' + pt + '" x2="' + r1(x(cut - 1)) + '" y2="' + (H - pb) + '" stroke="' + C.line + '" stroke-width="1" stroke-dasharray="3 3"/>');
    }
    // 历史超过 240 点时按原始索引等距抽稀（契约要求 >300 点降采样）
    o.push('<path d="' + pathOf(ptsOf(tr, x, y)) + '" fill="none" stroke="' + C.ink2 + '" stroke-width="1.2" stroke-linejoin="round"/>');

    var tp = cut ? [[x(cut - 1), y(tr[cut - 1])]] : [];
    te.forEach(function (v, i) { tp.push([x(cut + i), y(v)]); });
    o.push('<path d="' + pathOf(tp) + '" fill="none" stroke="' + C.grey + '" stroke-width="1.7" stroke-dasharray="4 3" stroke-linejoin="round"/>');

    if (pd.length) {
      var pp = cut ? [[x(cut - 1), y(tr[cut - 1])]] : [];
      pd.forEach(function (v, i) { pp.push([x(cut + i), y(v)]); });
      o.push('<path d="' + pathOf(pp) + '" fill="none" stroke="' + C.blue + '" stroke-width="2.2" stroke-linejoin="round"/>');
      var last = pp[pp.length - 1];
      o.push('<circle cx="' + r1(last[0]) + '" cy="' + r1(last[1]) + '" r="2.8" fill="' + C.blue + '"/>');
    }
    o.push('<text x="' + r1(pl + histW / 2) + '" y="' + (H - 8) + '" text-anchor="middle">' + cut + ' 天历史</text>');
    o.push('<text x="' + r1(pl + histW + futW / 2) + '" y="' + (H - 8) + '" text-anchor="middle">未来 ' + te.length + ' 步</text>');
    return svgWrap(W, H, o.join(''), 'N-HiTS 在同一序列上的预测');
  }

  /* ---------- 图 2：多尺度堆叠（block 数可调） ---------- */
  function stackChart(D, nAct) {
    var W = 332, H = 212, pl = 30, pr = 8, pt = 16, pb = 24;
    var blocks = (D.blocks || []).slice(0, 3);
    if (!blocks.length) return '';
    var L = horizonOf(D);
    var x = lin(0, Math.max(1, L - 1), pl, W - pr);
    var total = blocks.map(function (b) { return blockTotal(b, L); });
    var cum = [], k, i;
    for (k = 0; k < total.length; k++) {
      var row = [];
      for (i = 0; i < L; i++) row.push((k ? cum[k - 1][i] : 0) + total[k][i]);
      cum.push(row);
    }
    var te = arr(D.y_test).slice(0, L);
    var pool = [].concat(cum[cum.length - 1], te);
    var mn = Math.min.apply(null, pool), mx = Math.max.apply(null, pool);
    var pad = Math.max(6, (mx - mn) * 0.16);
    mn -= pad; mx += pad;
    var y = lin(mn, mx, H - pb, pt);
    var o = [];

    o.push(gridY(pl, W - pr, pt, H - pb, mn, mx));

    var n = clamp(nAct | 0, 1, cum.length);
    var baseY = H - pb;
    // 由下往上铺带：带的上沿是累计贡献，带厚 = 该 block 的贡献
    for (k = 0; k < n; k++) {
      var topPts = cum[k].map(function (v, idx) { return [x(idx), y(v)]; });
      var band;
      if (k === 0) {
        // 最底层铺到绘图区底边（纵轴已裁剪，不画到 0）
        band = pathOf(topPts) + ' L' + r1(x(L - 1)) + ' ' + r1(baseY) + ' L' + r1(x(0)) + ' ' + r1(baseY) + ' Z';
      } else {
        var botPts = cum[k - 1].map(function (v, idx) { return [x(idx), y(v)]; });
        band = pathOf(topPts) + ' ' + pathOf(botPts.slice().reverse()).replace(/^M/, 'L') + ' Z';
      }
      o.push('<path d="' + band + '" fill="' + B_FILL[k] + '" stroke="none"/>');
      o.push('<path d="' + pathOf(topPts) + '" fill="none" stroke="' + B_STROKE[k] + '" stroke-width="1.3"/>');
    }
    // 当前叠加结果（加粗）
    var curPts = cum[n - 1].map(function (v, idx) { return [x(idx), y(v)]; });
    o.push('<path d="' + pathOf(curPts) + '" fill="none" stroke="' + C.blue + '" stroke-width="2.4" stroke-linejoin="round"/>');
    curPts.forEach(function (p) {
      o.push('<circle cx="' + r1(p[0]) + '" cy="' + r1(p[1]) + '" r="1.9" fill="' + C.blue + '"/>');
    });
    // 真值
    if (te.length) {
      o.push('<path d="' + pathOf(te.map(function (v, idx) { return [x(idx), y(v)]; })) + '" fill="none" stroke="' + C.grey + '" stroke-width="1.6" stroke-dasharray="4 3" stroke-linejoin="round"/>');
    }
    [0, Math.floor((L - 1) / 2), L - 1].forEach(function (idx, j) {
      o.push('<text x="' + r1(x(idx)) + '" y="' + (H - 8) + '" text-anchor="' + (j === 0 ? 'start' : (j === 2 ? 'end' : 'middle')) + '">h=' + (idx + 1) + '</text>');
    });
    return svgWrap(W, H, o.join(''), 'N-HiTS 多尺度 block 堆叠分解');
  }

  /* ---------- 图 3：multi-rate pooling 对比 ---------- */
  function poolShape(v) {
    var out = [], n = v.length, i;
    for (i = 0; i < n; i++) {
      var a = v[Math.max(0, i - 1)], b = v[Math.min(n - 1, i + 1)];
      var j = Math.floor(i / 2) * 2;
      var pair = (v[j] + v[Math.min(n - 1, j + 1)]) / 2;
      out.push(0.55 * ((a + 2 * v[i] + b) / 4) + 0.45 * pair);
    }
    return out;
  }

  function rateChart(D, pool) {
    var W = 332, H = 174, pl = 30, pr = 8, pt = 14, pb = 24;
    var pd = arr(D.pred), te = arr(D.y_test).slice(0, Math.max(pd.length, 1));
    if (!pd.length) return '';
    var L = pd.length;
    var p2 = poolShape(pd);
    var x = lin(0, Math.max(1, L - 1), pl, W - pr);
    var all = pd.concat(p2, te);
    var mn = Math.min.apply(null, all), mx = Math.max.apply(null, all);
    var pad = Math.max(8, (mx - mn) * 0.16);
    mn -= pad; mx += pad;
    var y = lin(mn, mx, H - pb, pt);
    var o = [];

    o.push(gridY(pl, W - pr, pt, H - pb, mn, mx));
    if (te.length) {
      o.push('<path d="' + pathOf(te.map(function (v, i) { return [x(i), y(v)]; })) + '" fill="none" stroke="' + C.grey + '" stroke-width="1.6" stroke-dasharray="4 3" stroke-linejoin="round"/>');
    }
    // 两条曲线常驻，当前选中的加粗
    o.push('<path d="' + pathOf(p2.map(function (v, i) { return [x(i), y(v)]; })) + '" fill="none" stroke="' + C.blue3 + '" stroke-width="' + (pool === 2 ? 2.6 : 1.4) + '" opacity="' + (pool === 2 ? 1 : 0.55) + '" stroke-linejoin="round"/>');
    o.push('<path d="' + pathOf(pd.map(function (v, i) { return [x(i), y(v)]; })) + '" fill="none" stroke="' + C.blue + '" stroke-width="' + (pool === 1 ? 2.6 : 1.4) + '" opacity="' + (pool === 1 ? 1 : 0.55) + '" stroke-linejoin="round"/>');
    [0, Math.floor((L - 1) / 2), L - 1].forEach(function (idx, j) {
      o.push('<text x="' + r1(x(idx)) + '" y="' + (H - 8) + '" text-anchor="' + (j === 0 ? 'start' : (j === 2 ? 'end' : 'middle')) + '">h=' + (idx + 1) + '</text>');
    });
    return svgWrap(W, H, o.join(''), 'pool=1 与 pool=2 的预测形状对比');
  }

  /* ---------- 图 4：层级分解示意（低频 + 中频 + 高频） ---------- */
  function decompSvg() {
    var W = 332, H = 172, x0 = 28, x1 = 296, N = 48;
    var x = lin(0, N - 1, x0, x1);
    var lo = [], mid = [], hi = [], sum = [], i;
    for (i = 0; i < N; i++) {
      var a = Math.sin(2 * Math.PI * i / 48);
      var b = 0.4 * Math.sin(2 * Math.PI * i / 12 + 0.8);
      var c = 0.16 * Math.sin(2 * Math.PI * i / 4.2 + 1.6);
      lo.push(a); mid.push(b); hi.push(c); sum.push(a + b + c);
    }
    var sy = lin(-1.6, 1.6, 76, 26);
    var sp = sum.map(function (v, k) { return [x(k), sy(v)]; });
    var o = [];
    o.push('<text x="' + x0 + '" y="14" text-anchor="start">低频 + 中频 + 高频 = 这条销量曲线</text>');
    o.push('<path d="' + areaTo(sp, 76) + '" fill="' + C.wash + '" stroke="none"/>');
    o.push('<path d="' + pathOf(sp) + '" fill="none" stroke="' + C.blue + '" stroke-width="2" stroke-linejoin="round"/>');
    o.push('<line class="grid" x1="' + x0 + '" y1="76" x2="' + x1 + '" y2="76"/>');
    o.push('<text x="' + (x1 + 8) + '" y="48" text-anchor="start">合计</text>');

    var rows = [
      { k: '低频', v: lo, col: C.blue, by: 106, T: 48 },
      { k: '中频', v: mid, col: C.blue2, by: 132, T: 12 },
      { k: '高频', v: hi, col: C.blue3, by: 158, T: 4 }
    ];
    rows.forEach(function (row) {
      // 每行按自身幅度归一化到 ±10px（幅度差异写在图注里），否则高频那条会细成一条直线
      var m = 1e-9;
      row.v.forEach(function (v) { if (Math.abs(v) > m) m = Math.abs(v); });
      var pts = row.v.map(function (v, k) { return [x(k), row.by - (v / m) * 10]; });
      o.push('<line class="grid" x1="' + x0 + '" y1="' + row.by + '" x2="' + x1 + '" y2="' + row.by + '"/>');
      o.push('<path d="' + pathOf(pts) + '" fill="none" stroke="' + row.col + '" stroke-width="1.7" stroke-linejoin="round"/>');
      o.push('<text x="0" y="' + (row.by + 4) + '" text-anchor="start">' + row.k + '</text>');
      o.push('<text x="' + (x1 + 8) + '" y="' + (row.by + 4) + '" text-anchor="start">T=' + row.T + '</text>');
    });
    return svgWrap(W, H, o.join(''), '销量拆成低频、中频、高频三条曲线');
  }

  /* ---------- 挂载 ---------- */
  async function mount(root, ctx) {
    injectStyle();

    var raw = (ctx && ctx.data && ctx.data[ID]) || null;
    if (!raw && ctx && typeof ctx.load === 'function') {
      try { raw = await ctx.load('nhits.json'); } catch (e) { raw = null; }
    }
    var real = !!(raw && arr(raw.y_train).length >= 2 && (arr(raw.pred).length || arr(raw.y_test).length));
    var D = real ? normalize(raw) : demoData();
    if (!D.pred.length) D = demoData();
    var isDemo = D.demo;
    var demoCapText = '示意数据，真实结果生成后自动替换。';

    root.appendChild(html('p', 'algo-lead',
      'N-HiTS 属于 <b>N-BEATS</b> 家族：它不把序列当「时间」看，而是直接输出一整段未来值，' +
      '再把这段值拆成<b>若干个 block 的叠加</b>——浅层 block 学平缓的长期趋势，深层 block 学越来越短的周期波动。' +
      '每个 block 就是一个很小的 MLP，靠「残差堆叠」一层层补上前面没拟合的部分。' +
      'N-HiTS 相对 N-BEATS 只多两件事：<b>多速率输入采样</b>（不同 block 看不同分辨率的输入）和<b>层级插值</b>（粗粒度预测再插值到细粒度），换来长 horizon 上更少的算力。'));

    /* ---------- 例子 1 ---------- */
    var cards = el('div', 'sfw-nhits-cards');
    var c1 = el('div', 'sfw-nhits-card');
    c1.appendChild(html('div', 'sfw-nhits-hd',
      '<b>例子 1 · ' + (D.demo ? '一条日销量序列（示意）' : '一条真实日销量序列') + '</b><span>' + (D.series_id ? esc(D.series_id) : 'M4 · Daily') + ' · 训练 ' + arr(D.y_train).length + ' 天 / 预测 ' + arr(D.y_test).length + ' 天</span>'));
    c1.appendChild(html('div', 'sfw-legend',
      '<span><i style="background:' + C.ink2 + '"></i>历史</span>' +
      '<span><i style="background:' + C.grey + '"></i>未来真值</span>' +
      '<span><i style="background:' + C.blue + '"></i>N-HiTS 预测</span>'));
    var s1 = el('div', 'sfw-nhits-plot');
    s1.innerHTML = seriesChart(D);
    c1.appendChild(s1);
    c1.appendChild(html('p', 'sfw-cap',
      '整段 14 步一次性输出（direct multi-horizon），所以不存在 LightGBM 那种「把预测喂回特征」的递归累积。'));
    if (D.demo) c1.appendChild(html('p', 'sfw-cap sfw-nhits-demo', demoCapText));
    cards.appendChild(c1);

    /* ---------- 例子 2：多尺度分解（核心交互） ---------- */
    var maxB = Math.min(3, Math.max(1, (D.blocks || []).length));
    var c2 = el('div', 'sfw-nhits-card');
    c2.appendChild(html('div', 'sfw-nhits-hd', '<b>例子 2 · 多尺度分解：预测是三个 block 叠出来的</b><span>拖动滑块改叠加层数</span>'));

    var row = el('div', 'sfw-nhits-row');
    row.appendChild(html('label', 'sfw-nhits-lab', '参与叠加的 block'));
    var bIn = el('input', 'sfw-slider sfw-nhits-range');
    bIn.type = 'range'; bIn.min = '1'; bIn.max = String(maxB); bIn.step = '1';
    bIn.value = String(maxB);
    bIn.setAttribute('aria-label', '参与叠加的 block 数');
    row.appendChild(bIn);
    var bVal = el('span', 'sfw-nhits-hval');
    row.appendChild(bVal);
    c2.appendChild(row);

    var mrow = el('div', 'sfw-nhits-metricrow');
    var big = el('span', 'sfw-metric');
    mrow.appendChild(big);
    var sub = el('span', 'sfw-nhits-sub');
    mrow.appendChild(sub);
    c2.appendChild(mrow);

    var legend = el('div', 'sfw-legend');
    (D.blocks || []).slice(0, 3).forEach(function (b, k) {
      legend.appendChild(html('span', '', '<i style="background:' + B_FILL[k] + ';border:1px solid ' + B_STROKE[k] + '"></i>' + esc(shortName(b.name, k))));
    });
    legend.appendChild(html('span', '', '<i style="background:' + C.grey + '"></i>真值'));
    c2.appendChild(legend);

    var s2 = el('div', 'sfw-nhits-plot');
    c2.appendChild(s2);

    c2.appendChild(html('div', 'sfw-nhits-human',
      '<b>浅层 block 管长期趋势，深层 block 管短周期波动。</b>只给第一层，是一条几乎没有周内起伏的斜坡；叠上第二层才出现「周末高、周一低」；' +
      '第三层补的是最后那点锯齿。真实 N-HiTS 里这三层不是手工设计的，而是三个 MLP 在训练中自己分工出来的。'));
    c2.appendChild(html('p', 'sfw-cap',
      '纵轴裁剪到数据区间（不从 0 起），读的是「叠加后的形状」：每条带子的厚度就是该 block 这一层贡献的销量。'));
    if (D.demo) c2.appendChild(html('p', 'sfw-cap sfw-nhits-demo', demoCapText));
    cards.appendChild(c2);

    var L = horizonOf(D);
    var fullW = wapeAt(arr(D.pred), arr(D.y_test), L);
    function blockCum(k) {
      var tot = (D.blocks || []).slice(0, 3).map(function (b) { return blockTotal(b, L); });
      var out = [], i, j;
      for (j = 0; j < k; j++) {
        var row2 = [];
        for (i = 0; i < L; i++) row2.push((j ? out[j - 1][i] : 0) + tot[j][i]);
        out.push(row2);
      }
      return out.length ? out[out.length - 1] : [];
    }
    function updateB(n) {
      n = clamp(Math.round(n), 1, maxB);
      bVal.textContent = n + ' / ' + maxB;
      s2.innerHTML = stackChart(D, n);
      var cum = blockCum(n);
      var w = wapeAt(cum, arr(D.y_test), L);
      big.innerHTML = (isNum(w) ? (w * 100).toFixed(1) : '–') + '<span class="sfw-metric-u">% WAPE · 叠 ' + n + ' 层</span>';
      var gap = (isNum(w) && isNum(fullW)) ? (w - fullW) * 100 : null;
      sub.textContent = n === maxB
        ? '完整预测 ' + pct(fullW) + ' · 三层全部参与'
        : '完整预测 ' + pct(fullW) + ' · 还差 ' + (gap == null ? '–' : '+' + gap.toFixed(1) + 'pp');
    }
    bIn.addEventListener('input', function () { updateB(+bIn.value); });
    updateB(maxB);

    /* ---------- 例子 3：multi-rate pooling ---------- */
    var rates = (D.multi_rate || []).slice(0, 2);
    if (!rates.length) rates = [{ n_pool: 1, wape: wapeAt(arr(D.pred), arr(D.y_test), L) }];
    var c3 = el('div', 'sfw-nhits-card');
    c3.appendChild(html('div', 'sfw-nhits-hd', '<b>例子 3 · multi-rate pooling：看粗一点，省算力但丢细节</b><span>点 chip 切换池化比</span>'));
    var chips = html('div', 'sfw-chips sfw-nhits-chips', '');
    var cbtns = [];
    rates.forEach(function (rt, k) {
      var b = el('button', 'sfw-chip');
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      b.textContent = 'pool = ' + rt.n_pool;
      b.addEventListener('click', function () { setRate(k); });
      chips.appendChild(b);
      cbtns.push(b);
    });
    c3.appendChild(chips);

    var cells = el('div', 'sfw-nhits-rates');
    var cellv = [];
    rates.forEach(function (rt, k) {
      var cell = el('div', 'sfw-nhits-rate');
      var vv = el('span', 'sfw-nhits-ratev');
      vv.textContent = pct(rt.wape);
      cell.appendChild(vv);
      cell.appendChild(html('span', 'sfw-nhits-ratel', 'pool = ' + rt.n_pool + ' · ' + (Number(rt.n_pool) > 1 ? '输入下采样 ' + rt.n_pool + '×' : '原始分辨率')));
      cells.appendChild(cell);
      cellv.push({ cell: cell, v: vv });
    });
    c3.appendChild(cells);

    var rateSub = el('p', 'sfw-cap');
    c3.appendChild(rateSub);

    var s3 = el('div', 'sfw-nhits-plot');
    c3.appendChild(s3);
    c3.appendChild(html('p', 'sfw-cap',
      'pool=2 先把输入两两合并再上采样回来，序列变短、算力更省，但周内尖峰被抹平了——曲线更「顺」，代价直接写在 WAPE 上。' +
      '曲线是按池化比重采样的示形，WAPE 来自保存结果或已标注的示意值，曲线不是该池化参数的重新训练输出。'));
    if (D.demo) c3.appendChild(html('p', 'sfw-cap sfw-nhits-demo', demoCapText));
    cards.appendChild(c3);

    function setRate(k) {
      k = clamp(k | 0, 0, rates.length - 1);
      cbtns.forEach(function (b, j) {
        b.classList.toggle('is-on', j === k);
        b.setAttribute('aria-pressed', String(j === k));
      });
      cellv.forEach(function (c, j) { c.cell.classList.toggle('is-on', j === k); });
      var cur = rates[k], base = rates[0];
      var dpp = (cur.wape - base.wape) * 100;
      rateSub.textContent = k === 0
        ? '当前 pool=' + cur.n_pool + '：WAPE ' + pct(cur.wape) + '，用原始分辨率作基准'
        : '当前 pool=' + cur.n_pool + '：WAPE ' + pct(cur.wape) + '，比 pool=' + base.n_pool + ' ' +
          (dpp >= 0 ? '差 +' : '好 −') + Math.abs(dpp).toFixed(1) + 'pp';
      var pool = Number(cur.n_pool) || 1;
      s3.innerHTML = rateChart(D, pool === 2 ? 2 : 1);
    }
    setRate(0);

    /* ---------- 例子 4：层级分解示意 ---------- */
    var c4 = el('div', 'sfw-nhits-card');
    c4.appendChild(html('div', 'sfw-nhits-hd', '<b>层级分解 · 把销量拆成三条正弦</b><span>低频 → 中频 → 高频</span>'));
    var s4 = el('div', 'sfw-nhits-plot');
    s4.innerHTML = decompSvg();
    c4.appendChild(s4);
    c4.appendChild(html('p', 'sfw-cap',
      '三个分量的幅度比大约是 1 : 0.4 : 0.16：低频决定这条序列的「水平」在哪，中频给出一周内的起伏，高频只负责最后那点锯齿。' +
      'N-BEATS / N-HiTS 就是学这组基函数，只不过不是写死正弦，而是用 MLP 学出来的——所以它也能学出促销、节假日这种非正弦的形状。'));
    cards.appendChild(c4);

    /* ---------- 事实与结论 ---------- */
    var best = rates.reduce(function (a, b) { return (b.wape < a.wape ? b : a); }, rates[0]);
    root.appendChild(cards);

    root.appendChild(html('dl', 'algo-facts',
      '<div><dt>论文</dt><dd>Oreshkin et al. 2020, <i>N-BEATS: Neural basis expansion analysis for interpretable time series forecasting</i> · ICLR · ' +
      '<a href="https://arxiv.org/abs/1905.10437" target="_blank" rel="noopener">arXiv:1905.10437</a></dd></div>' +
      '<div><dt>N-HiTS</dt><dd>Challu et al. 2023, <i>Neural Hierarchical Interpolation for Time Series Forecasting</i> · AAAI · ' +
      '<a href="https://arxiv.org/abs/2201.12886" target="_blank" rel="noopener">arXiv:2201.12886</a></dd></div>' +
      '<div><dt>开源</dt><dd><a href="https://github.com/Nixtla/neuralforecast" target="_blank" rel="noopener">github.com/Nixtla/neuralforecast</a> · Apache-2.0</dd></div>' +
      '<div><dt>本机复现</dt><dd>WAPE ' + pct(arr(D.pred).length ? wapeAt(arr(D.pred), arr(D.y_test), L) : null) +
      ' · 最优 pool=' + best.n_pool + (D.demo ? ' · 示意' : '') + '</dd></div>'));

    root.appendChild(html('p', 'algo-note',
      '该用：序列多、预测步长较长（几十步以上）、需要概率区间和冷启动友好（纯历史输入，不依赖手工 lag 特征）。' +
      '不该用：只有几百条序列、要求毫秒级推理或强可解释性时，LightGBM + lag/rolling 往往更划算——神经网络的优势要在数据量和 horizon 都上去之后才显现。'));
  }

  function shortName(n, k) {
    var s = String(n == null ? '' : n);
    if (s.length > 18) s = s.slice(0, 17) + '…';
    return s || ('block ' + (k + 1));
  }

  window.SF = window.SF || {};
  window.SF.algo = window.SF.algo || {};
  window.SF.algo[ID] = {
    id: ID,
    title: 'N-HiTS',
    tagline: '把一条曲线拆成趋势 + 多尺度周期，用 MLP 学出来',
    mount: mount,
    _test: {
      demo: demoData, normalize: normalize, wapeAt: wapeAt,
      series: seriesChart, stack: stackChart, rate: rateChart, decomp: decompSvg
    }
  };
})();
