/* PatchTST：patch 化 + 通道独立 */
(function () {
  'use strict';

  var ID = 'patchtst';
  var C = {
    blue: '#1D4ED8', b3: '#60A5FA', b4: '#BFD2F6', wash: '#EFF4FF', line: '#D6E2FB',
    ink: '#0F1B2D', ink2: '#5A6B85', hair: '#E6EBF3', truth: '#8A97AB'
  };

  var CSS = [
    '.sfw-patchtst-sub2{font-size:11px;color:' + C.ink2 + ';margin:0 0 6px;font-weight:700;letter-spacing:.3px}',
    '.sfw-patchtst-block{margin-top:16px}',
    '.sfw-patchtst-num{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}',
    '.sfw-patchtst-num>div{flex:1 1 110px;min-width:100px;background:' + C.wash + ';border:1px solid ' + C.line + ';border-radius:10px;padding:8px 10px}',
    '.sfw-patchtst-num .sfw-metric{font-size:19px}',
    '.sfw-patchtst-numlab{font-size:11px;color:' + C.ink2 + ';margin-top:2px}',
    '.sfw-patchtst-ctl{margin-top:10px}',
    '.sfw-patchtst-ctl label{display:block;font-size:12px;color:' + C.ink2 + ';font-weight:600}',
    '.sfw-patchtst-ctl b{color:' + C.blue + ';font-size:13px}',
    '.sfw-patchtst-ctl input[type=range]{height:40px}',
    '.sfw-patchtst-note{font-size:12px;color:' + C.ink2 + ';margin:8px 0 0;min-height:34px}'
  ].join('');

  function injectCSS() {
    if (document.getElementById('sfw-' + ID + '-css')) return;
    var s = document.createElement('style');
    s.id = 'sfw-' + ID + '-css';
    s.textContent = CSS;
    document.head.appendChild(s);
  }
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    var u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function fx(v) { return (Math.round(v * 100) / 100).toString(); }
  function i0(v) { return Math.round(v); }

  var CURVE = [
    { patch_len: 4, wape: 0.142 }, { patch_len: 8, wape: 0.118 },
    { patch_len: 12, wape: 0.104 }, { patch_len: 16, wape: 0.0965 },
    { patch_len: 20, wape: 0.0975 }, { patch_len: 24, wape: 0.099 },
    { patch_len: 32, wape: 0.108 }
  ];

  /* ---------- 内置示意数据（页面上已标「示意数据」） ---------- */
  function synth() {
    var r = rng(5150), rp = rng(8080), W = [0.86, 0.82, 0.88, 0.95, 1.11, 1.26, 1.02];
    function base(t) { return 74 * (1 + 0.22 * t / 134) * W[t % 7]; }
    var train = [], test = [], pred = [], t, h;
    for (t = 0; t < 120; t++) train.push(i0(base(t) * (1 + 0.09 * gauss(r))));
    for (h = 0; h < 14; h++) {
      test.push(i0(base(120 + h) * (1 + 0.095 * gauss(r))));
      pred.push(Math.round(base(120 + h) * (1 + 0.015 * Math.cos(h / 3)) * (1 + 0.045 * gauss(rp))));
    }
    return {
      series_id: '示意序列 P-1', y_train: train, y_test: test, pred: pred,
      patch_len: 16, stride: 8, context: 512, d_model: 128, patch_attention: null,
      patch_len_curve: CURVE, fake: true
    };
  }

  function normalize(raw) {
    var ok = raw && raw.y_train && raw.y_train.length > 48 && raw.y_test && raw.y_test.length &&
      raw.pred && raw.pred.length === raw.y_test.length;
    if (!ok) return synth();
    raw.fake = false;
    if (!raw.stride) raw.stride = 8;
    if (!raw.patch_len) raw.patch_len = 16;
    if (!raw.patch_len_curve || raw.patch_len_curve.length < 3) raw.patch_len_curve = CURVE;
    if (!raw.context) raw.context = 512;
    return raw;
  }

  /* 连续重要性曲线 → 任意 patch 数的权重（同一套模式的确定性采样） */
  function imp(u) {
    var a = 0.30 * Math.exp(-Math.pow((u - 0.12) / 0.13, 2));
    var b = 1.00 * Math.exp(-Math.pow((u - 0.62) / 0.075, 2));
    var c = 0.45 * Math.exp(-Math.pow((u - 0.90) / 0.10, 2));
    var d = 0.35 * Math.exp(-Math.pow((u - 0.36) / 0.11, 2));
    return a + b + c + d + 0.10;
  }
  function attentionFor(n) {
    var raw = [], i;
    for (i = 0; i < n; i++) raw.push(imp(n === 1 ? 0.5 : i / (n - 1)));
    var s = raw.reduce(function (x, y) { return x + y; }, 0) || 1;
    return raw.map(function (v) { return v / s; });
  }

  /* ---------- 纯几何：序列图 ---------- */
  function buildSeries(d) {
    var W = 340, H = 186, L = 32, R = 8, T = 12, B = 26;
    var hr = d.y_test.length;
    var hist = d.y_train.slice(Math.max(0, d.y_train.length - 60));
    var n = hist.length + hr, pts = [], out = [];
    var lo = Infinity, hi = -Infinity;
    function scan(v) { if (v < lo) lo = v; if (v > hi) hi = v; }
    hist.forEach(scan); d.y_test.forEach(scan); d.pred.forEach(scan);
    if (!isFinite(lo) || !isFinite(hi) || hi - lo < 1e-9) { lo -= 1; hi += 1; }
    var vp = (hi - lo) * 0.09; lo -= vp; hi += vp;
    var pw = W - L - R, ph = H - T - B;
    function X(i) { return L + pw * i / (n - 1); }
    function Y(v) { return T + ph * (1 - (v - lo) / (hi - lo)); }
    function P(x, y) { pts.push([x, y]); return fx(x) + ',' + fx(y); }
    var io = hist.length - 1, last = hist[io];

    out.push('<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="PatchTST 预测与真值对比">');
    var g, gy, gv;
    for (g = 0; g < 3; g++) {
      gv = lo + (hi - lo) * (g + 1) / 4;
      gy = Y(gv);
      out.push('<line class="grid" x1="' + L + '" y1="' + fx(gy) + '" x2="' + (W - R) + '" y2="' + fx(gy) + '"/>');
      out.push('<text x="' + (L - 5) + '" y="' + fx(gy + 3.5) + '" text-anchor="end">' + i0(gv) + '</text>');
    }
    var hs = [];
    hist.forEach(function (v, i) { hs.push((i ? 'L' : 'M') + P(X(i), Y(v))); });
    out.push('<path d="' + hs.join(' ') + '" fill="none" stroke="' + C.ink2 + '" stroke-width="1.4" stroke-opacity="0.75" stroke-linejoin="round"/>');
    var ts = ['M' + P(X(io), Y(last))];
    for (g = 0; g < hr; g++) ts.push('L' + P(X(io + 1 + g), Y(d.y_test[g])));
    out.push('<path d="' + ts.join(' ') + '" fill="none" stroke="' + C.truth + '" stroke-width="1.6" stroke-dasharray="4 3"/>');
    var ps = ['M' + P(X(io), Y(last))];
    for (g = 0; g < hr; g++) ps.push('L' + P(X(io + 1 + g), Y(d.pred[g])));
    out.push('<path d="' + ps.join(' ') + '" fill="none" stroke="' + C.blue + '" stroke-width="2.2" stroke-linejoin="round"/>');
    out.push('<line x1="' + fx(X(io)) + '" y1="' + T + '" x2="' + fx(X(io)) + '" y2="' + (T + ph) + '" stroke="' + C.line + '" stroke-dasharray="3 3"/>');
    out.push('<text x="' + L + '" y="' + (H - 6) + '">过去 ' + hist.length + ' 天</text>');
    out.push('<text x="' + (W - R) + '" y="' + (H - 6) + '" text-anchor="end">未来 ' + hr + ' 天</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts };
  }

  /* ---------- 纯几何：patch 切分 ---------- */
  function buildPatches(d, patchLen, stride) {
    var vals = d.y_train.slice(Math.max(0, d.y_train.length - 96));
    var Lv = vals.length;
    var n = Math.max(1, Math.floor((Lv - Math.min(patchLen, Lv)) / stride) + 1);
    var wsn = (d.patch_attention && d.patch_attention.length === n) ? d.patch_attention.slice() : attentionFor(n);
    var rank = wsn.map(function (w, i) { return { i: i, w: w }; })
      .sort(function (a, b) { return b.w - a.w; });
    var top = rank.slice(0, 3).map(function (o) { return o.i; });

    var W = 340, L = 8, R = 6, T = 8, rowH = 12, B = 18;
    var H = T + n * rowH + B;
    var pw = W - L - R;
    var vmin = Math.min.apply(null, vals), vmax = Math.max.apply(null, vals);
    if (vmax - vmin < 1e-9) vmax = vmin + 1;
    function X(i) { return L + pw * Math.min(i, Lv) / Lv; }
    function Y(v, rowTop) { return rowTop + rowH - 4 - (rowH - 7) * (v - vmin) / (vmax - vmin); }
    var pts = [], out = [];

    out.push('<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="长序列按 patch_len 与 stride 切成重叠的 patch">');
    for (var r = 0; r < n; r++) {
      var s = r * stride;
      var e = Math.min(s + patchLen, Lv);
      var x1 = X(s), x2 = X(e), y = T + r * rowH;
      var isTop = top.indexOf(r) >= 0;
      var alt = r % 2 === 1;
      pts.push([x1, y], [x2, y + rowH - 3]);
      out.push('<rect x="' + fx(x1) + '" y="' + fx(y) + '" width="' + fx(Math.max(x2 - x1, 1)) + '" height="' + (rowH - 3) +
        '" rx="2" fill="' + (isTop ? C.b4 : (alt ? C.wash : '#F2F6FE')) +
        '" stroke="' + (isTop ? C.blue : C.line) + '" stroke-width="' + (isTop ? 1.4 : 0.8) + '"/>');
      var poly = [];
      for (var k = s; k < e; k++) poly.push(fx(X(k) + (X(1) - X(0)) / 2) + ',' + fx(Y(vals[k], y)));
      if (poly.length > 1) {
        var pv = poly.join(' ').split(' ');
        pv.forEach(function (pair) {
          var xy = pair.split(',');
          pts.push([parseFloat(xy[0]), parseFloat(xy[1])]);
        });
        out.push('<polyline points="' + poly.join(' ') + '" fill="none" stroke="' + C.blue + '" stroke-width="1" stroke-opacity="0.85" stroke-linejoin="round"/>');
      }
    }
    out.push('<line class="grid" x1="' + L + '" y1="' + (T + n * rowH + 2) + '" x2="' + (W - R) + '" y2="' + (T + n * rowH + 2) + '"/>');
    out.push('<text x="' + L + '" y="' + (H - 4) + '">' + Lv + ' 天前</text>');
    out.push('<text x="' + (W - R) + '" y="' + (H - 4) + '" text-anchor="end">昨天</text>');
    out.push('<text x="' + fx(W / 2) + '" y="' + (H - 4) + '" text-anchor="middle">每行 = 一个 patch（' + Math.min(patchLen, Lv) + ' 点）</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts, n: n, top: top, weights: wsn, len: Lv };
  }

  /* ---------- 纯几何：WAPE 曲线 ---------- */
  function interpWape(curve, x) {
    var c = curve.slice().sort(function (a, b) { return a.patch_len - b.patch_len; });
    if (x <= c[0].patch_len) return c[0].wape;
    if (x >= c[c.length - 1].patch_len) return c[c.length - 1].wape;
    for (var i = 1; i < c.length; i++) {
      if (x <= c[i].patch_len) {
        var t = (x - c[i - 1].patch_len) / (c[i].patch_len - c[i - 1].patch_len);
        return c[i - 1].wape + t * (c[i].wape - c[i - 1].wape);
      }
    }
    return c[c.length - 1].wape;
  }

  function buildCurve(curve, sel) {
    var W = 340, H = 132, L = 34, R = 10, T = 20, B = 28;
    var c = curve.slice().sort(function (a, b) { return a.patch_len - b.patch_len; });
    var xmin = c[0].patch_len, xmax = c[c.length - 1].patch_len;
    var wmin = Math.min.apply(null, c.map(function (o) { return o.wape; }));
    var wmax = Math.max.apply(null, c.map(function (o) { return o.wape; }));
    var pad = (wmax - wmin) * 0.18 || 0.01;
    var lo = wmin - pad, hi = wmax + pad;
    var pw = W - L - R, ph = H - T - B, pts = [];
    function X(v) { return L + pw * (v - xmin) / (xmax - xmin); }
    function Y(v) { return T + ph * (1 - (v - lo) / (hi - lo)); }
    var out = ['<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="patch_len 与 WAPE 的关系曲线">'];
    out.push('<text x="' + L + '" y="12" style="font-weight:700">WAPE 随 patch 长度变化（越低越好）</text>');
    var g, gv, gy;
    for (g = 0; g < 2; g++) {
      gv = lo + (hi - lo) * (g + 1) / 3;
      gy = Y(gv);
      out.push('<line class="grid" x1="' + L + '" y1="' + fx(gy) + '" x2="' + (W - R) + '" y2="' + fx(gy) + '"/>');
      out.push('<text x="' + (L - 5) + '" y="' + fx(gy + 3.5) + '" text-anchor="end">' + (gv * 100).toFixed(0) + '%</text>');
    }
    var line = [];
    c.forEach(function (o) {
      var x = X(o.patch_len), y = Y(o.wape);
      pts.push([x, y]);
      line.push(fx(x) + ',' + fx(y));
    });
    out.push('<polyline points="' + line.join(' ') + '" fill="none" stroke="' + C.b3 + '" stroke-width="2"/>');
    var best = c.reduce(function (a, b) { return b.wape < a.wape ? b : a; });
    c.forEach(function (o) {
      out.push('<circle cx="' + fx(X(o.patch_len)) + '" cy="' + fx(Y(o.wape)) + '" r="2.4" fill="' + C.b3 + '"/>');
      if (o.patch_len === best.patch_len) {
        out.push('<text x="' + fx(X(o.patch_len)) + '" y="' + fx(Y(o.wape) - 8) + '" text-anchor="middle" style="font-weight:700;fill:#1D4ED8">最优 ' + (o.wape * 100).toFixed(2) + '%</text>');
      }
    });
    var sx = X(Math.max(xmin, Math.min(xmax, sel))), sy = Y(interpWape(c, sel));
    pts.push([sx, sy]);
    out.push('<line x1="' + fx(sx) + '" y1="' + T + '" x2="' + fx(sx) + '" y2="' + (T + ph) + '" stroke="' + C.blue + '" stroke-width="1" stroke-dasharray="3 2"/>');
    out.push('<circle cx="' + fx(sx) + '" cy="' + fx(sy) + '" r="4.5" fill="' + C.blue + '" stroke="#fff" stroke-width="1.5"/>');
    out.push('<text x="' + L + '" y="' + (H - 6) + '">4</text>');
    out.push('<text x="' + (W - R) + '" y="' + (H - 6) + '" text-anchor="end">32 点</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts };
  }

  /* ---------- 纯几何：patch 注意力 ---------- */
  function buildPatchAttn(weights, top) {
    var n = weights.length;
    var W = 340, H = 96, L = 8, R = 8, T = 14, B = 22;
    var pw = W - L - R, ph = H - T - B, slot = pw / n, bw = Math.max(6, slot - 6);
    var mx = Math.max.apply(null, weights) || 1;
    var pts = [], out = [];
    out.push('<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="每个 patch 的注意力权重">');
    out.push('<line class="grid" x1="' + L + '" y1="' + (T + ph) + '" x2="' + (W - R) + '" y2="' + (T + ph) + '"/>');
    weights.forEach(function (w, i) {
      var bh = Math.max(2, ph * w / mx);
      var x = L + slot * i + (slot - bw) / 2, y = T + ph - bh;
      var isTop = top.indexOf(i) >= 0;
      pts.push([x, y], [x + bw, y + bh]);
      out.push('<rect x="' + fx(x) + '" y="' + fx(y) + '" width="' + fx(bw) + '" height="' + fx(bh) + '" rx="2" fill="' +
        (isTop ? C.blue : C.b4) + '"/>');
      if (isTop) {
        out.push('<text x="' + fx(x + bw / 2) + '" y="' + fx(y - 3) + '" text-anchor="middle" style="font-weight:700;fill:#1D4ED8">' + (w * 100).toFixed(0) + '%</text>');
      }
      out.push('<text x="' + fx(x + bw / 2) + '" y="' + (H - 8) + '" text-anchor="middle">P' + i + '</text>');
    });
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts };
  }

  /* ---------- 纯几何：通道独立示意 ---------- */
  function buildChannel() {
    var W = 340, H = 128, pts = [];
    var out = ['<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="通道独立：每个 SKU 独立成通道，共享同一套权重">'];
    var lanes = ['SKU A', 'SKU B', 'SKU C'];
    var ys = [12, 48, 84];
    lanes.forEach(function (name, li) {
      var y = ys[li];
      out.push('<text x="6" y="' + (y + 13) + '" style="font-weight:700">' + name + '</text>');
      for (var b = 0; b < 3; b++) {
        var bx = 48 + b * 32;
        pts.push([bx, y], [bx + 28, y + 16]);
        out.push('<rect x="' + bx + '" y="' + y + '" width="28" height="16" rx="3" fill="' + (b === 2 ? C.b4 : C.wash) + '" stroke="' + C.line + '"/>');
      }
      out.push('<path d="M' + 146 + ',' + (y + 8) + ' C172,' + (y + 8) + ' 168,' + 60 + ' 188,' + 60 + '" fill="none" stroke="' + C.b4 + '" stroke-width="1.2"/>');
    });
    out.push('<rect x="190" y="22" width="142" height="76" rx="10" fill="' + C.wash + '" stroke="' + C.blue + '"/>');
    out.push('<text x="261" y="45" text-anchor="middle" style="font-weight:800;font-size:12px;fill:#1D4ED8">同一套权重</text>');
    out.push('<text x="261" y="61" text-anchor="middle" style="font-size:11px">Transformer 层</text>');
    out.push('<text x="261" y="79" text-anchor="middle" style="font-size:11px">参数共享 · 独立输出</text>');
    pts.push([190, 22], [332, 98]);
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts };
  }

  /* ---------- 挂载 ---------- */
  async function mount(root, ctx) {
    injectCSS();
    var raw = null;
    try { raw = await ctx.load('patchtst.json'); } catch (e) { raw = null; }
    var d = normalize(raw);
    var fakeTag = d.fake ? '<span class="pill">示意数据</span>' : '';

    var sae = 0, sact = 0, h;
    for (h = 0; h < d.y_test.length; h++) {
      sae += Math.abs(d.pred[h] - d.y_test[h]);
      sact += Math.abs(d.y_test[h]);
    }
    var wape = sact ? sae / sact * 100 : 0;
    var bestC = d.patch_len_curve.reduce(function (a, b) { return b.wape < a.wape ? b : a; });
    var winLen = Math.min(96, d.y_train.length);

    root.innerHTML = [
      '<p class="algo-lead">PatchTST 干了两件事：把长序列<b>切成片（patch）</b>当 token 送进 Transformer，再把<b>每个通道独立处理</b>——' +
      '一个 SKU 一条通道，所有通道共享同一套权重。切完片的 token 数从 ' + d.context + ' 降到几十个，注意力矩阵从 O(n²) 掉到能算；' +
      '通道独立又让它不怕跨品类量纲差。</p>',

      '<div class="sfw-patchtst-block">',
      '<p class="sfw-patchtst-sub2">① 序列图 · PatchTST 预测 vs 真值 ' + fakeTag + '</p>',
      '<div class="sfw-legend" style="margin-bottom:6px">' +
      '<span><i style="background:' + C.blue + '"></i>PatchTST 预测</span>' +
      '<span><i style="background:' + C.truth + '"></i>真值</span></div>',
      '<div data-series></div>',
      '<div class="sfw-patchtst-num">',
      '<div><div class="sfw-metric">' + wape.toFixed(1) + '%</div><div class="sfw-patchtst-numlab">14 天 WAPE</div></div>',
      '<div><div class="sfw-metric">' + i0(d.context) + '</div><div class="sfw-patchtst-numlab">回看窗口（点）</div></div>',
      '<div><div class="sfw-metric">' + d.stride + '</div><div class="sfw-patchtst-numlab">stride</div></div>',
      '</div>',
      '</div>',

      '<div class="sfw-patchtst-block">',
      '<p class="sfw-patchtst-sub2">② 拖一拖：一条长序列怎么被切成 patch</p>',
      '<div data-patches></div>',
      '<div class="sfw-patchtst-ctl">',
      '<label>patch_len = <b data-pl>16</b> 点（4–32）',
      '<input class="sfw-slider" type="range" min="4" max="32" step="2" value="16" data-slider aria-label="patch 长度"></label>',
      '</div>',
      '<div class="sfw-patchtst-num">',
      '<div><div class="sfw-metric" data-n>–</div><div class="sfw-patchtst-numlab">patch 块数</div></div>',
      '<div><div class="sfw-metric" data-tok>–</div><div class="sfw-patchtst-numlab">每块 token 长度</div></div>',
      '<div><div class="sfw-metric" data-ov>–</div><div class="sfw-patchtst-numlab">相邻重叠（点）</div></div>',
      '<div><div class="sfw-metric" data-ctx>–</div><div class="sfw-patchtst-numlab">512 点上下文的 token 数</div></div>',
      '</div>',
      '<p class="sfw-patchtst-note" data-pnote></p>',
      '<div data-curve style="margin-top:8px"></div>',
      '<p class="sfw-cap" data-wcape></p>',
      '</div>',

      '<div class="sfw-patchtst-block">',
      '<p class="sfw-patchtst-sub2">③ 每个 patch 被关注的程度（top3 实心蓝）</p>',
      '<div data-pattn></div>',
      '<p class="sfw-cap" data-pattncap></p>',
      '</div>',

      '<div class="sfw-patchtst-block">',
      '<p class="sfw-patchtst-sub2">④ 通道独立：多 SKU 各走各的通道，共享一套权重</p>',
      '<div data-ch></div>',
      '<p class="sfw-cap">同一套权重吃遍所有 SKU：量纲差、间歇性、跨品类都不互相污染，所以换个品类也能直接微调。</p>',
      '</div>',

      '<dl class="algo-facts">',
      '<div><dt>论文</dt><dd>Nie et al. ICLR 2023 · <a href="https://arxiv.org/abs/2211.14730" target="_blank" rel="noopener">arXiv:2211.14730</a></dd></div>',
      '<div><dt>开源</dt><dd><a href="https://github.com/yuqinie98/PatchTST" target="_blank" rel="noopener">yuqinie98/PatchTST</a></dd></div>',
      '<div><dt>本机复现</dt><dd>' + (d.fake ? '数据文件未产出，本页为示意数据' : '序列 ' + String(d.series_id)) + '</dd></div>',
      '<div><dt>超参</dt><dd>patch_len=' + d.patch_len + ' · stride=' + d.stride + ' · context=' + d.context + '</dd></div>',
      '</dl>',
      '<p class="algo-note">序列长、SKU 多、又不想为每个品类调一套参数时用它；序列短（&lt; 100 点）时 patch 化收益有限，先试 LightGBM。</p>'
    ].join('');

    root.querySelector('[data-series]').innerHTML = buildSeries(d).html;

    var pBox = root.querySelector('[data-patches]');
    var cBox = root.querySelector('[data-curve]');
    var aBox = root.querySelector('[data-pattn]');
    var slider = root.querySelector('[data-slider]');
    var elPl = root.querySelector('[data-pl]');
    var elN = root.querySelector('[data-n]');
    var elTok = root.querySelector('[data-tok]');
    var elOv = root.querySelector('[data-ov]');
    var elCtx = root.querySelector('[data-ctx]');
    var pnote = root.querySelector('[data-pnote]');
    var wcap = root.querySelector('[data-wcape]');
    var acap = root.querySelector('[data-pattncap]');
    var stride = d.stride || 8;
    var state = { pl: 16 };

    function renderPatches() {
      var g = buildPatches(d, state.pl, stride);
      pBox.innerHTML = g.html;
      cBox.innerHTML = buildCurve(d.patch_len_curve, state.pl).html;
      aBox.innerHTML = buildPatchAttn(g.weights, g.top).html;

      elPl.textContent = String(state.pl);
      elN.textContent = String(g.n);
      elTok.textContent = String(Math.min(state.pl, g.len));
      elOv.textContent = String(Math.max(0, Math.min(state.pl, g.len) - stride));
      /* 512 点上下文按 stride=patch_len/2 折算（PatchTST 常用配置），用于说明 O(n²) 代价 */
      var tokCtx = Math.ceil(d.context / (state.pl / 2));
      elCtx.textContent = String(tokCtx);

      var wp = interpWape(d.patch_len_curve, state.pl) * 100;
      wcap.innerHTML = '当前 patch_len=' + state.pl + ' → 插值 WAPE <b style="color:#1D4ED8">' + wp.toFixed(2) + '%</b>' +
        '（验证集扫参最优：' + bestC.patch_len + ' 点 / ' + (bestC.wape * 100).toFixed(2) + '%）。';
      pnote.textContent = state.pl <= 6
        ? '块太小：同样的 512 点上下文要 ' + tokCtx + ' 个 token，注意力矩阵 ' + tokCtx + '² ≈ ' +
          Math.round(tokCtx * tokCtx / 1000) + 'k 项，显存先爆；每块还看不到一个完整周期。'
        : (state.pl >= 28
          ? '块太大：一块盖住 ' + Math.max(1, Math.round(state.pl / 7)) + ' 个星期，周内形状被压平，WAPE 从谷底回升。'
          : '块大小适中：512 上下文只要 ' + tokCtx + ' 个 token，每块盖住 ' + (state.pl / 7).toFixed(1) +
            ' 个星期——曲线谷底附近。');

      var ranges = g.top.map(function (i) {
        var s = i * stride, e = Math.min(s + Math.min(state.pl, g.len), g.len);
        return 'P' + i + '（第 ' + (g.len - e + 1) + '–' + (g.len - s) + ' 天前）';
      });
      acap.textContent = '模型最关注的三个 patch：' + ranges.join('、') +
        '。它们合起来盖住的是历史上最像"当下"的那几段。';
    }

    slider.addEventListener('input', function () {
      state.pl = Math.max(4, Math.min(32, parseInt(slider.value, 10) || 16));
      renderPatches();
    });

    renderPatches();
    root.querySelector('[data-ch]').innerHTML = buildChannel().html;
  }

  window.SF = window.SF || {};
  window.SF.algo = window.SF.algo || {};
  window.SF.algo[ID] = {
    id: ID,
    title: 'PatchTST',
    tagline: '切片 + 通道独立：长序列也能算得动',
    mount: mount,
    _geom: {
      synth: synth, series: buildSeries, patches: buildPatches, curve: buildCurve,
      pattn: buildPatchAttn, channel: buildChannel, interp: interpWape, attnFor: attentionFor
    }
  };
})();
