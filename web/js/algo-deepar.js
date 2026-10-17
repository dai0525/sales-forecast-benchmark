/* DeepAR：概率自回归 —— 一条预测，一片可能 */
(function () {
  'use strict';

  var ID = 'deepar';
  var C = {
    blue: '#1D4ED8', b3: '#60A5FA', b4: '#BFD2F6', wash: '#EFF4FF', line: '#D6E2FB',
    ink: '#0F1B2D', ink2: '#5A6B85', hair: '#E6EBF3', truth: '#8A97AB'
  };

  var CSS = [
    '.sfw-deepar-fig{margin:0}',
    '.sfw-deepar-ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}',
    '.sfw-deepar-ctl>label{flex:1 1 160px;min-width:150px;font-size:12px;color:' + C.ink2 + ';font-weight:600}',
    '.sfw-deepar-ctl input[type=range]{display:block;margin-top:2px;height:40px}',
    '.sfw-deepar-num{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}',
    '.sfw-deepar-num>div{flex:1 1 130px;min-width:126px;background:' + C.wash + ';border:1px solid ' + C.line + ';border-radius:10px;padding:8px 10px}',
    '.sfw-deepar-num .sfw-metric{font-size:22px;line-height:1.15}',
    '.sfw-deepar-sub{font-size:11px;color:' + C.ink2 + ';margin-top:3px;line-height:1.4}',
    '.sfw-deepar-sub2{font-size:11px;color:' + C.ink2 + ';margin:0 0 6px;font-weight:700;letter-spacing:.3px}',
    '.sfw-deepar-hint{font-size:12px;color:' + C.ink2 + ';margin:10px 0 0;min-height:34px}'
  ].join('');

  /* ---------- 基础工具 ---------- */
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
  function q(arr, p) {
    var a = arr.slice().sort(function (x, y) { return x - y; });
    if (!a.length) return 0;
    var i = (a.length - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
    return a[lo] + (a[hi] - a[lo]) * (i - lo);
  }
  function n1(v) { return Math.round(v * 10) / 10; }
  function i0(v) { return Math.round(v); }
  function fx(v) { return (Math.round(v * 100) / 100).toString(); }

  /* ---------- 内置示意数据（页面上已标「示意数据」） ---------- */
  function synth() {
    var r = rng(4711);
    var W = [0.83, 0.80, 0.86, 0.93, 1.14, 1.30, 1.02];
    function base(t) { return 118 * (1 + 0.14 * t / 134) * W[t % 7]; }
    var train = [], test = [], t, h;
    for (t = 0; t < 120; t++) train.push(i0(base(t) * (1 + 0.08 * gauss(r))));
    for (t = 120; t < 134; t++) test.push(i0(base(t) * (1 + 0.085 * gauss(r))));

    var draws = [], p;
    for (p = 0; p < 240; p++) {
      var shift = 0.055 * gauss(r), e = 0, row = [];
      for (h = 0; h < 14; h++) {
        e += 0.034 * gauss(r);
        row.push(Math.max(0, base(120 + h) * (1 + shift + e)));
      }
      draws.push(row);
    }
    var p10 = [], p50 = [], p90 = [], col;
    for (h = 0; h < 14; h++) {
      col = draws.map(function (x) { return x[h]; });
      p10.push(n1(q(col, 0.10)));
      p50.push(n1(q(col, 0.50)));
      p90.push(n1(q(col, 0.90)));
    }
    var samples = draws.slice(0, 20).map(function (row) {
      return row.map(function (v) { return i0(v); });
    });
    var cal = [], k;
    for (k = 1; k <= 9; k++) {
      var nom = k / 10;
      cal.push({
        nominal: nom,
        actual: n1((0.5 + (nom - 0.5) * 0.862 + (k % 2 ? 0.007 : -0.005)) * 1000) / 1000
      });
    }
    return {
      series_id: '示意序列 D-1', y_train: train, y_test: test,
      p10: p10, p50: p50, p90: p90, samples: samples, calibration: cal, fake: true
    };
  }

  function normalize(raw) {
    var ok = raw && raw.y_train && raw.y_train.length > 30 && raw.y_test && raw.y_test.length &&
      raw.p10 && raw.p10.length === raw.y_test.length && raw.p50 && raw.p50.length === raw.y_test.length &&
      raw.samples && raw.samples.length >= 3;
    if (!ok) return synth();
    raw.fake = false;
    if (!raw.p90 || raw.p90.length !== raw.y_test.length) {
      var s = raw.samples;
      raw.p90 = raw.y_test.map(function (_, i) { return q(s.map(function (p) { return p[i]; }), 0.9); });
    }
    if (!raw.calibration || !raw.calibration.length) raw.calibration = synth().calibration;
    return raw;
  }

  /* ---------- 纯几何：扇形图 ---------- */
  function buildFan(d, paths) {
    var W = 340, H = 206, L = 34, R = 8, T = 12, B = 26;
    var hr = d.y_test.length;
    var hist = d.y_train.slice(Math.max(0, d.y_train.length - 60));
    var n = hist.length + hr;
    var lo = Infinity, hi = -Infinity, pts = [], out = [];
    function scan(v) { if (v < lo) lo = v; if (v > hi) hi = v; }
    hist.forEach(scan); d.y_test.forEach(scan); d.p10.forEach(scan); d.p90.forEach(scan);
    paths.forEach(function (p) { p.forEach(scan); });
    if (!isFinite(lo) || !isFinite(hi) || hi - lo < 1e-9) { lo = lo - 1; hi = hi + 1; }
    var vpad = (hi - lo) * 0.08;
    lo -= vpad; hi += vpad;

    var pw = W - L - R, ph = H - T - B;
    function X(i) { return L + pw * i / (n - 1); }
    function Y(v) { return T + ph * (1 - (v - lo) / (hi - lo)); }
    function P(x, y) { pts.push([x, y]); return fx(x) + ',' + fx(y); }
    var io = hist.length - 1;
    var last = hist[io];

    out.push('<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="DeepAR 概率预测扇形图：历史、真值、P50 与 P10-P90 区间">');
    var g, gy, gv;
    for (g = 0; g < 3; g++) {
      gv = lo + (hi - lo) * (g + 1) / 4;
      gy = Y(gv);
      out.push('<line class="grid" x1="' + L + '" y1="' + fx(gy) + '" x2="' + (W - R) + '" y2="' + fx(gy) + '"/>');
      out.push('<text x="' + (L - 5) + '" y="' + fx(gy + 3.5) + '" text-anchor="end">' + i0(gv) + '</text>');
    }

    var band = ['M' + P(X(io), Y(last))];
    for (g = 0; g < hr; g++) band.push('L' + P(X(io + 1 + g), Y(d.p90[g])));
    for (g = hr - 1; g >= 0; g--) band.push('L' + P(X(io + 1 + g), Y(d.p10[g])));
    out.push('<path d="' + band.join(' ') + ' Z" fill="' + C.b4 + '" fill-opacity="0.55" stroke="' + C.line + '" stroke-width="1"/>');

    paths.forEach(function (p) {
      var s = ['M' + P(X(io), Y(last))];
      for (g = 0; g < hr; g++) s.push('L' + P(X(io + 1 + g), Y(p[g])));
      out.push('<path d="' + s.join(' ') + '" fill="none" stroke="' + C.blue + '" stroke-width="1" stroke-opacity="0.38" stroke-linejoin="round"/>');
    });

    var hs = [];
    hist.forEach(function (v, i) { hs.push((i ? 'L' : 'M') + P(X(i), Y(v))); });
    out.push('<path d="' + hs.join(' ') + '" fill="none" stroke="' + C.ink2 + '" stroke-width="1.5" stroke-linejoin="round"/>');

    var ts = ['M' + P(X(io), Y(last))];
    for (g = 0; g < hr; g++) ts.push('L' + P(X(io + 1 + g), Y(d.y_test[g])));
    out.push('<path d="' + ts.join(' ') + '" fill="none" stroke="' + C.truth + '" stroke-width="1.6" stroke-dasharray="4 3"/>');

    var ms = ['M' + P(X(io), Y(last))];
    for (g = 0; g < hr; g++) ms.push('L' + P(X(io + 1 + g), Y(d.p50[g])));
    out.push('<path d="' + ms.join(' ') + '" fill="none" stroke="' + C.blue + '" stroke-width="2.2" stroke-linejoin="round"/>');

    out.push('<line x1="' + fx(X(io)) + '" y1="' + T + '" x2="' + fx(X(io)) + '" y2="' + (T + ph) + '" stroke="' + C.line + '" stroke-width="1" stroke-dasharray="3 3"/>');
    out.push('<text x="' + L + '" y="' + (H - 6) + '">过去 ' + hist.length + ' 天</text>');
    out.push('<text x="' + (W - R) + '" y="' + (H - 6) + '" text-anchor="end">未来 ' + hr + ' 天</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts, dom: [lo, hi] };
  }

  /* ---------- 纯几何：区间宽度 ---------- */
  function buildWidths(d) {
    var W = 340, H = 168, L = 30, R = 8, T = 26, B = 26;
    var hr = d.y_test.length, w = [], h, pts = [];
    for (h = 0; h < hr; h++) w.push((d.p90[h] - d.p10[h]) / Math.max(1, d.p50[h]) * 100);
    var mx = Math.max.apply(null, w) * 1.08 || 1;
    var pw = W - L - R, ph = H - T - B, slot = pw / hr, bw = Math.min(18, slot * 0.64);
    var out = ['<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="P10-P90 区间宽度随预测步长增大">'];
    out.push('<text x="' + L + '" y="13" style="font-weight:700">80% 区间宽度 ÷ P50（%）</text>');
    out.push('<line class="grid" x1="' + L + '" y1="' + (T + ph) + '" x2="' + (W - R) + '" y2="' + (T + ph) + '"/>');
    for (h = 0; h < hr; h++) {
      var bh = Math.max(ph * w[h] / mx, 1.5);
      var x = L + slot * h + (slot - bw) / 2;
      var y = T + ph - bh;
      pts.push([x, y], [x + bw, y + bh]);
      out.push('<rect x="' + fx(x) + '" y="' + fx(y) + '" width="' + fx(bw) + '" height="' + fx(bh) + '" rx="2.5" fill="' + (h === hr - 1 ? C.blue : C.b4) + '"/>');
      if (h === 0 || h === 6 || h === hr - 1) {
        out.push('<text x="' + fx(x + bw / 2) + '" y="' + fx(y - 4) + '" text-anchor="middle" style="font-weight:700"' +
          (h === hr - 1 ? ' fill="#1D4ED8"' : '') + '>' + Math.round(w[h]) + '</text>');
      }
    }
    out.push('<text x="' + fx(L + slot * 0.5) + '" y="' + (H - 8) + '" text-anchor="middle">h=1</text>');
    out.push('<text x="' + fx(L + slot * 6.5) + '" y="' + (H - 8) + '" text-anchor="middle">h=7</text>');
    out.push('<text x="' + (W - R) + '" y="' + (H - 8) + '" text-anchor="end">h=' + hr + '</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts, info: { w1: w[0], wl: w[hr - 1], ratio: w[hr - 1] / Math.max(0.5, w[0]) } };
  }

  /* ---------- 纯几何：校准 ---------- */
  function buildCalib(d) {
    var cal = d.calibration.slice().sort(function (a, b) { return a.nominal - b.nominal; });
    var W = 340, H = 196, x0 = 42, y0 = 18, S = 124, pts = [];
    function X(v) { return x0 + Math.max(0, Math.min(1, v)) * S; }
    function Y(v) { return y0 + S - Math.max(0, Math.min(1, v)) * S; }
    var out = ['<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="校准图：名义覆盖率与实际覆盖率">'];
    out.push('<rect x="' + x0 + '" y="' + y0 + '" width="' + S + '" height="' + S + '" fill="#F7F9FD" stroke="' + C.hair + '"/>');
    out.push('<line x1="' + x0 + '" y1="' + (y0 + S) + '" x2="' + (x0 + S) + '" y2="' + y0 + '" stroke="' + C.truth + '" stroke-width="1.4" stroke-dasharray="4 3"/>');
    var poly = [];
    cal.forEach(function (c) {
      var px = X(c.actual), py = Y(c.nominal);
      pts.push([px, py]);
      poly.push(fx(px) + ',' + fx(py));
      out.push('<circle cx="' + fx(px) + '" cy="' + fx(py) + '" r="2.8" fill="' + C.blue + '"/>');
    });
    out.push('<polyline points="' + poly.join(' ') + '" fill="none" stroke="' + C.blue + '" stroke-width="1.6" stroke-opacity="0.55"/>');
    var c90 = cal[cal.length - 1];
    if (c90) {
      pts.push([X(c90.actual), Y(c90.nominal)]);
      out.push('<circle cx="' + fx(X(c90.actual)) + '" cy="' + fx(Y(c90.nominal)) + '" r="5" fill="none" stroke="' + C.blue + '" stroke-width="1.6"/>');
    }
    out.push('<text x="' + x0 + '" y="' + (y0 + S + 14) + '">0%</text>');
    out.push('<text x="' + (x0 + S) + '" y="' + (y0 + S + 14) + '" text-anchor="end">100%</text>');
    out.push('<text x="' + fx(x0 + S / 2) + '" y="' + (H - 6) + '" text-anchor="middle">名义覆盖率</text>');
    out.push('<text x="' + x0 + '" y="' + (y0 - 5) + '">实际</text>');
    out.push('<text x="' + (x0 - 6) + '" y="' + (y0 + 6) + '" text-anchor="end">100%</text>');
    out.push('<text x="' + (x0 - 6) + '" y="' + (y0 + S - 1) + '" text-anchor="end">0%</text>');
    var dev = c90 ? (c90.actual - c90.nominal) * 100 : 0;
    if (c90) {
      out.push('<text x="' + (x0 + S + 14) + '" y="' + (y0 + 22) + '">名义 90%</text>');
      out.push('<text x="' + (x0 + S + 14) + '" y="' + (y0 + 46) + '" style="font-size:18px;font-weight:800;fill:#1D4ED8">' + (c90.actual * 100).toFixed(1) + '%</text>');
      out.push('<text x="' + (x0 + S + 14) + '" y="' + (y0 + 62) + '">实际覆盖</text>');
      out.push('<text x="' + (x0 + S + 14) + '" y="' + (y0 + 84) + '">' + (dev < 0 ? '偏窄 ' : '偏宽 ') + Math.abs(dev).toFixed(1) + ' 点</text>');
    }
    out.push('<text x="' + x0 + '" y="' + (H - 24) + '" style="font-size:11px">虚线=理想，越贴越好</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts, actual90: c90 ? c90.actual : null };
  }

  /* ---------- 采样度量 ---------- */
  function measure(paths, yTest) {
    var hr = yTest.length, sumW = 0, hit = 0, h, col;
    for (h = 0; h < hr; h++) {
      col = paths.map(function (p) { return p[h]; });
      var lo = q(col, 0.05), hi = q(col, 0.95), mid = Math.max(1, q(col, 0.5));
      sumW += (hi - lo) / mid;
      if (yTest[h] >= lo && yTest[h] <= hi) hit++;
    }
    return { width: sumW / hr * 100, hit: hit / hr * 100 };
  }

  /* ---------- 挂载 ---------- */
  async function mount(root, ctx) {
    injectCSS();
    var raw = null;
    try { raw = await ctx.load('deepar.json'); } catch (e) { raw = null; }
    var d = normalize(raw);
    var fakeTag = d.fake ? '<span class="pill">示意数据</span>' : '';

    root.innerHTML = [
      '<p class="algo-lead">DeepAR 不预测「第 7 天卖 231 件」这一个数，它预测<b>一个分布</b>：每一步输出均值与方差，训练时最大化真实值的似然，预测时按这个分布采样。' +
      '序列长度的伸缩、跨 SKU 共享参数都是白送的——一条序列训练的全局模型，冷门 SKU 也能借到热门 SKU 的季节形状。</p>',

      '<div class="sfw-deepar-fig">',
      '<p class="sfw-deepar-sub2">① 扇形图 · 60 天历史 → 14 天预测 ' + fakeTag + '</p>',
      '<div class="sfw-legend" style="margin-bottom:6px">' +
      '<span><i style="background:' + C.ink2 + ';height:3px"></i>历史真实</span>' +
      '<span><i style="background:' + C.b4 + ';height:8px"></i>P10–P90</span>' +
      '<span><i style="background:' + C.blue + '"></i>P50 中位数</span>' +
      '<span><i style="background:' + C.truth + '"></i>真值</span>' +
      '</div>',
      '<div data-fan></div>',
      '<div class="sfw-deepar-ctl">',
      '<button class="sfw-btn" type="button" data-resample>重新采样</button>',
      '<label>采样路径数 <b data-nshow style="color:' + C.blue + '">20</b> 条',
      '<input class="sfw-slider" type="range" min="1" max="30" step="1" value="20" data-n aria-label="采样路径数"></label>',
      '</div>',
      '<div class="sfw-deepar-num">',
      '<div><div class="sfw-metric" data-w>–</div><div class="sfw-deepar-sub">本次采样 90% 区间均宽<br>（占预测值）</div></div>',
      '<div><div class="sfw-metric" data-h>–</div><div class="sfw-deepar-sub">14 天真值落在<br>区间内的比例</div></div>',
      '</div>',
      '<p class="sfw-deepar-hint" data-hint></p>',
      '<p class="sfw-cap">同一条 P50，配上不同的采样噪声，就是 20 种都说得通的未来。点「重新采样」，看区间宽度和命中率怎么抖。</p>',
      '</div>',

      '<div class="sfw-grid2" style="margin-top:16px">',
      '<div><p class="sfw-deepar-sub2">② 不确定性随步长变大</p><div data-widths></div>' +
      '<p class="sfw-cap" data-wcap style="margin-top:4px"></p></div>',
      '<div><p class="sfw-deepar-sub2">③ 校准：它说的 90% 是真 90% 吗</p><div data-calib></div>' +
      '<p class="sfw-cap" data-calcap style="margin-top:4px"></p></div>',
      '</div>',

      '<dl class="algo-facts">',
      '<div><dt>论文</dt><dd>Salinas et al. 2017 · <a href="https://arxiv.org/abs/1704.04110" target="_blank" rel="noopener">arXiv:1704.04110</a></dd></div>',
      '<div><dt>开源</dt><dd><a href="https://github.com/awslabs/gluonts" target="_blank" rel="noopener">awslabs/gluonts</a> · <a href="https://github.com/Nixtla/neuralforecast" target="_blank" rel="noopener">Nixtla/neuralforecast</a></dd></div>',
      '<div><dt>本机复现</dt><dd>' + (d.fake ? '数据文件未产出，本页为示意数据' : '序列 ' + d.series_id) + '</dd></div>',
      '<div><dt>核心机制</dt><dd>自回归 RNN · 每步输出分布参数 (μ,σ) · 似然训练 · 采样推理</dd></div>',
      '</dl>',
      '<p class="algo-note">要「按 90% 分位数备货」这类区间决策时用它；只要点预测，它比 LightGBM 贵且未必更准。</p>'
    ].join('');

    var fanBox = root.querySelector('[data-fan]');
    var wBox = root.querySelector('[data-widths]');
    var cBox = root.querySelector('[data-calib]');
    var hint = root.querySelector('[data-hint]');
    var elW = root.querySelector('[data-w]');
    var elH = root.querySelector('[data-h]');
    var nShow = root.querySelector('[data-nshow]');
    var slider = root.querySelector('[data-n]');
    var pool = d.samples;

    var state = { n: Math.min(20, pool.length), idx: null };
    function seq() {
      var a = [], i;
      for (i = 0; i < state.n; i++) a.push(i % pool.length);
      return a;
    }
    function rand() {
      var a = [], i;
      for (i = 0; i < state.n; i++) a.push(Math.floor(Math.random() * pool.length));
      return a;
    }

    function render() {
      if (!state.idx || state.idx.length !== state.n) state.idx = seq();
      var paths = state.idx.map(function (i) { return pool[i]; });
      fanBox.innerHTML = buildFan(d, paths).html;
      var m = measure(paths, d.y_test);
      elW.textContent = '±' + m.width.toFixed(0) + '%';
      elH.textContent = m.hit.toFixed(0) + '%';
      nShow.textContent = String(state.n);
      if (state.n <= 2) {
        hint.textContent = '只有 ' + state.n + ' 条路径时，5–95 分位几乎退化成一条线：宽度 ' + m.width.toFixed(0) +
          '%，看着最「确定」，其实最不可信。';
      } else if (state.n < 8) {
        hint.textContent = state.n + ' 条路径：盖住 ' + m.hit.toFixed(0) + '% 的真值，区间还在抖，再抽几次看稳定性。';
      } else {
        hint.textContent = state.n + ' 条路径的 5–95 分位盖住 ' + m.hit.toFixed(0) + '% 的真值，承诺的是 90%——' +
          (m.hit >= 75 ? '这次采样算诚实。' : '这次偏乐观，区间窄了。');
      }
    }

    var wg = buildWidths(d);
    wBox.innerHTML = wg.html;
    root.querySelector('[data-wcap]').textContent =
      'h=1 只有 ±' + Math.round(wg.info.w1) + '%，到 h=' + d.y_test.length + ' 变成 ±' + Math.round(wg.info.wl) +
      '%（' + wg.info.ratio.toFixed(1) + ' 倍）。越远越糊，不是模型偷懒，是它在诚实报价。';

    var cg = buildCalib(d);
    cBox.innerHTML = cg.html;
    root.querySelector('[data-calcap]').textContent = cg.actual90 == null
      ? '点贴着对角线才算诚实。'
      : '名义 90% 只覆盖了 ' + (cg.actual90 * 100).toFixed(1) + '%：区间偏窄。按它的 P90 备货会缺货。';

    root.querySelector('[data-resample]').addEventListener('click', function () {
      state.idx = rand();
      render();
    });
    slider.addEventListener('input', function () {
      state.n = Math.max(1, Math.min(30, parseInt(slider.value, 10) || 1));
      state.idx = null;
      render();
    });

    render();
  }

  window.SF = window.SF || {};
  window.SF.algo = window.SF.algo || {};
  window.SF.algo[ID] = {
    id: ID,
    title: 'DeepAR',
    tagline: '概率自回归：给一片未来，不是一个点',
    mount: mount,
    _geom: { synth: synth, fan: buildFan, widths: buildWidths, calib: buildCalib, measure: measure }
  };
})();
