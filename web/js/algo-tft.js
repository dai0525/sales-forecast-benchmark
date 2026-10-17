/* Temporal Fusion Transformer：变量选择 + 可解释注意力 */
(function () {
  'use strict';

  var ID = 'tft';
  var C = {
    blue: '#1D4ED8', b3: '#60A5FA', b4: '#BFD2F6', wash: '#EFF4FF', line: '#D6E2FB',
    ink: '#0F1B2D', ink2: '#5A6B85', hair: '#E6EBF3', truth: '#8A97AB'
  };

  var CSS = [
    '.sfw-tft-sub2{font-size:11px;color:' + C.ink2 + ';margin:0 0 6px;font-weight:700;letter-spacing:.3px}',
    '.sfw-tft-block{margin-top:16px}',
    '.sfw-tft-num{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}',
    '.sfw-tft-num>div{flex:1 1 120px;min-width:112px;background:' + C.wash + ';border:1px solid ' + C.line + ';border-radius:10px;padding:8px 10px}',
    '.sfw-tft-num .sfw-metric{font-size:20px}',
    '.sfw-tft-numlab{font-size:11px;color:' + C.ink2 + ';margin-top:2px}',
    '.sfw-tft-vars{display:flex;flex-direction:column;gap:2px;margin:6px 0 0}',
    '.sfw-tft-v{display:flex;align-items:center;gap:8px;width:100%;min-height:40px;padding:4px 6px;border:1px solid transparent;border-radius:9px;background:none;cursor:pointer;text-align:left;font:inherit}',
    '.sfw-tft-v:hover{background:#F7F9FD}',
    '.sfw-tft-v.is-on{border-color:' + C.line + ';background:' + C.wash + '}',
    '.sfw-tft-vn{flex:0 0 84px;font:600 12px/1.2 ui-monospace,monospace;color:' + C.ink + ';overflow-wrap:anywhere}',
    '.sfw-tft-vt{flex:1 1 auto;height:9px;background:' + C.hair + ';border-radius:6px;overflow:hidden;min-width:40px}',
    '.sfw-tft-vt i{display:block;height:100%;background:' + C.b4 + ';border-radius:6px}',
    '.sfw-tft-v.is-on .sfw-tft-vt i{background:' + C.blue + '}',
    '.sfw-tft-vw{flex:0 0 44px;text-align:right;font:700 12px/1 ui-monospace,monospace;color:' + C.ink2 + '}',
    '.sfw-tft-v.is-on .sfw-tft-vw{color:' + C.blue + '}',
    '.sfw-tft-ex{margin-top:10px;padding:10px 12px;background:' + C.wash + ';border-left:3px solid ' + C.blue + ';border-radius:0 9px 9px 0;font-size:13.5px;line-height:1.55;color:' + C.ink + ';min-height:62px}',
    '.sfw-tft-ex b{color:' + C.blue + '}',
    '.sfw-tft-detail{margin-top:8px;font-size:13px;color:' + C.ink + ';min-height:40px;line-height:1.5}',
    '.sfw-tft-detail b{color:' + C.blue + '}',
    '.sfw-tft-hit{stroke:transparent;fill:transparent;cursor:pointer}',
    '.sfw-tft-vars+.sfw-tft-ex{margin-top:8px}',
    '.sfw-tft-chips{margin:2px 0 8px}'
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
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c];
    });
  }
  function i0(v) { return Math.round(v); }

  /* ---------- 内置示意数据（页面上已标「示意数据」） ---------- */
  var VARS = [
    { name: 'lag_7', weight: 0.214 },
    { name: 'lag_14', weight: 0.146 },
    { name: 'rolling_mean_7', weight: 0.128 },
    { name: 'dayofweek', weight: 0.104 },
    { name: 'promo', weight: 0.091 },
    { name: 'lag_1', weight: 0.083 },
    { name: 'price_discount', weight: 0.062 },
    { name: 'month', weight: 0.048 },
    { name: 'holiday', weight: 0.041 },
    { name: 'sku_embedding', weight: 0.036 },
    { name: 'rolling_std_7', weight: 0.027 },
    { name: 'temperature', weight: 0.020 }
  ];

  function synth() {
    var r = rng(90210), rp = rng(31337), W = [0.84, 0.81, 0.87, 0.94, 1.12, 1.28, 1.03];
    function base(t) { return 96 * (1 + 0.18 * t / 134) * W[t % 7]; }
    var train = [], test = [], pred = [], t, h;
    for (t = 0; t < 120; t++) train.push(i0(base(t) * (1 + 0.085 * gauss(r))));
    for (h = 0; h < 14; h++) {
      test.push(i0(base(120 + h) * (1 + 0.09 * gauss(r))));
      pred.push(Math.round(base(120 + h) * (1 + 0.012 * Math.sin(h / 2.2)) * (1 + 0.042 * gauss(rp))));
    }
    /* 注意力基底：近因 + 7 日/14 日节律 */
    var jit = [], lag;
    for (lag = 0; lag < 28; lag++) jit.push(0.55 + 0.9 * r());
    return {
      series_id: '示意序列 T-1', y_train: train, y_test: test, pred: pred,
      variable_importance: VARS, attention: null, _jit: jit, fake: true
    };
  }

  function normalize(raw) {
    var ok = raw && raw.y_train && raw.y_train.length > 40 && raw.y_test && raw.y_test.length &&
      raw.pred && raw.pred.length === raw.y_test.length && raw.variable_importance && raw.variable_importance.length;
    if (!ok) return synth();
    raw.fake = false;
    raw._jit = raw._jit || synth()._jit;
    return raw;
  }

  /* ---------- 纯几何：序列图 ---------- */
  function buildSeries(d) {
    var W = 340, H = 196, L = 32, R = 8, T = 12, B = 26;
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

    out.push('<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="TFT 预测与真值对比">');
    var g, gy, gv;
    for (g = 0; g < 3; g++) {
      gv = lo + (hi - lo) * (g + 1) / 4;
      gy = Y(gv);
      out.push('<line class="grid" x1="' + L + '" y1="' + fx(gy) + '" x2="' + (W - R) + '" y2="' + fx(gy) + '"/>');
      out.push('<text x="' + (L - 5) + '" y="' + fx(gy + 3.5) + '" text-anchor="end">' + i0(gv) + '</text>');
    }
    var hs = [];
    hist.forEach(function (v, i) { hs.push((i ? 'L' : 'M') + P(X(i), Y(v))); });
    out.push('<path d="' + hs.join(' ') + '" fill="none" stroke="' + C.ink2 + '" stroke-width="1.4" stroke-linejoin="round" stroke-opacity="0.75"/>');

    var ts = ['M' + P(X(io), Y(last))];
    for (g = 0; g < hr; g++) ts.push('L' + P(X(io + 1 + g), Y(d.y_test[g])));
    out.push('<path d="' + ts.join(' ') + '" fill="none" stroke="' + C.truth + '" stroke-width="1.6" stroke-dasharray="4 3"/>');

    var ps = ['M' + P(X(io), Y(last))];
    for (g = 0; g < hr; g++) ps.push('L' + P(X(io + 1 + g), Y(d.pred[g])));
    out.push('<path d="' + ps.join(' ') + '" fill="none" stroke="' + C.blue + '" stroke-width="2.2" stroke-linejoin="round"/>');
    for (g = 0; g < hr; g++) {
      var cx = X(io + 1 + g), cy = Y(d.pred[g]);
      pts.push([cx, cy]);
      out.push('<circle cx="' + fx(cx) + '" cy="' + fx(cy) + '" r="2" fill="' + C.blue + '"/>');
    }
    out.push('<line x1="' + fx(X(io)) + '" y1="' + T + '" x2="' + fx(X(io)) + '" y2="' + (T + ph) + '" stroke="' + C.line + '" stroke-dasharray="3 3"/>');
    out.push('<text x="' + L + '" y="' + (H - 6) + '">过去 ' + hist.length + ' 天</text>');
    out.push('<text x="' + (W - R) + '" y="' + (H - 6) + '" text-anchor="end">未来 ' + hr + ' 天</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts };
  }

  /* ---------- 纯几何：注意力时间轴 ---------- */
  function attWeights(d, h) {
    /* 真实数据只给一组权重（不分步）→ 三个 h 复用同一组，并在图下标注 */
    if (d.attention && d.attention.length >= 10) {
      var src = d.attention.slice().sort(function (a, b) {
        return (a.t == null ? 0 : a.t) - (b.t == null ? 0 : b.t);
      });
      var m = src.length, out = [], j;
      for (j = 0; j < 28; j++) {
        var pos = Math.min(m - 1, Math.floor((j + 0.5) / 28 * m));
        out.push({ lag: 28 - j, w: Math.max(0, +src[pos].weight || 0) });
      }
      var tot = out.reduce(function (a, b) { return a + b.w; }, 0) || 1;
      out.forEach(function (o) { o.w = o.w / tot; });
      return out;
    }
    var jit = d._jit, raw = [], lag, j;
    for (j = 0; j < 28; j++) {
      lag = 28 - j;
      var v = 0.30 * Math.pow(0.94, lag) * jit[j] * 0.9;
      if (h === 1) {
        v += 1.30 * Math.exp(-Math.pow((lag - 1) / 1.7, 2));
        v += 0.22 * Math.exp(-Math.pow((lag - 7) / 2.6, 2));
      } else if (h === 7) {
        v += 1.45 * Math.exp(-Math.pow((lag - 7) / 1.8, 2));
        v += 0.30 * Math.exp(-Math.pow((lag - 1) / 2.4, 2));
        v += 0.22 * Math.exp(-Math.pow((lag - 14) / 2.6, 2));
      } else {
        v += 1.35 * Math.exp(-Math.pow((lag - 14) / 2.1, 2));
        v += 0.34 * Math.exp(-Math.pow((lag - 7) / 2.3, 2));
        v += 0.16 * Math.exp(-Math.pow((lag - 21) / 2.6, 2));
      }
      raw.push(v);
    }
    var sum = raw.reduce(function (a, b) { return a + b; }, 0) || 1;
    var ws = raw.map(function (v, idx) { return { lag: 28 - idx, w: v / sum }; });
    /* 归一化残差修正，确保总和 = 1 */
    var s2 = ws.reduce(function (a, b) { return a + b.w; }, 0);
    ws.forEach(function (o) { o.w = o.w / s2; });
    return ws;
  }

  function buildAttn(d, h, sel) {
    var ws = attWeights(d, h);
    var W = 340, H = 122, L = 8, R = 8, T = 16, B = 30;
    var pw = W - L - R, ph = H - T - B, slot = pw / ws.length, bw = Math.max(4, slot - 3);
    var mx = Math.max.apply(null, ws.map(function (o) { return o.w; })) || 1;
    var pts = [], out = [];
    out.push('<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="过去 28 天注意力权重">');
    out.push('<line class="grid" x1="' + L + '" y1="' + (T + ph) + '" x2="' + (W - R) + '" y2="' + (T + ph) + '"/>');
    var peak = 0;
    ws.forEach(function (o, i) { if (o.w > ws[peak].w) peak = i; });
    ws.forEach(function (o, i) {
      var x = L + slot * i + (slot - bw) / 2;
      var bh = Math.max(1.5, ph * o.w / mx);
      var y = T + ph - bh;
      var on = i === sel;
      pts.push([x, y], [x + bw, y + bh]);
      out.push('<rect x="' + fx(x) + '" y="' + fx(y) + '" width="' + fx(bw) + '" height="' + fx(bh) +
        '" rx="1.5" fill="' + (on ? C.blue : (o.w / mx > 0.62 ? C.b3 : C.b4)) + '"/>');
      if (i === peak) {
        out.push('<text x="' + fx(x + bw / 2) + '" y="' + fx(y - 3) + '" text-anchor="middle" style="font-weight:700;fill:#1D4ED8">' + Math.round(o.w * 100) + '%</text>');
      }
      out.push('<rect class="sfw-tft-hit" data-i="' + i + '" x="' + fx(L + slot * i) + '" y="' + T + '" width="' + fx(slot) + '" height="' + fx(ph) + '"/>');
    });
    if (sel >= 0) {
      var sx = L + slot * sel + slot / 2;
      pts.push([sx, T], [sx, T + ph]);
      out.push('<line x1="' + fx(sx) + '" y1="' + T + '" x2="' + fx(sx) + '" y2="' + (T + ph) + '" stroke="' + C.blue + '" stroke-width="1.5" stroke-dasharray="3 2"/>');
    }
    out.push('<text x="' + L + '" y="' + (H - 10) + '">28 天前</text>');
    out.push('<text x="' + (W - R) + '" y="' + (H - 10) + '" text-anchor="end">昨天</text>');
    out.push('<text x="' + fx(W / 2) + '" y="' + (H - 10) + '" text-anchor="middle">h=' + h + ' 主要看哪里</text>');
    out.push('</svg>');
    return { w: W, h: H, html: out.join(''), pts: pts, ws: ws, peak: peak };
  }

  /* ---------- 变量解释词典 ---------- */
  var MEAN = {
    lag_1: '昨天卖了多少——最近一天的动量，短促波动最先在这儿露头。',
    lag_7: '上周同一天卖了多少——把「星期几」的规律直接喂给模型，周中周末的差异靠它。',
    lag_14: '两周前同一天——双周发薪/促销周期的节律。',
    lag_28: '四周前同一天——月内位置效应（月初铺货、月末冲量）。',
    rolling_mean_7: '过去 7 天平均——剔掉单日噪声后的「当前水位」，模型用它定基线。',
    rolling_std_7: '过去 7 天的波动幅度——越不稳，模型越不敢给窄区间。',
    dayofweek: '星期效应：周末/工作日的客流结构不同，权重高说明这条序列吃「周节律」。',
    promo: '促销标记：打折、堆头、直播当天销量会跳，这是最该被单独解释的变量。',
    price_discount: '折扣力度：降几个点对应多少增量，直接决定促销 ROI 怎么算。',
    month: '月份/季节：换季、开学、年节带来的慢变量。',
    holiday: '节假日：春节/国庆的脉冲，以及节前透支、节后回落。',
    sku_embedding: '品类嵌入：这条 SKU 属于哪个品类，决定它跟谁借力。',
    temperature: '气温：饮料、取暖类商品的隐形开关。'
  };

  function meaning(name) {
    if (MEAN[name]) return MEAN[name];
    if (name.indexOf('lag_') === 0) return '滞后 ' + esc(name.slice(4)) + ' 天的销量——同一天的周期位置。';
    if (name.indexOf('rolling_') === 0) return '滚动统计量（' + esc(name) + '）：把最近一段时间的水平/波动压成一个数。';
    return '这个变量在模型里权重靠前，是变量选择网络自己挑出来的信号；业务含义需要跟你的特征表对齐。';
  }

  /* ---------- 挂载 ---------- */
  async function mount(root, ctx) {
    injectCSS();
    var raw = null;
    try { raw = await ctx.load('tft.json'); } catch (e) { raw = null; }
    var d = normalize(raw);
    var fakeTag = d.fake ? '<span class="pill">示意数据</span>' : '';

    /* 指标 */
    var sae = 0, sact = 0, h;
    for (h = 0; h < d.y_test.length; h++) {
      sae += Math.abs(d.pred[h] - d.y_test[h]);
      sact += Math.abs(d.y_test[h]);
    }
    var wape = sact ? sae / sact * 100 : 0;
    var sorted = d.variable_importance.slice().sort(function (a, b) { return b.weight - a.weight; });
    var top8 = sorted.slice(0, 8);
    var rest = sorted.slice(8).reduce(function (a, b) { return a + b.weight; }, 0);
    var mxw = top8[0].weight || 1;

    var rows = top8.map(function (v, i) {
      return '<button class="sfw-tft-v" type="button" data-v="' + esc(v.name) + '" data-w="' + v.weight + '" aria-pressed="false">' +
        '<span class="sfw-tft-vn">' + esc(v.name) + '</span>' +
        '<span class="sfw-tft-vt"><i style="width:' + fx(Math.max(2, v.weight / mxw * 100)) + '%"></i></span>' +
        '<span class="sfw-tft-vw">' + (v.weight * 100).toFixed(1) + '%</span>' +
        '</button>';
    }).join('');

    root.innerHTML = [
      '<p class="algo-lead">TFT 把预测拆成四件事：<b>静态协变量</b>（这家店在哪个城市、什么品类）、<b>已知未来输入</b>（星期几、促销排期）、<b>历史观测</b>（过去卖了多少），' +
      '再让<b>变量选择网络</b>自己决定每一步该听谁的，最后用可解释多头注意力在时间轴上打权重。预测完你能指着图说「因为它看了 7 天前那次促销」。</p>',

      '<div class="sfw-tft-block">',
      '<p class="sfw-tft-sub2">① 序列图 · TFT 预测 vs 真值 ' + fakeTag + '</p>',
      '<div class="sfw-legend" style="margin-bottom:6px">' +
      '<span><i style="background:' + C.blue + '"></i>TFT 预测</span>' +
      '<span><i style="background:' + C.truth + '"></i>真值</span></div>',
      '<div data-series></div>',
      '<div class="sfw-tft-num">',
      '<div><div class="sfw-metric">' + wape.toFixed(1) + '%</div><div class="sfw-tft-numlab">14 天 WAPE</div></div>',
      '<div><div class="sfw-metric">' + (sae / d.y_test.length).toFixed(1) + '</div><div class="sfw-tft-numlab">MAE（件/天）</div></div>',
      '<div><div class="sfw-metric">' + d.variable_importance.length + '</div><div class="sfw-tft-numlab">输入变量数</div></div>',
      '</div>',
      '</div>',

      '<div class="sfw-tft-block">',
      '<p class="sfw-tft-sub2">② 变量选择权重 · 点一行看它在生意里是什么</p>',
      '<div class="sfw-tft-vars" data-vars>' + rows + '</div>',
      '<div class="sfw-tft-ex" data-ex></div>',
      '<p class="sfw-cap">权重直接从数据来：前 ' + top8.length + ' 项占 ' +
      (top8.reduce(function (a, b) { return a + b.weight; }, 0) * 100).toFixed(1) + '%' +
      (sorted.length > top8.length ? '，其余 ' + (sorted.length - top8.length) + ' 项合计 ' + (rest * 100).toFixed(1) + '%' : '') +
      '。权重高 = 模型在很多步上都靠它，不是「相关性高」那么含糊。</p>',
      '</div>',

      '<div class="sfw-tft-block">',
      '<p class="sfw-tft-sub2">③ 注意力回溯 · 过去 28 天，它盯着哪天</p>',
      '<div class="sfw-chips sfw-tft-chips" data-chips>' +
      '<button class="sfw-chip is-on" type="button" data-h="1">预测 h=1</button>' +
      '<button class="sfw-chip" type="button" data-h="7">预测 h=7</button>' +
      '<button class="sfw-chip" type="button" data-h="14">预测 h=14</button></div>',
      '<div data-attn></div>',
      '<div class="sfw-tft-detail" data-detail></div>',
      '<p class="sfw-cap" data-attncap></p>',
      '</div>',

      '<p class="algo-note"><b>一句话</b>：TFT = 变量选择 + 静态协变量编码 + 可解释多头注意力。要「为什么这么预测」而不是只要一个数时选它——促销归因、库存解释、跟业务对齐口径，都靠这套。</p>',

      '<dl class="algo-facts">',
      '<div><dt>论文</dt><dd>Lim et al. 2021, IJF · <a href="https://arxiv.org/abs/1912.09363" target="_blank" rel="noopener">arXiv:1912.09363</a></dd></div>',
      '<div><dt>开源</dt><dd><a href="https://github.com/jdb78/pytorch-forecasting" target="_blank" rel="noopener">jdb78/pytorch-forecasting</a></dd></div>',
      '<div><dt>本机复现</dt><dd>' + (d.fake ? '数据文件未产出，本页为示意数据' : '序列 ' + esc(d.series_id)) + '</dd></div>',
      '<div><dt>输入三类</dt><dd>静态协变量 · 已知未来输入 · 历史观测</dd></div>',
      '</dl>'
    ].join('');

    root.querySelector('[data-series]').innerHTML = buildSeries(d).html;

    /* ② 变量 */
    var exBox = root.querySelector('[data-ex]');
    var vBtns = Array.prototype.slice.call(root.querySelectorAll('[data-v]'));
    function showVar(btn) {
      vBtns.forEach(function (b) {
        var on = b === btn;
        b.classList.toggle('is-on', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      var name = btn.getAttribute('data-v');
      var w = parseFloat(btn.getAttribute('data-w'));
      exBox.innerHTML = '<b>' + esc(name) + '</b> · 权重 ' + (w * 100).toFixed(1) + '%<br>' + meaning(name);
    }
    vBtns.forEach(function (b) { b.addEventListener('click', function () { showVar(b); }); });
    if (vBtns.length) showVar(vBtns[0]);

    /* ③ 注意力 */
    var attnBox = root.querySelector('[data-attn]');
    var detail = root.querySelector('[data-detail]');
    var attnCap = root.querySelector('[data-attncap]');
    var chips = Array.prototype.slice.call(root.querySelectorAll('[data-h]'));
    var st = { h: 1, sel: -1 };

    function singleAtt() { return !!(d.attention && d.attention.length >= 10); }

    function attnText(idx) {
      var ws = attWeights(d, st.h);
      var o = ws[idx];
      if (!o) return '';
      var dayVal = d.y_train.length >= o.lag ? d.y_train[d.y_train.length - o.lag] : null;
      var rank = ws.slice().sort(function (a, b) { return b.w - a.w; }).indexOf(o) + 1;
      return '模型做第 <b>' + st.h + '</b> 步预测时，把 <b>' + (o.w * 100).toFixed(1) + '%</b> 的注意力放在了 <b>' +
        o.lag + ' 天前</b>（注意力排名第 ' + rank + '）——那天实际卖了 <b>' + i0(dayVal) + '</b> 件。';
    }

    function renderAttn() {
      var g = buildAttn(d, st.h, st.sel);
      attnBox.innerHTML = g.html;
      detail.innerHTML = attnText(st.sel >= 0 ? st.sel : g.peak);
      if (singleAtt()) {
        var pk = attWeights(d, st.h).slice().sort(function (a, b) { return b.w - a.w; })[0];
        attnCap.textContent = '数据文件只提供一组注意力权重（不分预测步），三个 h 按钮复用同一组并在此标注；峰值落在 ' +
          pk.lag + ' 天前，占 ' + (pk.w * 100).toFixed(1) + '%。';
      } else {
        attnCap.textContent = st.h === 1
          ? '预测明天：权重几乎全压在最近几天，越近越重——短期靠惯性。'
          : (st.h === 7
            ? '预测一周后：峰值挪到 7 天前，模型在找「上周同日」这个锚点。'
            : '预测两周后：峰值挪到 14 天前，同时仍看 7 天前——同星期几的节律在远处更值钱。');
      }
      bindAttn();
    }
    function bindAttn() {
      var svg = attnBox.querySelector('svg');
      if (!svg || svg.getAttribute('data-bound')) return;
      svg.setAttribute('data-bound', '1');
      svg.addEventListener('click', function (e) {
        var t = e.target.closest ? e.target.closest('[data-i]') : null;
        if (!t) return;
        var i = parseInt(t.getAttribute('data-i'), 10);
        if (isNaN(i)) return;
        st.sel = i;
        attnBox.innerHTML = buildAttn(d, st.h, st.sel).html;
        detail.innerHTML = attnText(i);
        bindAttn();
      });
    }

    chips.forEach(function (c) {
      c.addEventListener('click', function () {
        chips.forEach(function (x) { x.classList.toggle('is-on', x === c); });
        st.h = parseInt(c.getAttribute('data-h'), 10) || 1;
        var g = buildAttn(d, st.h, -1);
        st.sel = g.peak;
        renderAttn();
      });
    });

    renderAttn();
  }

  window.SF = window.SF || {};
  window.SF.algo = window.SF.algo || {};
  window.SF.algo[ID] = {
    id: ID,
    title: 'TFT',
    tagline: '变量选择 + 注意力：能说清「为什么」',
    mount: mount,
    _geom: { synth: synth, series: buildSeries, attn: buildAttn, weights: attWeights, vars: function () { return VARS; } }
  };
})();
