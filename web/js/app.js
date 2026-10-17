/* 工作台主控：导航、数据加载、算法挂载、成绩表、论文库 */
(function () {
  'use strict';
  var ORDER = ['lightgbm', 'nhits', 'deepar', 'tft', 'patchtst'];
  var cache = {};
  var DATA = {};

  function load(name) {
    if (cache[name]) return cache[name];
    cache[name] = fetch('data/' + name, { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .catch(function () { return null; });
    return cache[name];
  }

  var fmt = {
    pct: function (v, d) { return (v == null || isNaN(v)) ? '–' : (v * 100).toFixed(d == null ? 1 : d) + '%'; },
    num: function (v, d) { return (v == null || isNaN(v)) ? '–' : (+v).toFixed(d == null ? 2 : d); },
    sec: function (v) { if (v == null) return '–'; return v < 60 ? v.toFixed(1) + 's' : (v / 60).toFixed(1) + 'min'; }
  };

  /* ---------- 导航 ---------- */
  function selectPanel(id) {
    document.querySelectorAll('.panel').forEach(function (p) {
      p.classList.toggle('is-on', p.id === id);
    });
    document.querySelectorAll('[data-panel]').forEach(function (b) {
      b.setAttribute('aria-selected', String(b.getAttribute('data-panel') === id));
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (location.hash !== '#' + id) history.replaceState(null, '', '#' + id);
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest('[data-panel]');
    if (b) selectPanel(b.getAttribute('data-panel'));
  });

  /* ---------- 算法挂载 ---------- */
  function mountAlgos() {
    var host = document.getElementById('algo-list');
    var reg = (window.SF && window.SF.algo) || {};
    var ids = ORDER.filter(function (k) { return reg[k]; })
      .concat(Object.keys(reg).filter(function (k) { return ORDER.indexOf(k) < 0; }));
    host.innerHTML = '';
    if (!ids.length) {
      host.innerHTML = '<p class="sfw-cap">算法模块未加载。</p>';
      return;
    }
    ids.forEach(function (id, i) {
      var mod = reg[id];
      var sec = document.createElement('section');
      sec.className = 'algo';
      sec.id = 'algo-' + id;
      sec.innerHTML =
        '<header class="algo-hd">' +
        '<span class="algo-rank">' + String(i + 1).padStart(2, '0') + '</span>' +
        '<h2 class="algo-name"></h2><p class="algo-tag"></p></header>' +
        '<div class="algo-body"></div>';
      sec.querySelector('.algo-name').textContent = mod.title || id;
      sec.querySelector('.algo-tag').textContent = mod.tagline || '';
      host.appendChild(sec);
      try {
        var p = mod.mount(sec.querySelector('.algo-body'), { load: load, data: DATA, fmt: fmt });
        if (p && p.catch) p.catch(function (err) {
          console.error('mount failed', id, err);
          sec.querySelector('.algo-body').insertAdjacentHTML('beforeend',
            '<p class="sfw-cap">这个模块渲染出错：' + String(err && err.message || err) + '</p>');
        });
      } catch (err) {
        console.error(err);
      }
    });
  }

  /* ---------- 实测成绩 ---------- */
  function svgBars(models) {
    var W = 720, H = 260, pad = { l: 8, r: 8, t: 18, b: 46 };
    var rows = models.filter(function (m) { return m.wape != null; })
      .slice().sort(function (a, b) { return a.wape - b.wape; });
    if (!rows.length) return '';
    var max = Math.max.apply(null, rows.map(function (m) { return m.wape; })) * 1.15 || 1;
    var iw = (W - pad.l - pad.r) / rows.length;
    var bw = Math.min(64, iw * 0.6);
    var best = rows[0].wape;
    var out = ['<svg class="sfw-plot" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="各算法 WAPE 对比">'];
    for (var g = 0; g <= 3; g++) {
      var y = pad.t + (H - pad.t - pad.b) * g / 3;
      out.push('<line class="grid" x1="' + pad.l + '" y1="' + y + '" x2="' + (W - pad.r) + '" y2="' + y + '"/>');
    }
    rows.forEach(function (m, i) {
      var h = (H - pad.t - pad.b) * (m.wape / max);
      var x = pad.l + iw * i + (iw - bw) / 2;
      var y2 = H - pad.b - h;
      var isBest = m.wape === best;
      out.push('<rect x="' + x + '" y="' + y2 + '" width="' + bw + '" height="' + Math.max(h, 2) +
        '" rx="6" fill="' + (isBest ? '#1D4ED8' : '#BFD2F6') + '"/>');
      out.push('<text x="' + (x + bw / 2) + '" y="' + (y2 - 6) + '" text-anchor="middle" fill="' +
        (isBest ? '#1D4ED8' : '#5A6B85') + '" font-weight="700">' + (m.wape * 100).toFixed(1) + '</text>');
      out.push('<text x="' + (x + bw / 2) + '" y="' + (H - pad.b + 16) + '" text-anchor="middle">' +
        m.short.replace(/[<>&]/g, '') + '</text>');
    });
    out.push('<text x="' + pad.l + '" y="10">WAPE 越低越好（%）</text></svg>');
    return out.join('');
  }

  function renderBench(b) {
    var host = document.getElementById('bench-body');
    if (!b || !b.models || !b.models.length) {
      host.innerHTML = '<p class="sfw-cap">实测数据还在跑（模型训练中），刷新可看最新结果。</p>';
      return;
    }
    var ds = b.dataset || {};
    document.getElementById('bench-sub').textContent =
      '数据集：' + (ds.name || 'M4') + '，' + (ds.n_series || '?') + ' 条序列，预测 ' +
      (ds.horizon || '?') + ' 步；同一套切分、同一套指标。';
    var rows = b.models.slice().sort(function (a, c) { return (a.wape == null) - (c.wape == null) || a.wape - c.wape; });
    var html = svgBars(rows);
    html += '<div class="tablewrap" style="margin-top:14px"><table><thead><tr>' +
      '<th>模型</th><th class="num">WAPE</th><th class="num">sMAPE</th><th class="num">MASE</th>' +
      '<th class="num">耗时</th></tr></thead><tbody>';
    var bestW = rows.filter(function (m) { return m.wape != null; }).map(function (m) { return m.wape; });
    var bW = bestW.length ? Math.min.apply(null, bestW) : null;
    rows.forEach(function (m) {
      html += '<tr><td>' + m.name + (m.reproduced === false ? ' <span class="pill">论文口径</span>' : '') + '</td>' +
        '<td class="num' + (m.wape === bW ? ' best' : '') + '">' + fmt.pct(m.wape) + '</td>' +
        '<td class="num">' + fmt.pct(m.smape) + '</td>' +
        '<td class="num">' + fmt.num(m.mase, 3) + '</td>' +
        '<td class="num">' + fmt.sec(m.runtime_s) + '</td></tr>';
    });
    html += '</tbody></table></div>';
    if (b.pending && b.pending.length) {
      html += '<div class="algo-note" style="margin-top:12px"><b>未跑完，不计入排名</b><ul style="margin:6px 0 0 18px;padding:0">';
      b.pending.forEach(function (x) {
        html += '<li><b>' + x.name + '</b>：' + x.why + '</li>';
      });
      html += '</ul></div>';
    }
    html += '<p class="sfw-cap">' + (b.note || '指标由本机脚本实测，原始输出见 repro/metrics_*.json。') + '</p>';
    host.innerHTML = html;
  }

  /* ---------- 论文库 ---------- */
  function renderPapers(p) {
    var host = document.getElementById('papers-body');
    var list = (p && (p.papers || p)) || null;
    if (!Array.isArray(list) || !list.length) {
      host.innerHTML = '<p class="sfw-cap">论文数据加载中…</p>';
      return;
    }
    document.getElementById('papers-sub').textContent =
      '共 ' + list.length + ' 篇：综述 ' + list.filter(function (x) { return x.type === 'survey'; }).length +
      ' 篇，方法/竞赛论文 ' + list.filter(function (x) { return x.type !== 'survey'; }).length + ' 篇。链接均可点开。';
    var fam = {};
    list.forEach(function (x) { (fam[x.algo_family] = fam[x.algo_family] || []).push(x); });
    var order = Object.keys(fam).sort(function (a, b) { return fam[b].length - fam[a].length; });
    var html = '';
    order.forEach(function (k) {
      html += '<h3 class="sfw-cap" style="margin:18px 0 6px;color:#1D4ED8;font-weight:800;font-size:13px;letter-spacing:.5px;text-transform:uppercase">' +
        k + ' · ' + fam[k].length + ' 篇</h3><div class="papers">';
      fam[k].forEach(function (x) {
        html += '<article class="paper"><div class="paper-id">' + (x.id || '') + '</div><div>' +
          '<h3><a href="' + x.url + '" target="_blank" rel="noopener">' + x.title + '</a></h3>' +
          '<div class="meta">' + (x.authors || '') + ' · ' + (x.venue || '') + ' ' + (x.year || '') +
          (x.type === 'survey' ? ' <span class="pill">综述</span>' : '') + '</div>' +
          '<p class="why">' + (x.why_matters_for_sales_forecasting || x.key_contribution || '') + '</p>' +
          '</div></article>';
      });
      html += '</div>';
    });
    host.innerHTML = html;
  }

  /* ---------- 怎么用 ---------- */
  function renderHow() {
    var host = document.getElementById('how-body');
    host.innerHTML =
      '<section class="algo"><header class="algo-hd"><span class="algo-rank">A</span>' +
      '<h2 class="algo-name" style="font-size:17px">从哪开始</h2></header>' +
      '<p class="algo-lead">先跑<b>季节性朴素基线</b>，再上 LightGBM 全局模型。两个数字拿到手，你才知道深度模型的提升值不值那个工程成本。</p>' +
      '<p class="algo-note">经验顺序：SeasonalNaive → LightGBM(全局 + lag/rolling 特征) → N-HiTS/DeepAR(需要概率区间时) → TFT(需要可解释变量归因时)</p></section>' +
      '<section class="algo"><header class="algo-hd"><span class="algo-rank">B</span>' +
      '<h2 class="algo-name" style="font-size:17px">评价指标别选错</h2></header>' +
      '<p class="algo-lead">销量预测优先看 <b>WAPE</b>（加权绝对百分比误差，对总量敏感、零值友好）；MASE 用来和基线比"比 naive 好多少"；RMSE 只在你想重罚大误差时用。sMAPE 在接近 0 的序列上会爆炸，别单独用。</p></section>' +
      '<section class="algo"><header class="algo-hd"><span class="algo-rank">C</span>' +
      '<h2 class="algo-name" style="font-size:17px">三个常见坑</h2></header>' +
      '<p class="algo-lead">① <b>随机切分</b>：时序必须按时间切，随机切会让 lag 特征泄漏未来。<br>' +
      '② <b>递归多步</b>：h 步里每一步都用上一步的预测当特征，误差会累积——本工作台的 LightGBM 演示就是让你亲手看这个累积。<br>' +
      '③ <b>间歇需求</b>：大量 0 的 SKU 用 log1p/普通 L2 会预测出 0.3 这种无意义值，改用 Tweedie/Poisson 目标或 Croston。</p></section>' +
      '<section class="algo"><header class="algo-hd"><span class="algo-rank">D</span>' +
      '<h2 class="algo-name" style="font-size:17px">数据与代码</h2></header>' +
      '<p class="algo-lead">本工作台所有数字都是在本机跑出来的：数据集为 M4 竞赛的 Weekly 子集（359 条真实周度销量序列，每条取最近 200 周，预测未来 13 周），同一套切分、同一套指标。代码与原始输出都在 <code>repro/</code> 与 <code>web/data/</code> 下。</p>' +
      '<p class="algo-note">复现脚本：<code>repro/baselines.py</code>、<code>repro/lgbm_global.py</code>、<code>repro/deep_models.py</code>、<code>repro/bsts52.py</code>；指标：<code>web/data/bench.json</code></p></section>';
  }

  /* ---------- 启动 ---------- */
  Promise.all([
    load('bench.json'), load('papers.json'), load('how.json'),
    load('lgbm.json'), load('nhits.json'), load('deepar.json'), load('tft.json'), load('patchtst.json'), load('series.json')
  ]).then(function (r) {
    DATA.bench = r[0]; DATA.papers = r[1]; DATA.how = r[2];
    DATA.lgbm = r[3]; DATA.nhits = r[4]; DATA.deepar = r[5]; DATA.tft = r[6]; DATA.patchtst = r[7]; DATA.series = r[8];
    mountAlgos();
    renderBench(r[0]);
    renderPapers(r[1]);
    renderHow(r[2]);
    var b = r[0];
    document.getElementById('built').textContent =
      '构建于 ' + new Date().toISOString().slice(0, 16).replace('T', ' ') + ' UTC · 数据 ' +
      ((b && b.dataset && b.dataset.name) || 'M4') + ' · 全部指标为本机实测';
    if (location.hash) selectPanel(location.hash.slice(1));
  });
})();
