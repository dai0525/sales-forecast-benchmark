> 历史前端协作约定。旧机器绝对路径仅供追溯；当前仓库以相对路径运行。审计、数据口径和重现入口以 README.md / AUDIT.md 为准。

# 工作台前端契约（必须严格遵守）

目标：手机优先（竖屏 390px 宽 / 横屏 844×390 都要能看能点），纯静态，无框架、无 CDN、无网络请求。
页面由主控（main agent）组装 `index.html`；你只负责**一个 JS 文件**里的自绘组件。

## 文件位置
- 你的组件：`/opt/data/workspace/sales-forecast/web/js/algo-<ID>.js`（ID 见任务）
- 已存在可读数据：`/opt/data/workspace/sales-forecast/web/data/*.json`（同目录的相对路径，运行时用 `fetch('data/xxx.json')`）
- 不允许修改：`web/index.html`、`web/css/app.css`、其它人的 js

## 组件接口
```js
(function(){
  const id = 'lightgbm';
  async function mount(root, ctx){ ... }   // ctx = { data: <已预取的 JSON 合集>, fmt: helpers }
  window.SF = window.SF || {};
  window.SF.algo = window.SF.algo || {};
  window.SF.algo[id] = { id, title:'LightGBM', tagline:'一行副标题', mount };
})();
```
- `mount(root, ctx)` 里 `root` 是一个空 `<section class="algo">`，你全权渲染。
- 异步取数据用 `ctx.load('lgbm.json')`（已实现，返回 Promise<object>；失败返回 null）。
- **禁止**：`document.write`、全局样式污染（不要写 `<style>` 里选择器不加 `.sfw-` 前缀）、`innerHTML` 拼接用户输入、外链请求、jQuery/d3 等。
- 必须处理数据缺失：拿不到就用标注「示意数据」的合成曲线，绝不能白屏。

## DOM 结构（主控已定义，按此渲染）
```html
<section class="algo" id="algo-lightgbm">
  <header class="algo-hd">
    <span class="algo-rank">01</span>
    <h2 class="algo-name">LightGBM</h2>
    <p class="algo-tag">为什么 M5 冠军用的就是它</p>
  </header>
  <p class="algo-lead">3-5 行：直觉、核心机制、适用数据形态</p>
  <div class="sfw-...">自定义交互区</div>
  <dl class="algo-facts">
    <div><dt>论文</dt><dd>…</dd></div>
    <div><dt>开源</dt><dd><a href="…" target="_blank">github.com/…</a></dd></div>
    <div><dt>本机复现</dt><dd>WAPE 8.2% · 42s</dd></div>
  </dl>
  <p class="algo-note">一句话：什么场景该用它 / 不该用它</p>
</section>
```

## 可复用 CSS 类（已在 app.css 定义，直接用）
- 布局：`sfw-row` `sfw-col` `sfw-grid2`（窄屏自动单列）
- 控件：`sfw-btn`（次要按钮）`sfw-btn is-on`（选中）`sfw-slider`（range input）`sfw-chips` + `sfw-chip`
- 数值：`sfw-metric`（大数字）`sfw-metric-u`（单位，小字）
- 图例：`sfw-legend` `sfw-legend i`（16px 色块）
- 说明：`sfw-cap`（12px 灰注释）
- 你的私有样式一律 `.sfw-<ID>-*` 前缀，写在 JS 里注入一个 `<style>` 也可以，但必须以 `.sfw-<ID>-` 开头。

## 配色（蓝白，唯一强调色）
- 强调蓝 `#1D4ED8`，浅蓝底 `#EFF4FF`，浅蓝线 `#D6E2FB`
- 正文墨 `#0F1B2D`，次要 `#5A6B85`，网格线 `#E6EBF3`
- 不要暖奶油底、不要红/橙做主色、不要近黑+荧光色。多序列对比可用蓝的同色阶（#1D4ED8 / #60A5FA / #BFD2F6）+ 一条中性灰虚线（真实值）。

## 交互质量硬要求
1. 触控目标 ≥ 40px 高；range 滑块要能拖动且实时更新（`input` 事件）。
2. 每次交互必须**看得见数字变化**（WAPE / 误差 / 区间宽度），不能只是动画。
3. 图表用内联 SVG，`viewBox` + `width:100%`，文字 ≥ 11px；横屏时自动变宽不溢出。
4. 首屏渲染 < 200ms，数据文件 > 200KB 时降采样到 ≤ 300 点。
5. 每个算法 1–2 个「例子」：一个真实 M4 序列（来自 `series.json`）+ 一个可调参数的沙盒。
6. 关键结论用一句人话写在图下面（`.sfw-cap`），别让用户自己悟。

## 验收
- `node --check web/js/algo-<ID>.js` 通过
- 本地起 `python3 -m http.server` 后，用无头浏览器截图 390×844 与 844×390 各一张，确认不溢出、不重叠
- 最终回复里贴：文件路径、节点行数、截图路径、每个交互的一句话描述
