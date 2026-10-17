# 合并约定（给下一个接手这个工作台的 agent 看）

这个站点有两个来源的内容，**不要互相覆盖**：

## 1. `index.html` —— 工作台主页面（原来的 Agent 维护）
- 结构：4 个 tab（算法讲解 / 实测成绩 / 论文库 / 怎么用）+ 底部导航
- 数据：`data/bench.json`（排行榜）、`data/papers.json`、`data/series.json`、`data/*.json`（各算法交互数据）
- 交互：`js/app.js` + `js/algo-*.js`
- 改这里之前先看 `CONTRACT.md`

## 2. `principles.html` —— 从零开始（机制详解）
- 单文件，纯内联 CSS/JS/SVG，不依赖 `css/`、`js/`、`data/`
- 内容：28 天手算 4 种基线 → ETS/Theta → ARIMA → LightGBM 残差提升与特征构造 → 5 个深度模型计算图 → BSTS/卡尔曼
- 由「机制详解」那一路维护，**只往这个文件里加内容**；index 只保留一个入口链接

## 合并规则
1. 排名/指标数字只有一份真相：`data/bench.json`。任何新模型跑完，先写进 bench.json，再考虑在 principles.html 里补讲解。
2. 指标口径必须一致：周度 = 359 条、预测 13 周、逐序列 WAPE 取均值、SeasonalNaive 季节周期 52。
   新脚本必须在同一批序列上重算 SeasonalNaive 并复现 15.55%，否则口径不一致，不能入榜。
3. 发布方式：`publish.py publish --source-dir web`（整站），不要单发 `index.html`，否则 css/js/data 会 404。
4. 版本号写在 `#built` 里，格式 `v1.x · 日期`；改完 index 顺手 +0.1。
