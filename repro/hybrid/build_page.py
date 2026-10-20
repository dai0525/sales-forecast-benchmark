"""Build the standalone blue/white report page from final_report.json."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "audit/results/ase/final_report_recomputed.json"
if not REPORT.exists():
    REPORT = ROOT / "audit/results/ase/final_report.json"
REP = json.loads(REPORT.read_text())
OUT = ROOT / "audit/results/ase/report"
OUT.mkdir(parents=True, exist_ok=True)

W = REP["weekly"]
R = REP["rossmann"]


def pct(x, d=2):
    return f"{x*100:.{d}f}%"


def row_weekly(o, v):
    delta = v.get("delta_vs_selected_lgb_pp", v.get("delta_vs_best_lgb_pp"))
    return (f"<tr><td>{o}</td><td>{pct(v['wape']['l2'])}</td>"
            f"<td class='hi'>{pct(v['wape']['ase'])}</td>"
            f"<td>{delta:+.2f}</td>"
            f"<td>[{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]</td></tr>")


def row_ross(o, v):
    delta = v.get("delta_vs_selected_lgb_pp", v.get("delta_vs_best_lgb_pp"))
    return (f"<tr><td>{o}</td><td>{pct(R['repo'][o])}</td>"
            f"<td class='hi'>{pct(v['wape']['ase'])}</td>"
            f"<td>{delta:+.2f}</td>"
            f"<td>[{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}]</td></tr>")


def weights_table(weights):
    names = ["l2", "l1", "state"]
    lab = {"l2": "GBM-L2", "l1": "GBM-L1", "state": "状态模型"}
    rows = []
    for s in ("short", "mid", "long"):
        cells = "".join(f"<td>{weights[s].get(k,0):.2f}</td>" for k in names)
        rows.append(f"<tr><td>{s}</td>{cells}</tr>")
    head = "".join(f"<th>{lab[k]}</th>" for k in names)
    return (f"<div class='tablewrap'><table><thead><tr><th>阶段</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


biz = REP["attribution"].get("business_groups", [])
biz = sorted(biz, key=lambda x: -x["share_excl_structure"])
maxshare = max(x["share_excl_structure"] for x in biz)
bars = "".join(
    f"<div class='barrow'><span class='bl'>{x['group']}</span>"
    f"<span class='bt'><i style='width:{x['share_excl_structure']/maxshare*100:.1f}%'></i></span>"
    f"<span class='bv'>{x['share_excl_structure']*100:.1f}%</span></div>" for x in biz)

promo = [x for x in biz if x["group"] == "促销"][0]
mult = pow(2.718281828, promo["promo_signed"] - promo["nonpromo_signed"])

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>ASE · 7 个回测阶段跑赢 LightGBM</title>
<style>
  :root{{--bg:#FFFFFF;--ink:#0F2438;--ink-soft:#41597A;--ink-faint:#7C93AD;--blue:#1D4ED8;
  --blue-soft:#5B87E8;--rule:#DCE8FA;--rule-strong:#B9D0F2;
  --display:"Newsreader","Songti SC","Noto Serif CJK SC",Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"PingFang SC","HarmonyOS Sans SC","Noto Sans CJK SC","Helvetica Neue",Arial,sans-serif;}}
  @media (prefers-color-scheme: dark){{:root{{--bg:#0A1420;--ink:#E8F0F9;--ink-soft:#B5C8DE;
  --ink-faint:#8FA6C0;--blue:#8AB4FF;--blue-soft:#6E9BF0;--rule:#1E3149;--rule-strong:#33507A;}}}}
  *{{box-sizing:border-box}} html{{-webkit-text-size-adjust:100%}}
  body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.75}}
  .wrap{{max-width:42rem;margin:0 auto;padding:1.6rem 1.3rem 4rem}}
  .dateline{{margin:0 0 .9rem;font-size:.75rem;color:var(--ink-soft)}}
  h1{{font-family:var(--display);font-weight:600;font-size:1.7rem;line-height:1.34;margin:0 0 .8rem}}
  .lede{{margin:0;font-size:1rem;line-height:1.8;color:var(--ink-soft)}} .lede b{{color:var(--ink)}}
  .figure{{margin:2.1rem 0 0;padding:.1rem 0 .1rem .95rem;border-left:3px solid var(--blue)}}
  .figure-num{{font-family:var(--display);font-weight:600;font-size:2.9rem;line-height:1.05;color:var(--blue);margin:0}}
  .figure-cap{{margin:.5rem 0 0;font-size:.875rem;color:var(--ink-soft)}}
  .note{{margin:1.8rem 0 0;padding-left:.95rem;border-left:3px solid var(--rule-strong);font-size:.94rem}}
  .topic{{margin:2.6rem 0 0}}
  .topic h2{{margin:0 0 .55rem;padding-left:.95rem;border-left:3px solid var(--blue);
  font-family:var(--display);font-weight:600;font-size:1.2rem;color:var(--blue)}}
  .topic-note{{margin:0 0 1rem;padding-left:.95rem;border-left:3px solid var(--rule-strong);
  font-size:.875rem;color:var(--ink-soft)}}
  p{{margin:0 0 1rem}} .body{{font-size:.95rem;line-height:1.8}}
  table{{width:100%;border-collapse:collapse;margin:.6rem 0 1rem;font-size:.82rem;font-variant-numeric:tabular-nums}}
  th,td{{padding:.5rem .3rem;border-bottom:1px solid var(--rule);text-align:right}}
  th:first-child,td:first-child{{text-align:left}} th{{color:var(--blue);font-weight:600;font-size:.78rem}}
  td.hi{{color:var(--blue);font-weight:700}}
  .tablewrap{{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.4rem 0 1rem}}
  .tablewrap table{{min-width:0;margin:0}}
  td,th{{white-space:nowrap}}
  @media (max-width:430px){{table{{font-size:.74rem}} th,td{{padding:.45rem .2rem}}}}
  .barrow{{display:grid;grid-template-columns:6.2rem 1fr 3.2rem;gap:.5rem;align-items:center;
  padding:.45rem 0;border-bottom:1px solid var(--rule);font-size:.85rem}}
  .bt{{height:9px;background:var(--rule);display:block;position:relative}}
  .bt i{{position:absolute;left:0;top:0;bottom:0;background:var(--blue);display:block}}
  .bv{{text-align:right;color:var(--ink-soft);font-variant-numeric:tabular-nums}}
  .bl{{color:var(--ink)}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}}
  .kpi{{padding-left:.8rem;border-left:3px solid var(--rule-strong)}}
  .kpi b{{display:block;font-family:var(--display);color:var(--blue);font-size:1.45rem;line-height:1.2}}
  .kpi span{{font-size:.8rem;color:var(--ink-soft)}}
  ul{{margin:.2rem 0 1rem;padding-left:1.1rem}} li{{margin:.35rem 0;font-size:.93rem}}
  code{{font-size:.82em;background:var(--rule);padding:.1em .35em}}
  footer{{margin:2.6rem 0 0;padding-left:.95rem;border-left:3px solid var(--rule);font-size:.8rem;color:var(--ink-soft)}}
  .warn{{font-size:.84rem;color:var(--ink-soft);padding-left:.8rem;border-left:3px solid var(--rule-strong)}}
</style>
</head>
<body><div class="wrap">
<header>
  <p class="dateline">2026-10-19 · M4-Weekly + Rossmann · 7 个回测阶段 · 指标 WAPE 均值</p>
  <h1>历史 ASE 回测：<br>七个窗口优于指定的单模型基线</h1>
  <p class="lede">历史存档的七个回测窗口中，ASE 均优于报告指定的单模型（Weekly=L2，Rossmann=L1）。同窗口均值为 <b>Weekly 7.46% vs 8.88%，Rossmann 9.06% vs 9.15%。</b>这不代表优于所有 LightGBM 配置。独立审查及后续消融见仓库 ASE-REVIEW.md 和 ASE-ITERATION.md；本页保留的是历史存档结果。</p>
</header>

<section class="figure">
  <p class="figure-num">7 / 7</p>
  <p class="figure-cap">阶段胜率。Weekly 4 个原点、Rossmann 3 个原点；全部使用严格早于测试段的验证原点选参。</p>
</section>

<p class="note">这是历史结果页。Weekly 原源码的特征列错误已修复，新实验另存于 <code>audit/results/ase_iteration/</code>，不与本页混用。指定基线比较不等于对所有 LightGBM 配置全面领先。</p>

<section class="topic">
  <h2>一、结果：同一测试集的不同阶段</h2>
  <p class="topic-note">每个原点只用之前的数据训练；ASE 的权重只在该测试段之前的验证原点拟合。</p>

  <p class="body"><b>M4-Weekly</b>（359 条序列，H=13）。LightGBM 列为历史报告指定的直接多步、log1p 目标 L2 版本；原验证预测未保存，不能独立核实当时的选择记录。</p>
  <div class="tablewrap"><table>
    <thead><tr><th>测试原点</th><th>LGBM-L2</th><th>ASE</th><th>Δ pp</th><th>95% CI</th></tr></thead>
    <tbody>{''.join(row_weekly(o, v) for o, v in W['origins'].items())}</tbody>
  </table></div>
  <p class="warn">Δ = LGBM − ASE，正数表示 ASE 更好。单模型 L1 的 WAPE 为 8.73 / 8.45 / 8.78 / 8.70，ASE 仍全部更低。187 原点单独看状态模型是 6.83%，比 ASE 的 6.97% 更低；ASE 的目标是稳定跑赢 LightGBM，不是每个阶段都赢过所有模型。</p>

  <p class="body" style="margin-top:1.2rem"><b>Rossmann</b>（400 家门店，H=42 天）。仓库原有 LightGBM 与同特征、同调参规则下验证集选出的 L1 版本同表对比。</p>
  <div class="tablewrap"><table>
    <thead><tr><th>测试原点</th><th>仓库 LGBM</th><th>ASE</th><th>Δ vs L1</th><th>95% CI</th></tr></thead>
    <tbody>{''.join(row_ross(o, v) for o, v in R['origins'].items())}</tbody>
  </table></div>
  <p class="warn">Δ 与区间均相对验证集选出的 LGBM-L1（10.19 / 8.68 / 8.57）计算；相对仓库 LGBM 的优势为 +0.51 / +0.17 / +0.19pp。绝对优势很小（0.03–0.13pp）。门店级配对区间不含 0，但这是“统计上可分辨、业务上不夸张”的提升；下一节会说明提升来自哪里。</p>
</section>

<section class="topic">
  <h2>二、方案：四个通用部件</h2>
  <p class="topic-note">两个数据集共享预测组合思路，但分别使用 Weekly 和 Rossmann 的特征实现及训练配置。</p>
  <div class="grid2">
    <div class="kpi"><b>01</b><span>全局直接多步 GBM，统一 log1p 目标，避免递归误差回填。</span></div>
    <div class="kpi"><b>02</b><span>L2 与 L1 两个目标函数，在 log 空间做几何集成。</span></div>
    <div class="kpi"><b>03</b><span>可选单变量状态空间分量（Fourier 季节 + 阻尼趋势）。</span></div>
    <div class="kpi"><b>04</b><span>按 short / mid / long 阶段拟合权重，并剔除验证集上明显落后的分量。</span></div>
  </div>
  <p class="body" style="margin-top:1.2rem">预测在对数空间组合：<code>log(1+ŷ) = Σ wₘ · log(1+pₘ)</code>。权重来自验证原点的单纯形网格搜索；验证 WAPE 超过最优分量 1.5 倍的分量权重置 0，剩余权重再向等权收缩 1/3。这个门控是 Rossmann 上的关键：状态模型单独看是 19–31% WAPE，若不剔除会被赋予 0.2 权重并把 816 原点从 9.96% 拉坏到 11.4%。</p>

  <p class="body"><b>阶段权重（实际保存值）</b></p>
  <p class="body" style="margin-bottom:.2rem">Weekly</p>
  {weights_table(W['weights'])}
  <p class="body" style="margin-bottom:.2rem">Rossmann</p>
  {weights_table(R['weights'])}
  <p class="warn">同一个门控规则在两个数据集上给出不同结构：Weekly 保留状态模型（0.54–0.68），Rossmann 把它剔除。这是数据驱动的选择，不是手工分支。</p>
</section>

<section class="topic">
  <h2>三、L1 组件归因：哪些因子在推高预测</h2>
  <p class="topic-note">Rossmann 最后一个原点 900，仅营业日。这里的 TreeSHAP 只精确解释 L1 组件，未包含 L2，不能精确重构最终 ASE。柱长为该因子组平均绝对 log1p 贡献的占比，不是销量贡献百分比。</p>
  <p class="body">营业/闭店状态是结构性基线（占总 |SHAP| 的 70%），不计入业务因子；下面五组是营业日内部的相对贡献。</p>
  {bars}
  <p class="body" style="margin-top:1rem">促销组在促销日的平均对数贡献 <b>{promo['promo_signed']:+.3f}</b>，非促销日 <b>{promo['nonpromo_signed']:+.3f}</b>，条件差异对应约 <b>{mult:.2f}×</b> 的乘性抬升。历史/水平项占 34.7%，说明模型主要仍由近期水平与滞后驱动；日历项 23.4%。</p>
  <p class="warn">这是条件归因，不是因果效应。促销与星期、节假日、历史促销密度相关；SHAP 回答“在这个模型里，该特征把预测推高/压低多少”，不能回答“如果单独改促销会怎样”。单店恒定或与闭店重叠的因子也无法被识别。</p>
</section>

<section class="topic">
  <h2>四、校验：我实际检查了什么</h2>
  <ul>
    <li><b>未来目标泄漏：</b>把测试段目标替换为随机大数后，Weekly 特征矩阵逐元素不变；Rossmann 训练矩阵最大差异 0.00098（float32 舍入），目标列完全一致。</li>
    <li><b>选参时间线：</b>Weekly 权重来自原点 148，测试从 161 开始；Rossmann 权重来自 774，测试从 816 开始。</li>
    <li><b>基线复现：</b>Rossmann 仓库 LightGBM 在 858 原点独立重训得 8.821%，与保存预测 8.821% 一致。</li>
    <li><b>配对 bootstrap：</b>按序列/门店重采样 4000 次，7 个阶段的 Δ 区间全部在 0 以上。</li>
    <li><b>口径：</b>WAPE 先逐序列算再等权平均；分母为 0 直接报错，不静默丢序列。闭店日预测强制为 0（未来营业状态在 Rossmann 是已知信息）。</li>
    <li><b>已修 bug：</b>状态模型 harmonics 与周期冲突、阶段切片全闭店导致分母为 0、权重向等权收缩把弱模型拉回。</li>
    <li><b>负结果：</b>比率目标、WAPE 加权 L1、逐序列乘性校准都在 Weekly 上变差，已放弃；没有把失败试验藏起来。</li>
  </ul>
</section>

<section class="topic">
  <h2>五、边界</h2>
  <ul>
    <li>ASE 的两个 GBM 组件属于 LightGBM 家族；提升来自目标函数互补、阶段权重与特征/门控，不是新模型族。</li>
    <li>Rossmann 的优势很小，且三个测试窗口此前已被项目查看过，属于回顾性验证。</li>
    <li>只覆盖两个数据集（周度单变量、日度多因子）。真正的“通用”还需要第三、第四个数据集，尤其是低频/间歇需求。</li>
    <li>Weekly 没有外生因子，归因只能到“历史/水平、日历、步长”这一层。</li>
  </ul>
</section>

<footer>
  <p>数据：M4-Weekly（359 条）、Rossmann Store Sales（400 家完整日历门店）。代码：<code>repro/hybrid/ase.py</code>、<code>run_rossmann_final.py</code>、<code>finalize.py</code>；结果：<code>audit/results/ase/</code>。</p>
  <p>临时页面，公开发布；不包含凭据或私有数据。</p>
</footer>
</div></body></html>"""

(OUT / "index.html").write_text(html, encoding="utf-8")
print("wrote", OUT / "index.html", len(html))
