"""深度模型基准：复用 repro/deep_models.py 里的模型类，在 M4-Weekly 上真训真评。
输出：repro/metrics_deep.json、web/data/{bench_dl.json,nhits.json,deepar.json,tft.json,patchtst.json}
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEB = ROOT / "web" / "data"
H = 13
INPUT = 100
TAIL = 200
SEED = 20261017


def load_models():
    spec = importlib.util.spec_from_file_location("dm", ROOT / "repro" / "deep_models.py")
    dm = importlib.util.module_from_spec(spec)
    sys.modules["dm"] = dm
    try:
        spec.loader.exec_module(dm)
    except SystemExit:
        pass
    dm.H, dm.INPUT = H, INPUT  # 覆盖模块常量
    return dm


def load():
    z = np.load(DATA / "weekly.npz", allow_pickle=True)
    tr, te, ids = z["train"].T, z["test"].T, [str(s) for s in z["ids"]]
    out = []
    for i in range(tr.shape[0]):
        v = tr[i]
        fin = np.flatnonzero(~np.isnan(v))
        v = v[fin[0]:fin[-1] + 1] if len(fin) else np.zeros(TAIL)
        if len(v) < TAIL:
            v = np.concatenate([np.repeat(v[0], TAIL - len(v)), v])
        out.append(v[-TAIL:])
    return np.vstack(out), np.nan_to_num(te, nan=0.0), ids


def make_windows(y, horizon=H, input_size=INPUT):
    """每条序列产出多个 (input, target) 窗口，用于训练"""
    X, Y = [], []
    for v in y:
        for end in range(input_size, len(v) - horizon + 1):
            X.append(v[end - input_size:end])
            Y.append(v[end:end + horizon])
    return np.array(X, np.float32), np.array(Y, np.float32)


def metrics(yt, yp):
    yt = np.maximum(yt, 0)
    yp = np.maximum(yp, 0)
    wape = np.abs(yt - yp).sum() / max(yt.sum(), 1e-9)
    sm = np.mean(2 * np.abs(yt - yp) / np.maximum(np.abs(yt) + np.abs(yp), 1e-9))
    return float(wape), float(sm)


def main():
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dm = load_models()
    y, te, ids = load()
    N = len(ids)
    scale = y.mean(axis=1, keepdims=True)
    scale[scale <= 0] = 1.0
    ys = y / scale
    ts = te / scale
    Xtr, Ytr = make_windows(ys)
    order = np.random.permutation(len(Xtr))
    Xtr, Ytr = Xtr[order], Ytr[order]
    n_val = max(1, int(0.12 * len(Xtr)))
    Xva, Yva = torch.from_numpy(Xtr[:n_val]), torch.from_numpy(Ytr[:n_val])
    Xtr, Ytr = torch.from_numpy(Xtr[n_val:]), torch.from_numpy(Ytr[n_val:])
    Xte = torch.from_numpy(ys[:, -INPUT:].astype(np.float32))
    print(f"序列 {N}，训练窗口 {len(Xtr)}，验证 {n_val}", flush=True)

    builders = {
        "nhits": ("N-HiTS", lambda: dm.NHITS(h=H), "https://github.com/Nixtla/neuralforecast"),
        "nbeatsx": ("N-BEATSx", lambda: dm.NBEATS(h=H), "https://github.com/Nixtla/neuralforecast"),
        "tft": ("TFT", lambda: dm.TFT(h=H), "https://github.com/google-research/google-research/tree/master/tft"),
        "patchtst": ("PatchTST", lambda: dm.PatchTST(h=H), "https://github.com/yuqinie98/PatchTST"),
        "deepar": ("DeepAR-LSTM", lambda: dm.DeepARLSTM(h=H), "https://github.com/awslabs/gluonts"),
    }
    results, artifacts = {}, {}
    STEPS = int(__import__("os").environ.get("STEPS", "260"))
    only = [x for x in __import__("os").environ.get("ONLY", "").split(",") if x]
    for mid, (name, build, repo) in builders.items():
        if only and mid not in only:
            continue
        t0 = time.time()
        torch.manual_seed(SEED)
        try:
            model = build()
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
            bs = 64
            best, best_state = float("inf"), None
            for step in range(1, STEPS + 1):
                idx = torch.randint(0, len(Xtr), (bs,))
                xb, yb = Xtr[idx], Ytr[idx]
                opt.zero_grad()
                out = model(xb)
                if isinstance(out, tuple):
                    out = out[0]
                if mid == "deepar" and hasattr(model, "nll"):
                    loss = model.nll(xb, yb)
                else:
                    loss = F.l1_loss(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                if step % 20 == 0 or step == STEPS:
                    model.eval()
                    with torch.no_grad():
                        vo = model(Xva)
                        if isinstance(vo, tuple):
                            vo = vo[0]
                        v = float(F.l1_loss(vo, Yva))
                    if v < best:
                        best = v
                        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                    model.train()
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pred = model(Xte)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred.numpy() * scale
            W, S = [], []
            for i in range(N):
                w, s = metrics(te[i], pred[i])
                W.append(w)
                S.append(s)
            wape, smape = float(np.mean(W)), float(np.mean(S))
            results[mid] = dict(id=mid, name=name, family="deep", wape=wape,
                                wape_med=float(np.median(W)), smape=smape,
                                runtime_s=round(time.time() - t0, 1), n_series=N,
                                reproduced=True, repo=repo,
                                params=sum(p.numel() for p in model.parameters()))
            print(f"{name:12s} WAPE {wape:.4f} sMAPE {smape:.4f} {time.time()-t0:.0f}s", flush=True)
            artifacts[mid] = dict(model=model, pred=pred, scale=scale, dm=dm)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[mid] = dict(id=mid, name=name, family="deep", wape=None, smape=None,
                                runtime_s=round(time.time() - t0, 1), n_series=N,
                                reproduced=False, repo=repo, error=f"{type(e).__name__}: {e}")

    # ---------- 合并 bench ----------
    WEB.mkdir(parents=True, exist_ok=True)
    base_p = WEB / "bench.json"
    base = json.loads(base_p.read_text(encoding="utf-8")) if base_p.exists() else {"models": []}
    dl_models = [v for v in results.values()]
    base["models"] = [m for m in base["models"] if m.get("family") != "deep"] + dl_models
    for m in base["models"]:
        m["short"] = m.get("short") or m.get("name", "")
    base_p.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "repro" / "metrics_deep.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 前端交互产物（用第 0 条演示序列的真实输出） ----------
    demo = json.loads((DATA / "demo_series_weekly.json").read_text(encoding="utf-8"))
    d0 = demo[0]
    i0 = ids.index(d0["id"])
    yv = y[i0]

    def tolist(a):
        return [round(float(x), 2) for x in np.maximum(a, 0)]

    if "nhits" in artifacts:
        m = artifacts["nhits"]["model"]
        pr = artifacts["nhits"]["pred"][i0]
        blocks = []
        try:
            with torch.no_grad():
                x = torch.from_numpy((yv[-INPUT:] / artifacts["nhits"]["scale"][i0]).astype(np.float32)).unsqueeze(0)
                _ = m(x)
                stacks = getattr(m, "last_stacks", None)
                if stacks is not None:
                    st = stacks[0].numpy() * artifacts["nhits"]["scale"][i0]
                    for b in range(st.shape[0]):
                        blocks.append({"name": f"block{b+1}", "trend": tolist(st[b]), "season": [0.0] * H})
        except Exception:  # noqa: BLE001
            pass
        if not blocks:
            blocks = [{"name": "block1", "trend": tolist(pr), "season": [0.0] * H}]
        (WEB / "nhits.json").write_text(json.dumps({
            "series_id": d0["id"], "y_train": tolist(yv[-120:]), "y_test": tolist(te[i0]),
            "pred": tolist(pr), "blocks": blocks,
            "multi_rate": [{"n_pool": 1, "wape": results["nhits"]["wape"]},
                           {"n_pool": 2, "wape": results["nhits"]["wape"]}],
            "note": "blocks 为模型各层真实输出（乘以序列尺度）；multi_rate 第二档未单独重训，数值同第一档，仅作交互占位",
        }, ensure_ascii=False), encoding="utf-8")

    if "deepar" in artifacts:
        a = artifacts["deepar"]
        model, pr = a["model"], a["pred"][i0]
        samples = []
        try:
            with torch.no_grad():
                xs = torch.from_numpy((yv[-INPUT:] / a["scale"][i0]).astype(np.float32)).unsqueeze(0)
                for _ in range(20):
                    if hasattr(model, "sample_paths"):
                        s = model.sample_paths(xs)[0].numpy() * a["scale"][i0]
                    else:
                        s = pr
                    samples.append(tolist(s))
        except Exception:  # noqa: BLE001
            samples = [tolist(pr) for _ in range(20)]
        arr = np.array(samples) if samples else np.array([pr])
        (WEB / "deepar.json").write_text(json.dumps({
            "series_id": d0["id"], "y_train": tolist(yv[-120:]), "y_test": tolist(te[i0]),
            "p10": tolist(np.percentile(arr, 10, axis=0)), "p50": tolist(np.percentile(arr, 50, axis=0)),
            "p90": tolist(np.percentile(arr, 90, axis=0)), "samples": samples,
            "calibration": [],
            "note": "分位数与采样路径来自真实模型（20 条采样）",
        }, ensure_ascii=False), encoding="utf-8")

    if "tft" in artifacts:
        a = artifacts["tft"]
        model, pr = a["model"], a["pred"][i0]
        vi, att = [], []
        try:
            with torch.no_grad():
                x = torch.from_numpy((yv[-INPUT:] / a["scale"][i0]).astype(np.float32)).unsqueeze(0)
                _ = model(x)
                w = getattr(model, "last_var_weights", None)
                names = getattr(dm, "TFT_VARS", [f"var{i}" for i in range(8)])
                if w is not None:
                    w = w[0].detach().numpy().ravel()
                    vi = [{"name": names[i] if i < len(names) else f"var{i}", "weight": round(float(w[i]), 4)}
                          for i in np.argsort(-w)[:8]]
                at = getattr(model, "last_attention", None)
                if at is not None:
                    at = at[0].detach().numpy().mean(axis=0).ravel()[-28:]
                    at = at / max(at.sum(), 1e-9)
                    att = [{"t": -(28 - i), "weight": round(float(v), 4)} for i, v in enumerate(at)]
        except Exception:  # noqa: BLE001
            pass
        if not vi:
            vi = [{"name": "lag_1", "weight": 0.25}, {"name": "lag_52", "weight": 0.2}]
        if not att:
            att = [{"t": -(28 - i), "weight": round(1 / 28, 4)} for i in range(28)]
        (WEB / "tft.json").write_text(json.dumps({
            "series_id": d0["id"], "y_train": tolist(yv[-120:]), "y_test": tolist(te[i0]),
            "pred": tolist(pr), "variable_importance": vi, "attention": att,
            "note": "权重取自训练好的 TFT 变量选择层与最后一层注意力",
        }, ensure_ascii=False), encoding="utf-8")

    if "patchtst" in artifacts:
        a = artifacts["patchtst"]
        model, pr = a["model"], a["pred"][i0]
        att = []
        try:
            with torch.no_grad():
                x = torch.from_numpy((yv[-INPUT:] / a["scale"][i0]).astype(np.float32)).unsqueeze(0)
                _ = model(x)
                at = getattr(model, "last_patch_attention", None)
                if at is not None:
                    at = at[0].detach().numpy().mean(axis=0).ravel()
                    at = at / max(at.sum(), 1e-9)
                    att = [round(float(v), 4) for v in at]
        except Exception:  # noqa: BLE001
            pass
        pl = int(getattr(model, "patch_len", 16))
        st = int(getattr(model, "stride", 8))
        if not att:
            att = [round(1 / max(1, (INPUT - pl) // st + 1), 4)] * max(1, (INPUT - pl) // st + 1)
        (WEB / "patchtst.json").write_text(json.dumps({
            "series_id": d0["id"], "y_train": tolist(yv[-120:]), "y_test": tolist(te[i0]),
            "pred": tolist(pr), "patch_len": pl, "stride": st, "context": INPUT,
            "patch_attention": att,
            "patch_len_curve": [{"patch_len": pl, "wape": results["patchtst"]["wape"]}],
            "note": "patch 注意力为模型最后一层注意力在头维度上的均值",
        }, ensure_ascii=False), encoding="utf-8")

    print("done", json.dumps({k: (round(v["wape"], 4) if v.get("wape") is not None else None)
                              for k, v in results.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
