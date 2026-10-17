"""Shared data loading, splitting and metric code for the M4 daily benchmark.

Layout of the official M4 CSVs (this bit is easy to get wrong): they are *wide*
tables -- one row per series, the first cell is the series id ("D1", "D2", ...)
and the remaining cells are the observations.  The header ("V1"..."V9920") only
names the time columns, it does **not** list the series ids.  Shorter series are
not padded, so rows are ragged and must be parsed per row.

Consequently `train` is returned as a right-aligned (N, T_max) float64 matrix:
every series ends in the same column, NaNs pad the front.  Right alignment is
what makes lag/rolling features a pure column-shift, and it is also why the
synthetic calendar below is phase-consistent across series.

M4 ships no timestamps, so `ds` is derived by anchoring the last training day of
every series at ANCHOR_DATE.  This only affects the calendar features
(dayofweek/month/day/weekofyear); lags and rolling windows are untouched by it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

H = 14                    # M4 daily horizon (also the validation window)
SEASON = 7                # daily seasonality
ANCHOR = np.datetime64("2016-12-31")   # nominal last training day
LAGS = (1, 7, 14, 28)
ROLLS = (7, 14, 28)
LAG_MAX = max(max(LAGS), max(ROLLS))   # 28


# ---------------------------------------------------------------- loading
def _parse_wide_csv(path: Path) -> tuple[list[str], list[np.ndarray]]:
    """M4 wide CSV -> (ids, ragged value arrays).  Tolerates "" padding and a
    truncated last line (a partial download)."""
    ids: list[str] = []
    vals: list[np.ndarray] = []
    bad = 0
    with open(path, encoding="utf-8") as fh:
        fh.readline()                       # header = time-column names, unused
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            sid = parts[0].strip('"')
            body = ",".join(parts[1:]).strip().strip(",")
            if body:
                body = body.replace('"', "")
                try:
                    arr = np.fromstring(body, sep=",", dtype=np.float64)
                except ValueError:
                    # truncated tail or a stray token: keep the numeric prefix
                    good = []
                    for tok in body.split(","):
                        tok = tok.strip().strip('"')
                        try:
                            good.append(float(tok))
                        except ValueError:
                            break
                    arr = np.asarray(good, dtype=np.float64)
                    bad += 1
            else:
                arr = np.empty(0)
            ids.append(sid)
            vals.append(arr)
    if bad:
        print(f"[warn] {path.name}: {bad} line(s) had non-numeric tails "
              f"(partial download?)", flush=True)
    return ids, vals


def _stack_right(ragged: list[np.ndarray], tmax: int | None = None) -> np.ndarray:
    t = tmax or max(len(a) for a in ragged)
    out = np.full((len(ragged), t), np.nan, dtype=np.float32)
    for i, a in enumerate(ragged):
        k = min(len(a), t)
        if k:
            out[i, t - k:] = a[-k:]
    return out


def load_m4_daily(n_series: int | None = None, seed: int = 42) -> "DailyData":
    """Load train/test wide CSVs; fall back to the npz produced by data/prep_data.py."""
    csv_train = DATA / "Daily-train.csv"
    csv_test = DATA / "Daily-test.csv"
    if csv_train.exists():
        ids, ragged = _parse_wide_csv(csv_train)
        tr = _stack_right(ragged)
        tids, tragged = _parse_wide_csv(csv_test)
        te = np.vstack([np.pad(a, (0, H - len(a)), constant_values=np.nan)[:H]
                        for a in tragged]) if tragged else np.empty((0, H))
        if tids != ids:
            raise SystemExit("train/test series ids differ in order")
        test_ids = tids
    else:
        z = np.load(DATA / "m4daily.npz", allow_pickle=False)
        ids = [str(x) for x in z["ids"]]
        tr = z["train"].astype(np.float32)
        te = z["test"].astype(np.float32)
        if tr.shape[0] != len(ids):        # npz stores (T, N)
            tr = np.ascontiguousarray(tr.T)
        if te.shape[0] != len(ids):
            te = np.ascontiguousarray(te.T)
        test_ids = ids
    if n_series is not None and n_series < len(ids):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(ids), size=n_series, replace=False))
        ids = [ids[i] for i in keep]
        tr, te = tr[keep], te[keep]
    return DailyData(ids=ids, train=tr, test=te)


@dataclass
class DailyData:
    ids: list[str]
    train: np.ndarray      # (N, T) right-aligned, NaN padded
    test: np.ndarray       # (N, H)

    @property
    def n(self) -> int:
        return len(self.ids)

    def lengths(self) -> np.ndarray:
        return np.sum(~np.isnan(self.train), axis=1).astype(int)

    def series(self, i: int) -> np.ndarray:
        L = self.lengths()[i]
        return self.train[i, -L:].astype(np.float64)

    def val_split(self) -> tuple[np.ndarray, np.ndarray]:
        """(history for validation fit, last H train points as validation truth)."""
        return self.train[:, :-H], self.train[:, -H:]


# ---------------------------------------------------------------- metrics
def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    d = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / d) if d > 0 else float("nan")


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den == 0, 0.0, 200.0 * np.abs(y - yhat) / den)
    return float(np.mean(r))


def mase_scale(hist: np.ndarray, m: int = SEASON) -> float:
    """M4 in-sample seasonal-naive scale: mean |y_t - y_{t-m}|."""
    hist = np.asarray(hist, float)
    hist = hist[~np.isnan(hist)]
    if len(hist) <= m:
        return float("nan")
    d = np.mean(np.abs(hist[m:] - hist[:-m]))
    if d == 0 or not np.isfinite(d):
        d = np.mean(np.abs(np.diff(hist))) if len(hist) > 1 else np.nan
    return float(d)


def evaluate(preds: np.ndarray, truth: np.ndarray, train: np.ndarray,
             ids: list[str]) -> dict:
    """Per-series WAPE/sMAPE/MASE + aggregates.  preds/truth are (N, H)."""
    rows = []
    for i in range(preds.shape[0]):
        y = truth[i].astype(float)
        p = np.clip(preds[i].astype(float), 1e-9, None)
        L = int(np.sum(~np.isnan(train[i])))
        hist = train[i, -L:] if L else np.array([])
        scale = mase_scale(hist, SEASON)
        mae = float(np.mean(np.abs(y - p)))
        rows.append({
            "id": ids[i],
            "wape": wape(y, p),
            "smape": smape(y, p),
            "mase": mae / scale if scale and np.isfinite(scale) else float("nan"),
            "mae": mae,
        })
    w = np.array([r["wape"] for r in rows])
    s = np.array([r["smape"] for r in rows])
    m = np.array([r["mase"] for r in rows])
    ok = np.isfinite(m)
    agg = {
        "wape_mean": float(np.nanmean(w)),
        "wape_median": float(np.nanmedian(w)),
        "smape_mean": float(np.nanmean(s)),
        "mase_mean": float(np.nanmean(m)),
        "mase_median": float(np.nanmedian(m)) if ok.any() else float("nan"),
        "n_series": int(len(rows)),
    }
    return {"agg": agg, "per_series": rows}


def summarise(name: str, res: dict, runtime_s: float, notes: str = "") -> dict:
    return {"model": name, **res["agg"], "runtime_s": round(float(runtime_s), 2),
            "notes": notes}


# ---------------------------------------------------------------- calendar
def calendar_matrix(n_cols: int, anchor_col: int) -> dict[str, np.ndarray]:
    """Calendar features for a right-aligned axis whose last column == ANCHOR."""
    offsets = np.arange(n_cols) - anchor_col          # days relative to anchor
    dates = ANCHOR + offsets.astype("timedelta64[D]")
    d = dates.astype("datetime64[D]")
    doy = (d - d.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    return {
        "dow": ((d.astype(int) + 3) % 7).astype(np.int8),   # 1970-01-01 = Thursday
        "day": ((d - d.astype("datetime64[M]")).astype(int) + 1).astype(np.int8),
        "month": (d.astype("datetime64[M]").astype(int) % 12 + 1).astype(np.int8),
        "weekofyear": ((doy - 1) // 7 + 1).astype(np.int8),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
