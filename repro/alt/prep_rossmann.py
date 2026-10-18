"""Build the Rossmann modelling panel (idempotent) used by bench_rossmann.py.

Panel definition
----------------
* unit = store, frequency = daily, target = Sales (turnover).
* only stores with a complete 2013-01-01..2015-07-31 calendar are eligible:
  180 stores are missing the whole 2014-07-01..2014-12-31 block (renovation)
  and 1 store misses a single day, so 934 stores are complete.  We do **not**
  impute; the eligible pool is simply the complete ones.
* a fixed random sample of N=400 stores (seed 42) is drawn from that pool, so
  every model in the benchmark sees exactly the same series.
* closed days are kept as Sales=0 -- zero inflation is part of the factor
  structure we want to measure, not an artefact to clean away.

Stored covariates (all "known future" at forecast time in the Rossmann
competition, which supplies them for the test period as well):
  promo, open, school_holiday, state_holiday (0/none,1/a,2/b,3/c)
and static store attributes: store_type, assortment, competition_distance,
promo2.

Output: data/alt/rossmann/panel.npz  (+ panel_meta.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "alt" / "rossmann"
OUT = SRC / "panel.npz"
META = SRC / "panel_meta.json"

N_SERIES = 400
SEED = 42
STATE_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}


def main() -> None:
    train = pd.read_csv(SRC / "train.csv", dtype={"StateHoliday": str},
                        parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(SRC / "store.csv")

    dates = pd.date_range("2013-01-01", "2015-07-31", freq="D")
    T = len(dates)
    assert T == 942, T

    # ---- eligible stores: full calendar present, no duplicate (store, date)
    have = train.groupby("Store")["Date"].nunique()
    complete = sorted(have[have == T].index.tolist())
    if len(complete) != 934:
        raise SystemExit(f"expected 934 complete stores, got {len(complete)}")
    if train.duplicated(["Store", "Date"]).any():
        raise SystemExit("duplicate (Store, Date) rows")

    # ---- fixed sample
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(np.asarray(complete), size=N_SERIES, replace=False))
    sub = train[train.Store.isin(keep)].copy()

    # ---- wide matrices, verified dense
    wide = {c: sub.pivot(index="Store", columns="Date", values=c).reindex(
        index=keep, columns=dates) for c in
        ["Sales", "Promo", "Open", "SchoolHoliday", "StateHoliday"]}
    for name, df in wide.items():
        if df.shape != (N_SERIES, T):
            raise SystemExit(f"{name}: shape {df.shape}")
        if df.isna().any().any():
            raise SystemExit(f"{name}: unexpected NaN after pivoting complete stores")

    Y = wide["Sales"].to_numpy(np.float32)
    if not np.isfinite(Y).all() or (Y < 0).any():
        raise SystemExit("Sales must be finite and non-negative")

    state = wide["StateHoliday"].astype(str).apply(
        lambda col: col.map(STATE_MAP)).to_numpy(np.int8)

    meta = store.set_index("Store").reindex(keep)
    type_map = {c: i for i, c in enumerate(sorted(store.StoreType.unique()))}
    asst_map = {c: i for i, c in enumerate(sorted(store.Assortment.unique()))}
    comp = meta.CompetitionDistance.to_numpy(np.float64)
    comp = np.where(np.isfinite(comp), comp, np.nan)

    np.savez_compressed(
        OUT,
        y=Y,
        ids=np.array([f"S{s}" for s in keep]),
        stores=keep.astype(np.int32),
        dates=dates.to_numpy("datetime64[D]").astype("int64"),
        promo=wide["Promo"].to_numpy(np.int8),
        open_=wide["Open"].to_numpy(np.int8),
        school=wide["SchoolHoliday"].to_numpy(np.int8),
        state=state,
        store_type=meta.StoreType.map(type_map).to_numpy(np.int8),
        assortment=meta.Assortment.map(asst_map).to_numpy(np.int8),
        promo2=meta.Promo2.to_numpy(np.int8),
        comp_dist=comp.astype(np.float32),
    )

    info = {
        "dataset": "Rossmann Store Sales",
        "unit": "store", "freq": "D", "target": "Sales",
        "n_complete_stores": len(complete),
        "n_series_sampled": int(N_SERIES), "sample_seed": SEED,
        "sample_rule": ("from the 934 stores with a complete 942-day calendar, draw "
                        "400 without replacement using numpy default_rng(42), then sort"),
        "n_days": int(T), "date_min": str(dates[0].date()), "date_max": str(dates[-1].date()),
        "n_rows": int(N_SERIES * T),
        "zero_sales_share": float((Y == 0).mean()),
        "closed_share": float((wide["Open"].to_numpy() == 0).mean()),
        "promo_share": float((wide["Promo"].to_numpy() == 1).mean()),
        "state_holiday_share": float((state > 0).mean()),
        "school_holiday_share": float((wide["SchoolHoliday"].to_numpy() == 1).mean()),
        "mean_sales": float(Y.mean()), "median_sales": float(np.median(Y)),
        "series_mean_sales_min": float(Y.mean(1).min()), "series_mean_sales_max": float(Y.mean(1).max()),
        "store_type_levels": type_map, "assortment_levels": asst_map,
        "comp_dist_missing": int(np.isnan(comp).sum()),
    }
    META.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2)[:1200])
    print(f"[ok] {OUT}  {Y.shape}")


if __name__ == "__main__":
    main()
