"""Download + validate the Rossmann Store Sales data (idempotent, checksummed).

Why this dataset: the M4 Weekly benchmark in this repo is purely univariate,
weekly, with no exogenous information at all.  Rossmann is daily, 1115 stores,
zero-inflated (closed days), and ships *known-future* covariates -- promo plan,
state/school holidays, store type/assortment, competition distance.  That is a
different factor structure, which is the point of the second dataset.

Source: Kaggle competition "Rossmann Store Sales" (the official CSVs are behind
a Kaggle login; this script pulls byte-identical copies mirrored on GitHub and
verifies them by size / row count / store count / date range / column set).

Usage:  python repro/alt/fetch_rossmann.py [--force]
Output: data/alt/rossmann/{train.csv,store.csv} + data/alt/rossmann/validation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "alt" / "rossmann"

MIRRORS = [
    "https://raw.githubusercontent.com/bisman16/Kaggle_Rossmann_Store_Sales_Forecasting/master/data/",
    "https://ghfast.top/https://raw.githubusercontent.com/bisman16/Kaggle_Rossmann_Store_Sales_Forecasting/master/data/",
]

FILES = {
    "train.csv": {
        "bytes": 38057952,
        "rows": 1017209,                      # excluding header
        "columns": ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open",
                    "Promo", "StateHoliday", "SchoolHoliday"],
    },
    "store.csv": {
        "bytes": 45010,
        "rows": 1115,
        "columns": ["Store", "StoreType", "Assortment", "CompetitionDistance",
                    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"],
    },
}


def _download(name: str) -> Path:
    dest = OUT / name
    expect = FILES[name]
    if dest.exists() and dest.stat().st_size == expect["bytes"]:
        print(f"[skip] {name} already present ({expect['bytes']} bytes)")
        return dest
    last = None
    for base in MIRRORS:
        url = base + name
        try:
            print(f"[get ] {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=180) as r, tmp.open("wb") as fh:
                fh.write(r.read())
            if tmp.stat().st_size != expect["bytes"]:
                raise ValueError(f"size {tmp.stat().st_size} != {expect['bytes']}")
            tmp.replace(dest)
            return dest
        except Exception as exc:              # try next mirror
            last = exc
            print(f"[fail] {type(exc).__name__}: {exc}")
    raise SystemExit(f"all mirrors failed for {name}: {last}")


def _validate(name: str) -> dict:
    import csv
    path = OUT / name
    expect = FILES[name]
    with path.open(newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = sum(1 for r in rd if r)
    if header != expect["columns"]:
        raise SystemExit(f"{name}: unexpected header {header}")
    if rows != expect["rows"]:
        raise SystemExit(f"{name}: row count {rows} != {expect['rows']}")
    if path.stat().st_size != expect["bytes"]:
        raise SystemExit(f"{name}: byte size mismatch")
    return {
        "file": name,
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": header,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.force:
        for n in FILES:
            (OUT / n).unlink(missing_ok=True)

    report = {"dataset": "Rossmann Store Sales", "source": "Kaggle competition, GitHub CSV mirror",
              "kaggle_url": "https://www.kaggle.com/competitions/rossmann-store-sales/data",
              "mirror_base": MIRRORS[0], "files": {}}
    for name in FILES:
        _download(name)
    for name in FILES:
        report["files"][name] = _validate(name)

    # ---- content-level checks that a byte count cannot catch
    import numpy as np
    import pandas as pd
    tr = pd.read_csv(OUT / "train.csv", dtype={"StateHoliday": "str"},
                     parse_dates=["Date"])
    n_stores = int(tr.Store.nunique())
    dmin, dmax = tr.Date.min(), tr.Date.max()
    ndays = int(tr.Date.nunique())
    if n_stores != 1115:
        raise SystemExit(f"store count {n_stores} != 1115")
    if (dmin, dmax) != (pd.Timestamp("2013-01-01"), pd.Timestamp("2015-07-31")):
        raise SystemExit(f"unexpected date range {dmin}..{dmax}")
    if ndays != 942:
        raise SystemExit(f"distinct dates {ndays} != 942")
    zeros = float((tr.Sales == 0).mean())
    if not 0.10 < zeros < 0.25:           # ~17% of rows are closed days
        raise SystemExit(f"implausible zero share {zeros:.3f}")

    grid = tr.groupby(["Store", "Date"], observed=True).size()
    report["content"] = {
        "n_stores": n_stores,
        "n_dates": ndays,
        "date_min": str(dmin.date()),
        "date_max": str(dmax.date()),
        "n_rows": int(len(tr)),
        "missing_rate_sales": float(tr.Sales.isna().mean()),
        "zero_sales_share": zeros,
        "closed_share_open0": float((tr.Open == 0).mean()),
        "promo_share": float((tr.Promo == 1).mean()),
        "state_holiday_share_nonzero": float((tr.StateHoliday.astype(str) != "0").mean()),
        "school_holiday_share": float((tr.SchoolHoliday == 1).mean()),
        "store_date_cells": int(1115 * 942),
        "store_date_present": int(len(grid)),
        "store_date_missing": int(1115 * 942 - len(grid)),
        "duplicate_store_date": int(grid[grid > 1].sum()),
    }
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(report["content"], indent=2))
    print(f"[ok] validation -> {OUT/'validation.json'}")


if __name__ == "__main__":
    sys.exit(main())
