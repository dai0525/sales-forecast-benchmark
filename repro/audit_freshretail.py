"""Download pinned official FreshRetailNet files and audit structure, not performance.

Run: python repro/audit_freshretail.py
Requires pandas and pyarrow. Raw files stay under ignored data/raw/.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c374e85975c82816c495acfa904cc55ab14a47d3"
HASHES = {
    "train": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    "eval": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    raw = ROOT / "data/raw/freshretail"
    raw.mkdir(parents=True, exist_ok=True)
    report = {"revision": REVISION, "purpose": "Structural audit only; no model selection or test scoring", "splits": {}}
    keys = {}
    for split, expected in HASHES.items():
        url = f"https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/{REVISION}/data/{split}.parquet"
        path = raw / f"{split}.parquet"
        if not path.exists():
            temporary = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as f:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    f.write(block)
            if digest(temporary) != expected:
                raise ValueError(f"Checksum mismatch: {split}")
            temporary.replace(path)
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {split}")
        schema = pq.read_schema(path)
        columns = [name for name in schema.names if name not in ("hours_sale", "hours_stock_status")]
        df = pd.read_parquet(path, columns=columns)
        group = df.groupby(["store_id", "product_id"]).size()
        keys[split] = pd.MultiIndex.from_frame(df[["store_id", "product_id", "dt"]])
        report["splits"][split] = {
            "url": url, "bytes": path.stat().st_size, "sha256": expected,
            "rows": len(df), "schema": {f.name: str(f.type) for f in schema},
            "date_min": df.dt.min(), "date_max": df.dt.max(),
            "days": int(df.dt.nunique()), "series": len(group),
            "rows_per_series_min": int(group.min()), "rows_per_series_max": int(group.max()),
            "stores": int(df.store_id.nunique()), "products": int(df.product_id.nunique()),
            "cities": int(df.city_id.nunique()),
            "duplicate_series_dates": int(keys[split].duplicated().sum()),
            "scalar_null_counts": {k: int(v) for k, v in df.isna().sum().items()},
        }
    report["overlapping_series_dates"] = len(keys["train"].intersection(keys["eval"]))
    out = ROOT / "audit/results/dataset_expansion/freshretail_structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(out)
    print(json.dumps({k: {f: v for f, v in s.items() if f not in ("schema", "scalar_null_counts", "url")}
                      for k, s in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
