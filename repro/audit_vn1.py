"""Fetch the organizer's VN1 bundle, pin its checksum and audit CSV structure.

Requires pandas. Does not execute downloaded code or score any model.
"""
import json
import hashlib
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

ROOT = Path(__file__).resolve().parents[1]
URL = "https://supchains.com/downloads/vn1-forecasting-competition.zip"
SHA256 = "0b5fe01aca827fb223533afe50e234e568ab304164c22e5d93037a69289e23a9"


def main():
    path = ROOT / "data/raw/vn1/vn1.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as f:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                f.write(block)
        if digest(temporary) != SHA256:
            raise ValueError("VN1 archive changed; review the source before updating the checksum")
        temporary.replace(path)
    if digest(path) != SHA256:
        raise ValueError("VN1 checksum mismatch")
    report = {"source": URL, "sha256": SHA256, "bytes": path.stat().st_size, "files": {}}
    reference = None
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if "/Phase " not in name or not name.endswith(".csv"):
                continue
            with archive.open(name) as f:
                df = pd.read_csv(f)
            identifiers = df[["Client", "Warehouse", "Product"]]
            if reference is None:
                reference = identifiers
            weeks = list(df.columns[3:])
            report["files"][name] = {
                "rows": len(df), "weeks": len(weeks), "first_week": weeks[0], "last_week": weeks[-1],
                "clients": int(df.Client.nunique()), "warehouses": int(df.Warehouse.nunique()),
                "products": int(df.Product.nunique()), "duplicate_ids": int(identifiers.duplicated().sum()),
                "same_ids_and_order_as_first_file": identifiers.equals(reference),
                "missing_value_cells": int(df.iloc[:, 3:].isna().sum().sum()),
            }
    dest = ROOT / "audit/results/dataset_expansion/vn1_structure.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
