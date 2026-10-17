#!/usr/bin/env python
"""Resumable parallel HTTP range downloader for the M4 CSVs.

The link to raw.githubusercontent.com is throttled to a few KB/s per
connection from this host, so we fetch fixed-size byte ranges concurrently
and stitch them back together.  Chunks that already exist on disk with the
expected size are skipped, so re-running resumes instead of restarting.

Usage: python repro/fetch_m4.py [--workers 12] [--chunk-mb 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from pathlib import Path

RAW = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
# ghproxy mirrors raw.githubusercontent and is significantly faster here; the
# raw host is kept as a fallback for each chunk.
MIRRORS = [
    "https://ghfast.top/" + RAW,
    "https://ghproxy.net/" + RAW,
    "https://gh-proxy.com/" + RAW,
    RAW,
]

FILES = [
    ("Train/Monthly-train.csv", "Monthly-train.csv"),
    ("Train/Daily-train.csv", "Daily-train.csv"),
    ("Test/Monthly-test.csv", "Monthly-test.csv"),
    ("Test/Daily-test.csv", "Daily-test.csv"),
]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw"


def head_size(url: str) -> int | None:
    for attempt in range(6):
        base = MIRRORS[attempt % len(MIRRORS)]
        try:
            r = subprocess.run(
                ["curl", "-sIL", "--max-time", "30", f"{base}/{url}"],
                capture_output=True, text=True, timeout=45,
            )
        except subprocess.TimeoutExpired:
            continue
        sizes = [ln.split(":", 1)[1].strip() for ln in r.stdout.splitlines()
                 if ln.lower().startswith("content-length")]
        txt = r.stdout
        if "429" in txt[:20]:
            time.sleep(5 * (attempt + 1))
            continue
        if sizes:
            try:
                size = int(sizes[-1])
            except ValueError:
                continue
            if size > 1000:
                return size
        time.sleep(3)
    return None


def fetch_chunk(rel: str, idx: int, start: int, end: int, dest: Path,
                timeout: int = 150) -> tuple[int, int]:
    if dest.exists() and dest.stat().st_size == end - start + 1:
        return idx, dest.stat().st_size
    last = 0
    for attempt in range(5):
        base = MIRRORS[attempt % len(MIRRORS)]
        tmp = dest.with_suffix(dest.suffix + ".part")
        cmd = ["curl", "-sL", "--max-time", str(timeout), "-r", f"{start}-{end}",
               "-o", str(tmp), f"{base}/{rel}"]
        try:
            subprocess.run(cmd, timeout=timeout + 25)
        except subprocess.TimeoutExpired:
            pass
        if tmp.exists():
            got = tmp.stat().st_size
            ok = got == end - start + 1
            if ok:
                with open(tmp, "rb") as fh:
                    head = fh.read(4096)
                # a range fetched as compressed/other payload can still have the
                # right length; require plain ASCII CSV before accepting it
                if any(b > 127 for b in head) or not any(c in head for c in b'"0123456789'):
                    ok = False
            if ok:
                tmp.replace(dest)
                return idx, got
            last = got
            # a short body is almost always a rate-limit / error page
            time.sleep(2 + attempt * 3)
    if dest.exists():
        return idx, dest.stat().st_size
    raise RuntimeError(f"chunk {idx} of {rel} failed (last size {last})")


def download(rel: str, name: str, workers: int, chunk_mb: int,
             deadline: float | None = None) -> Path | None:
    final = OUT / name
    if final.exists() and final.stat().st_size > 0:
        print(f"[skip] {name} already present ({final.stat().st_size} bytes)", flush=True)
        return final
    size = head_size(rel)
    if size is None:
        raise SystemExit(f"cannot determine size of {rel}")
    csz = chunk_mb * 1024 * 1024
    nchunks = (size + csz - 1) // csz
    cdir = OUT / f".{name}.chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    have = sum(1 for i in range(nchunks)
               if (cdir / f"chunk{i:05d}").exists())
    print(f"[get ] {name}: {size/1e6:.1f} MB, {nchunks} chunks, {have} already done, "
          f"{workers} workers", flush=True)

    t0 = time.time()
    done = 0
    todo = [i for i in range(nchunks) if not (cdir / f"chunk{i:05d}").exists()]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for i in todo:
            start = i * csz
            end = min(start + csz, size) - 1
            futs[ex.submit(fetch_chunk, rel, i, start, end,
                           cdir / f"chunk{i:05d}")] = i
        for fut in cf.as_completed(futs):
            idx, got = fut.result()
            done += 1
            if done % 5 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"       {name}: +{done}/{len(todo)} chunks in {el:.0f}s", flush=True)
            if deadline and time.time() > deadline:
                for f in futs:
                    f.cancel()
                print(f"[time] {name}: budget hit, {have+done}/{nchunks} chunks saved; "
                      f"rerun to resume", flush=True)
                return None

    with open(final, "wb") as fh:
        for i in range(nchunks):
            with open(cdir / f"chunk{i:05d}", "rb") as ch:
                while True:
                    buf = ch.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
    for p in cdir.iterdir():
        p.unlink()
    cdir.rmdir()
    with open(final, "rb") as fh:
        head = fh.read(200).decode("utf-8", "replace")
    if "V1" not in head.split("\n")[0]:
        bad = final.with_suffix(".bad")
        final.replace(bad)
        raise RuntimeError(f"{name} assembled but header looks wrong: {head[:80]!r} "
                           f"(kept as {bad.name})")
    print(f"[ok  ] {name}: {final.stat().st_size} bytes in {time.time()-t0:.0f}s", flush=True)
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--time-budget", type=float, default=0,
                    help="seconds; 0 = no limit")
    ap.add_argument("--only", default=None, help="substring of the file name")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.time_budget if args.time_budget else None
    files = [f for f in FILES if not args.only or args.only.lower() in f[1].lower()]
    for rel, name in files:
        if deadline and time.time() > deadline:
            print(f"[time] stopping before {name}", flush=True)
            break
        try:
            download(rel, name, min(args.workers, 16), args.chunk_mb, deadline)
        except Exception as exc:  # keep going: one bad file must not kill the rest
            print(f"[FAIL] {name}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
