"""并行分片下载 + 逐片校验 + 失败重试（本机单连接被限速到 ~18KB/s，必须多连接）"""
import concurrent.futures as cf
import os
import sys
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}


def head_size(url):
    r = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(r, timeout=30) as resp:
        n = resp.headers.get("Content-Length")
        return int(n)


def fetch_range(url, start, end, part, tries=6):
    want = end - start + 1
    for t in range(tries):
        try:
            if part.exists() and part.stat().st_size == want:
                return part
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) == want:
                part.write_bytes(data)
                return part
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                print(f"FAIL {start}-{end}: {type(e).__name__} {e}", flush=True)
        if part.exists():
            part.unlink()
    return None


def main(url, out, n=32):
    out = Path(out)
    size = head_size(url)
    print(f"size={size} parts={n}", flush=True)
    d = out.parent / (out.name + ".parts")
    d.mkdir(parents=True, exist_ok=True)
    ch = (size + n - 1) // n
    jobs = [(i, i * ch, min((i + 1) * ch - 1, size - 1)) for i in range(n)]
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(fetch_range, url, s, e, d / f"p{i}"): (i, s, e) for i, s, e in jobs}
        for f in cf.as_completed(futs):
            if f.result() is not None:
                ok += 1
            else:
                print("missing", futs[f], flush=True)
    print(f"parts ok {ok}/{n}", flush=True)
    with out.open("wb") as w:
        for i in range(n):
            p = d / f"p{i}"
            if not p.exists():
                print("ABORT: missing part", i, flush=True)
                return 2
            w.write(p.read_bytes())
    got = out.stat().st_size
    print(f"final {got} / {size}", flush=True)
    if got != size:
        return 3
    for p in d.glob("p*"):
        p.unlink()
    d.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32))
