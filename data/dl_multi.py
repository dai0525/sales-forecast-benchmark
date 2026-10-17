"""多源并行分片下载：自动在多个镜像间分摊，遇 429 自动退避重试"""
import concurrent.futures as cf
import sys
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
PATH = "Mcompetitions/M4-methods/master/Dataset/Train/Daily-train.csv"
MIRRORS = [
    f"https://raw.githubusercontent.com/{PATH}",
    f"https://ghproxy.net/https://raw.githubusercontent.com/{PATH}",
    f"https://ghfast.top/https://raw.githubusercontent.com/{PATH}",
    f"https://raw.gitmirror.com/{PATH}",
]


def head_size(url):
    r = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(r, timeout=30) as resp:
        return int(resp.headers["Content-Length"])


def fetch(url, s, e, part, tries=8):
    want = e - s + 1
    delay = 2.0
    for t in range(tries):
        if part.exists() and part.stat().st_size == want:
            return True
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            if len(data) == want:
                part.write_bytes(data)
                return True
        except Exception as e2:  # noqa: BLE001
            code = getattr(e2, "code", None)
            if t == tries - 1:
                print(f"give up {url[:28]} {s}-{e} {type(e2).__name__} {code}", flush=True)
            time.sleep(delay if code == 429 else min(delay, 5))
            delay = min(delay * 1.7, 20)
    return False


def main(out, n=24):
    out = Path(out)
    size = None
    for m in MIRRORS:
        try:
            size = head_size(m)
            print(f"head ok {m[:40]} size={size}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print("head fail", m[:40], e, flush=True)
    if not size:
        return 1
    d = out.parent / (out.name + ".parts")
    d.mkdir(parents=True, exist_ok=True)
    ch = (size + n - 1) // n
    jobs = []
    for i in range(n):
        s = i * ch
        e = min((i + 1) * ch - 1, size - 1)
        jobs.append((i, s, e, MIRRORS[i % len(MIRRORS)]))
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=min(n, 32)) as ex:
        futs = {ex.submit(fetch, u, s, e, d / f"p{i}"): i for i, s, e, u in jobs}
        for f in cf.as_completed(futs):
            if f.result():
                ok += 1
            if ok % 8 == 0:
                print(f"done {ok}/{n}", flush=True)
    print(f"parts {ok}/{n}", flush=True)
    with out.open("wb") as w:
        for i in range(n):
            p = d / f"p{i}"
            if not p.exists():
                print("MISSING", i, flush=True)
                return 2
            w.write(p.read_bytes())
    got = out.stat().st_size
    print(f"final {got}/{size}", flush=True)
    if got != size:
        return 3
    for p in d.glob("p*"):
        p.unlink()
    d.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24))
