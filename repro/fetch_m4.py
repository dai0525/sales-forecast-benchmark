"""Download official M4 Daily/Weekly CSVs atomically into data/."""
import argparse
import csv
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--only', choices=['Daily','Weekly'], default=None)
    args=ap.parse_args()
    for freq in ([args.only] if args.only else ['Weekly','Daily']):
        for split in ['Train','Test']:
            dest=ROOT/'data'/f'{freq}-{split.lower()}.csv'
            if dest.exists():
                print('exists',dest); continue
            tmp=dest.with_suffix('.csv.part')
            with urllib.request.urlopen(f'{BASE}/{split}/{dest.name}',timeout=120) as r, tmp.open('wb') as f:
                while chunk:=r.read(1024*1024): f.write(chunk)
            with tmp.open() as f:
                rows=csv.reader(f); header=next(rows); first=next(rows)
                if not first[0].startswith(freq[0]): raise ValueError('Unexpected CSV IDs')
            tmp.replace(dest); print('downloaded',dest, dest.stat().st_size)
if __name__=='__main__': main()
