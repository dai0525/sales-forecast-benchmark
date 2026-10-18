"""Package the complete working repository, data and research evidence.
Exclude VCS internals, installed environments and disposable Python caches.
"""
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','venv','__pycache__','.pytest_cache','.cache','node_modules'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT.parent/'sales-forecast-benchmark-complete.zip');args=ap.parse_args()
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts) and p.suffix not in {'.pyc','.part'} and p.name!='DELIVERY-MANIFEST.json' and p.resolve()!=args.output.resolve()]
    paths.sort()
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'original_commit':'83f3947','note':'Working-tree source, reports, input data and experiment artifacts; excludes installed environments and VCS internals. Includes downloaded Daily training CSV if present. No remote push.', 'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    out=ROOT/'DELIVERY-MANIFEST.json';out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));paths.append(out)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    with zipfile.ZipFile(args.output) as z:
        bad=z.testzip()
        if bad:raise ValueError('Corrupt archived file: '+bad)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.zip.sha256').write_text(digest+'  '+args.output.name+'\n')
    print(json.dumps({'archive':str(args.output),'bytes':args.output.stat().st_size,'files':len(paths),'sha256':digest},indent=2))
if __name__=='__main__':main()
