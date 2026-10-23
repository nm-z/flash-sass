#!/usr/bin/env python3
import csv,gzip,hashlib,json,shutil
from pathlib import Path
base=Path("/home/nate/claude/flash-sass-prs")
src=base/"paired-depth-trace3/runs"
out=base/"decode-review-evidence"
out.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
files=sorted(src.glob("*.csv.gz"))
entries=[dict(path=str(p.relative_to(src)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
save(out/"raw-trace-archive.json",dict(root=str(src),files=entries,total_bytes=sum(x["bytes"] for x in entries),note="All original per-phase and host CSV gzip streams; these hashes identify the full collection retained on archy."))
ex=out/"raw-excerpts";ex.mkdir(exist_ok=True)
for n in (3100,8000):
    for placement in ("host","vram"):
        for die in (0,4):
            p=src/f"depth-{n}-pair-2-{placement}-die-{die}.csv.gz"
            target=ex/p.name
            with gzip.open(p,"rt",newline="") as f,gzip.open(target,"wt",newline="") as g:
                r=csv.DictReader(f);w=csv.DictWriter(g,fieldnames=r.fieldnames);w.writeheader()
                for row in r:
                    if int(row["position"]) in (n,n+1):w.writerow(row)
        p=src/f"depth-{n}-pair-2-{placement}-host.csv.gz"
        shutil.copyfile(p,ex/p.name)
save(ex/"README.json",dict(positions="N and N+1 are two adjacent fully decoded positions after the prompt-boundary output at N-1.",selection="Pair 2 is illustrative. These excerpts do not replace all-run aggregates or the full archive.",dies=[0,4],prompt_depths=[3100,8000],placements=["host","vram"],host_traces="Complete 129-output host trace for each illustrated run."))
setup=out/"setup";setup.mkdir(exist_ok=True)
for job,folder in ((3,"paired-depth-trace3"),(4,"paired-depth-untraced4")):
    shutil.copyfile(base/f"bench_pair_job{job}.py",setup/f"bench_pair_job{job}.py")
    for name in ("job-exit.json","build.json","command.json","launch-environment.json"):
        p=base/folder/name
        if p.exists():shutil.copyfile(p,setup/f"job{job}-{name}")
save(out/"STATUS.json",dict(status="PASS",raw_files=len(entries),raw_bytes=sum(x["bytes"] for x in entries)))
print(json.dumps(dict(status="PASS",files=len(entries),bytes=sum(x["bytes"] for x in entries))))
