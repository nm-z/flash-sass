#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess,time,socket,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--revision",required=True);p.add_argument("--out",required=True);p.add_argument("targets",nargs="+");a=p.parse_args()
src=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
if not (src/"maxas").exists():(src/"maxas").symlink_to("/home/nate/claude/flash-sass/maxas",target_is_directory=True)
env=os.environ.copy()
for k in ("DIE_TRACE","DIE_TRACE16","FLASH_TRACE"):env.pop(k,None)
cmd=["make","CUDA=/opt/cuda","PTXAS11=/opt/cuda-11.4/bin/ptxas","FLASH_REV="+a.revision]+a.targets
t=time.monotonic()
with (out/"build.log").open("w") as f:
 try:rc=subprocess.run(cmd,cwd=src/"kern",env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
 except subprocess.TimeoutExpired:rc=124
def ident(path):
 b=path.read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
row=dict(source_revision=a.revision,hostname=socket.gethostname(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,exit=rc,seconds=time.monotonic()-t,artifacts={x:ident(src/"kern"/x) for x in a.targets if (src/"kern"/x).is_file()},maxas={x:ident(src/"maxas"/"lib"/"MaxAs"/x) for x in ("Cubin.pm","MaxAs.pm","MaxAsGrammar.pm")})
(out/"build.json").write_text(json.dumps(row,indent=2)+"\n")
print(json.dumps(row),flush=True);raise SystemExit(rc)
