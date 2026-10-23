#!/usr/bin/env python3
import hashlib,json,os,subprocess,time
from pathlib import Path
base=Path("/home/nate/claude/flash-sass-prs")
src=base/"qsa-build-1e299ef"
out=base/"qsa-block3-validation-1e299ef";out.mkdir(exist_ok=False)
records=[]
def save():
    (out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
def run(label,args,timeout=600,env=None):
    start=time.monotonic()
    with (out/(label+".log")).open("w") as f:
        try:r=subprocess.run(args,cwd=src/"kern",stdout=f,stderr=subprocess.STDOUT,timeout=timeout,env=env);code=r.returncode
        except subprocess.TimeoutExpired:code=124
    records.append(dict(label=label,argv=args,exit=code,seconds=time.monotonic()-start));save()
    print(json.dumps(records[-1]),flush=True)
    if code:raise RuntimeError(label+" failed")
try:
    run("build",["make","CUDA=/opt/cuda","PTXAS11=/opt/cuda-11.4/bin/ptxas","FLASH_REV=1e299ef5e169ed9a88f2f6c8d0241831ab688f72","die.cubin","model_test","hold_cache"],1200)
    identities={}
    for name in ("die.cubin","model_test","hold_cache"):
        p=src/"kern"/name;identities[name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (out/"build.json").write_text(json.dumps(dict(source_revision="1e299ef5e169ed9a88f2f6c8d0241831ab688f72",files=identities),indent=2)+"\n")
    symbols=subprocess.run(["nm","-u",str(src/"kern/hold_cache")],capture_output=True,text=True,check=True).stdout
    (out/"hold-cache-symbols.txt").write_text(symbols)
    assert not any(s in symbols for s in ("cuInit","cuMem","cuIpc","cuCtx"))
    cache=out/"cache";(cache/"tensors").mkdir(parents=True)
    run("cache-export",[str(src/"kern/hold_cache"),"/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf","0","3","4",str(cache/"manifest"),"cache="+str(cache/"tensors")],1800)
    (out/"ready.json").write_text(json.dumps(dict(status="READY_FOR_GPU",source_revision="1e299ef5e169ed9a88f2f6c8d0241831ab688f72",cache=str(cache/"tensors")))+"\n")
    print("READY_FOR_GPU",flush=True)
except Exception as e:
    (out/"FAILED.json").write_text(json.dumps(dict(error=repr(e)))+"\n")
    raise
