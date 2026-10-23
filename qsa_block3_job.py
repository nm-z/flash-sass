#!/usr/bin/env python3
import os,json,subprocess,time,hashlib,fcntl
from pathlib import Path
root=Path("/home/nate/claude/flash-sass-prs")
src=root/"qsa-build-1e299ef"/"kern"
out=root/"qsa-block3-validation-1e299ef"
lock=open("/home/nate/codex/asm-port/GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4.lock","a")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ids=out/"ids9.txt"
ids.write_text("10 11 12 13 14 15 16 17 18\n")
env=os.environ.copy()
env.update(MODEL_ATTN="qsa",FLASH_NBLK="1",MODEL_IDS=str(ids),CUDA_VISIBLE_DEVICES="0,1,2,3,4,5",CUDA_DEVICE_ORDER="PCI_BUS_ID",SPLIT_SHARES="112,97,81,116,97,97",ATTN_DIE="1",ATTN_MAP="1,1,1,1,4,4,4,4,5,5,5,5")
for k in ("MODEL_NOTWIN","MODEL_INJECT","MODEL_CAPTURE","MODEL_LOOP","MODEL_LAYERS","MODEL_PLE"):env.pop(k,None)
cmd=[str(src/"model_test"),str(out/"cache"/"tensors"),"3","4","9","-"]+["/tmp/hold-h%d.manifest"%i for i in range(6)]
def record(name,args,e):
    start=time.time()
    with open(out/(name+".stdout"),"w") as stdout,open(out/(name+".stderr"),"w") as stderr:
        try:code=subprocess.run(args,cwd=src,env=e,stdout=stdout,stderr=stderr,timeout=180).returncode
        except subprocess.TimeoutExpired:code=124
    row=dict(name=name,argv=args,exit=code,seconds=time.time()-start,environment={k:v for k,v in e.items() if k.startswith(("MODEL_","FLASH_","CUDA_")) or k in ("SPLIT_SHARES","ATTN_DIE","ATTN_MAP")})
    print(json.dumps(row),flush=True);return row
rows=[]
rows.append(record("gpu-before",["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm","--format=csv"],env))
for placement in ("0","1"):
    for injection in (None,"selection","layer"):
        e=env.copy();e["FLASH_KV_HOST"]=placement
        if injection:e["MODEL_INJECT"]=injection
        name="kv"+placement+"-"+(injection or "positive")
        row=record(name,cmd,e);row["expected_exit"]=3 if injection else 0
        row["passed"]=row["exit"]==row["expected_exit"];rows.append(row)
        (out/"gpu-results.json").write_text(json.dumps(rows,indent=2)+"\n")
        if row["exit"] not in (0,3):raise SystemExit("Unexpected failure: "+name)
rows.append(record("gpu-after",["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm","--format=csv"],env))
result=dict(source_revision="1e299ef5e169ed9a88f2f6c8d0241831ab688f72",scope="actual-weight block 3 integration; synthetic initial stream; nine teacher-forced IDs; small budget forces exclusions; no embedding/head or full-model quality claim",rows=rows,passed=all(r.get("passed",True) for r in rows))
(out/"gpu-results.json").write_text(json.dumps(result,indent=2)+"\n")
raise SystemExit(0 if result["passed"] else 3)
