#!/usr/bin/env python3
import os,json,subprocess,time,hashlib,fcntl,array,math,shutil,datetime,socket
from pathlib import Path
base=Path("/home/nate/claude/flash-sass-prs")
src=base/"quality-build-06968d1"/"kern";out=base/"quality-validation-06968d1"/"gpu";out.mkdir()
lock=open("/home/nate/codex/asm-port/GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=os.environ.copy()
for k in tuple(env):
 if k.startswith(("MODEL_","FLASH_")):env.pop(k)
env.update(MODEL_ATTN="qsa",MODEL_NOTWIN="1",MODEL_PLE="/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",MODEL_PLE_OFF="192",CUDA_VISIBLE_DEVICES="0,1,2,3,4,5",CUDA_DEVICE_ORDER="PCI_BUS_ID",SPLIT_SHARES="112,97,81,116,97,97",ATTN_DIE="1",ATTN_MAP="1,1,1,1,4,4,4,4,5,5,5,5")
def digest(p):
 b=p.read_bytes();return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def command(label,cmd,e=env,timeout=180):
 t=time.monotonic()
 with (out/(label+".stdout")).open("w") as stdout,(out/(label+".stderr")).open("w") as stderr:
  try:rc=subprocess.run(cmd,cwd=src,env=e,stdout=stdout,stderr=stderr,timeout=timeout).returncode
  except subprocess.TimeoutExpired:rc=124
 row=dict(label=label,argv=cmd,environment={k:v for k,v in e.items() if k.startswith(("MODEL_","FLASH_","CUDA_")) or k in ("SPLIT_SHARES","ATTN_DIE","ATTN_MAP")},exit=rc,seconds=time.monotonic()-t)
 print(json.dumps(row),flush=True);return row
before=command("gpu-before",["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm","--format=csv"])
rows=[];comparison=[]
shutil.copyfile(base/"weights-sha256.json",out/"weights-sha256.json")
for T,budget in ((4,512),(9,1)):
 for kv in (0,1):
  label=f"t{T}-budget{budget}-kv{kv}";ids=out/(label+"-ids.txt");ids.write_text(" ".join(str(10+i) for i in range(T))+"\n")
  capture=out/(label+"-raw");e=env.copy();e.update(MODEL_IDS=str(ids),MODEL_CAPTURE=str(capture),FLASH_NBLK=str(budget),FLASH_KV_HOST=str(kv))
  cmd=[str(src/"model_test"),"/unused-with-MODEL_NOTWIN","0","48",str(T),"eh"]+[f"/tmp/hold-h{i}.manifest" for i in range(6)]
  row=command(label,cmd,e);rows.append(row)
  if row["exit"]==0:
   raw=(capture/"logits.f32").read_bytes();values=array.array("f");values.frombytes(raw)
   actual_ids=array.array("I");actual_ids.frombytes((capture/"ids.u32").read_bytes())
   info=json.loads((capture/"capture_info.json").read_text())
   row["capture"]=dict(logits=digest(capture/"logits.f32"),ids=digest(capture/"ids.u32"),info=info,finite=all(math.isfinite(x) for x in values),minimum=min(values),maximum=max(values),expected_bytes=T*248320*4)
   row["passed"]=len(raw)==T*248320*4 and list(actual_ids)==list(range(10,10+T)) and row["capture"]["finite"] and info["positions"]==T and info["quality_validation"] is False
  else:row["passed"]=False
  (out/"results.json").write_text(json.dumps(dict(status="RUNNING",rows=rows),indent=2)+"\n")
  if not row["passed"]:raise SystemExit("Capture failed: "+label)
 a=out/f"t{T}-budget{budget}-kv0-raw"/"logits.f32";b=out/f"t{T}-budget{budget}-kv1-raw"/"logits.f32"
 comparison.append(dict(positions=T,budget=budget,placement_bit_exact=a.read_bytes()==b.read_bytes(),scope="same implementation and inputs; different KV placement"))
after=command("gpu-after",["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm","--format=csv"])
result=dict(status="PASS",hostname=socket.gethostname(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision="06968d1a5d6d7fdaf12106236cc39a1cb5cdf389",scope="Four short full-48-layer teacher-forced GPU captures with actual embedding, head, and disk PLE weights; valid token microfixtures; no scalar twin or independent original checkpoint comparison",quality_validation=False,independent_reference=False,rows=rows,placement_comparisons=comparison,gpu_before=before,gpu_after=after,weights_manifest=digest(out/"weights-sha256.json"))
(out/"results.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"status":"PASS","placement_comparisons":comparison}),flush=True)
