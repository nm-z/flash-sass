#!/usr/bin/env python3
"""Run checks on fresh, owned cores that import the existing holder weights."""
import os,json,subprocess,time,hashlib,fcntl,socket,datetime,sys,signal
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text());src=Path(cfg["source"])/"kern";out=Path(cfg["validation"])/cfg["label"];out.mkdir(parents=True,exist_ok=False)
lock=open("/home/nate/codex/asm-port/GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows=[];records=[];passed=False
def write():
 (out/"results.json").write_text(json.dumps(dict(status="PASS" if passed else "RUNNING_OR_FAILED",config=cfg,commands=rows,cores=records,runner_sha256=sha(__file__)),indent=2)+"\n")
def run(label,argv,env,timeout=360):
 t=time.monotonic()
 with (out/(label+".stdout")).open("w") as f,(out/(label+".stderr")).open("w") as g:
  try:rc=subprocess.run(argv,cwd=src,env=env,stdout=f,stderr=g,timeout=timeout).returncode
  except subprocess.TimeoutExpired:rc=124
 row=dict(label=label,argv=argv,exit=rc,seconds=time.monotonic()-t);rows.append(row);write();print(json.dumps(row),flush=True)
 if rc:raise RuntimeError(label+" returned "+str(rc))
def query(sock,cmd):
 with socket.socket(socket.AF_UNIX) as s:
  s.settimeout(5);s.connect(sock);s.sendall((cmd+"\n").encode())
  with s.makefile("r") as f:return f.readline().strip()
env=os.environ.copy()
for k in tuple(env):
 if k.startswith(("FLASH_","MODEL_")):env.pop(k)
env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,7",CUDA_DEVICE_ORDER="PCI_BUS_ID",SPLIT_SHARES="112,97,81,116,97,97",ATTN_DIE="1",ATTN_MAP="1,1,1,1,4,4,4,4,5,5,5,5",FLASH_ANCHOR_DIE="7",FLASH_MAXT="512",FLASH_ENGINE_MAXT="512",FLASH_MAXIMG="1",FLASH_PLE_DISK="1",FLASH_PLE_OFF="192",FLASH_NBLK="512",FLASH_KV_HOST="0",FLASH_WATCHDOG_S="60")
env.update(cfg.get("extra_env",{}))
run("gpu-before",["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm","--format=csv"],env)
for paired in cfg.get("paired_modes",[0]):
 e=env.copy();e.update(FLASH_TWO_POSITION=str(paired),FLASH_TRACE=str(cfg.get("trace_tokens",0)),FLASH_CUBIN=str(src/cfg.get("cubin","die.cubin")))
 sock="/tmp/flash-owned-"+cfg["label"]+"-pair"+str(paired)+".sock";assert not Path(sock).exists();e["FLASH_SOCK"]=sock
 lib=src/cfg.get("library","libengine.so")
 cmd=[str(src/"flashcore"),"/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",str(lib)]+[f"/tmp/hold-h{i}.manifest" for i in range(6)]
 log=(out/("core-pair"+str(paired)+".log")).open("w")
 cmd=cfg.get("core_prefix",[])+cmd
 p=subprocess.Popen(cmd,cwd=src,env=e,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 record=dict(pid=p.pid,argv=cmd,environment={k:v for k,v in e.items() if k.startswith(("FLASH_","CUDA_")) or k in ("SPLIT_SHARES","ATTN_DIE","ATTN_MAP")},library_sha256=sha(lib),cubin_sha256=sha(e["FLASH_CUBIN"]),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());records.append(record);write()
 try:
  deadline=time.monotonic()+90
  while True:
   if p.poll() is not None:raise RuntimeError("Owned core exited "+str(p.returncode))
   try:
    line=query(sock,"config")
    if line.startswith("config "):break
   except (OSError,TimeoutError):pass
   if time.monotonic()>deadline:raise RuntimeError("Owned core startup timeout")
   time.sleep(.1)
  record["config"]=line;write()
  assert "rev="+cfg["revision"] in line,line
  assert "layers="+str(cfg.get("layers",48))+"/48" in line and "ple_disk=1" in line,line
  if cfg["kind"]=="ple":
   run("cpu-pair"+str(paired),[str(src/"ple_io_test")],e)
   run("descriptor-pair"+str(paired),[str(src/"pair_prog_test")],e)
   run("socket-pair"+str(paired),["python3",str(src/"ple_io_check.py"),sock],e)
   script=Path(cfg["http_script"]);record["http_script_sha256"]=sha(script)
   run("http-pair"+str(paired),["python3",str(script),"--flashd",str(src/"flashd"),"--gguf","/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf","--core",sock,"--out",str(out/("http-pair"+str(paired)))],e)
  elif cfg["kind"]=="trace":
   command=["python3",str(src/"trace_lifecycle_check.py"),"--socket",sock,"--source-sha",cfg["revision"],"--out",str(out/("trace-pair"+str(paired)))]
   if paired:command.append("--pair-check")
   run("lifecycle-pair"+str(paired),command,e)
  elif cfg["kind"]=="command":
   for k,command in enumerate(cfg["commands"]):
    command=[v.format(source=str(src),socket=sock,out=str(out),revision=cfg["revision"],paired=paired) for v in command]
    run("command"+str(k)+"-pair"+str(paired),command,e,cfg.get("command_timeout",360))
  else:raise ValueError("Unknown validation kind")
  record["post_check_info"]=query(sock,"info")
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
  record["owned_core_exit"]=p.returncode;record["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat();log.close();write()
run("gpu-after",["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.free,clocks.sm","--format=csv"],env)
passed=True;write();print("PASS owned-core validation",flush=True)
