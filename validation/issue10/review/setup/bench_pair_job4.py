import json,os,pathlib,socket,subprocess,time
base=pathlib.Path("/home/nate/claude/flash-sass-prs")
source=base/"issue-10/kern";runner=base/"issue-14/kern/paired_depth_bench.py"
out=base/"paired-depth-untraced4";out.mkdir(exist_ok=False)
live=pathlib.Path("/home/nate/claude/flash-sass-next/kern")
def control(command):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(300);s.connect("/tmp/flash.sock");s.sendall((command+"\n").encode())
        with s.makefile("r") as f:line=f.readline().strip()
    print(line,flush=True)
    if not line or line.startswith("error"):raise RuntimeError(line)
    return line
def get(args):
    p=subprocess.run(args,capture_output=True,text=True)
    return dict(argv=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
build=dict(engine_source="bac5b88b665ee6973488dd40b363ded3134136f6",
    core_source="resident core executable; source revision not embedded; binary SHA256 recorded",
    command="make CUDA=/opt/cuda PTXAS11=/opt/cuda-11.4/bin/ptxas die.cubin die.cubin libengine.so",
    compiler=get(["cc","--version"]),assembler=get(["/opt/cuda-11.4/bin/ptxas","--version"]),
    flags="-std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -shared",
    original_checkpoint="Qwen/Qwen3.8-Flash-Next @ de4b8e4d43b917e7706784d8bb445c9af86a3540",
    gguf="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    conversion_revision=None,ple="disk, offset192, same NFS shard; warmup/repetitions; storage cache not flushed",
    authoritative=False,note="uninstrumented matched decode; shared holders and private host core; comparison against earlier traces requires clock/temperature qualification")
(out/"build.json").write_text(json.dumps(build,indent=2)+"\n")
env=os.environ.copy();env.update(FLASH_MAXT="8192",FLASH_ENGINE_MAXT="8192",FLASH_TRACE="0",FLASH_NBLK="512",FLASH_MAXIMG="1",FLASH_PLE_DISK="1",FLASH_PLE_OFF="192",FLASH_CUBIN=str(source/"die.cubin"),CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,7",SPLIT_SHARES="112,97,81,116,97,97",ATTN_DIE="1",ATTN_MAP="1,1,1,1,4,4,4,4,5,5,5,5")
(out/"launch-environment.json").write_text(json.dumps({k:v for k,v in env.items() if k.startswith(("FLASH_","CUDA_","SPLIT_","ATTN_"))},indent=2)+"\n")
cores=[];logs=[];code=1;records=[]
try:
    records.append(control(f"reload {source}/libengine.so FLASH_CUBIN={source}/die.cubin FLASH_TRACE=0 FLASH_ENGINE_MAXT=8192 FLASH_KV_HOST=0 FLASH_MAXIMG=1"))
    sock="/tmp/flash-bench-host4.sock"
    e=env.copy();e.update(FLASH_KV_HOST="1",FLASH_SOCK=sock)
    log=(out/"host-core.log").open("w");logs.append(log)
    cmd=[str(live/"flashcore"),"/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",str(source/"libengine.so"),*[f"/tmp/hold-h{i}.manifest" for i in range(6)]]
    p=subprocess.Popen(cmd,cwd=source,env=e,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);cores.append((p,sock))
    deadline=time.monotonic()+30
    while not pathlib.Path(sock).exists():
        if p.poll() is not None:raise RuntimeError("host core exited")
        if time.monotonic()>deadline:raise RuntimeError("host core not ready")
        time.sleep(.1)
    cmd=["python3",str(runner),"--host-socket",sock,"--vram-socket","/tmp/flash.sock","--out",str(out/"runs"),"--source-sha",build["engine_source"],"--depths","3100,8000","--build-record",str(out/"build.json"),"--weight-manifest",str(base/"weights-sha256.json")]
    for artifact in (source/"libengine.so",source/"die.cubin",live/"flashcore"):cmd+=["--artifact",str(artifact)]
    (out/"command.json").write_text(json.dumps(cmd,indent=2)+"\n")
    code=subprocess.run(cmd,timeout=1800,env=env).returncode
finally:
    for p,path in cores:
        if p.poll() is None:
            try:
                with socket.socket(socket.AF_UNIX) as s:
                    s.settimeout(5);s.connect(path);s.sendall(b"quit\n")
                p.wait(timeout=15)
            except Exception:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
    for f in logs:f.close()
    records.append(control(f"reload {live}/libengine.so FLASH_CUBIN={live}/die.cubin FLASH_TRACE=0 FLASH_ENGINE_MAXT=230000 FLASH_KV_HOST=0 FLASH_MAXIMG=4096"))
    records.append(control("config"))
    (out/"job-exit.json").write_text(json.dumps(dict(exit=code,control=records),indent=2)+"\n")
raise SystemExit(code)
