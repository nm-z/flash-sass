#!/usr/bin/env python3
import hashlib,importlib.util,json,os,socket,sys,xml.etree.ElementTree as ET
from pathlib import Path
base=Path("/home/nate/claude/flash-sass-prs")
source=base/"metadata-smoke-ec86562/paired_depth_bench.py"
out=base/"metadata-smoke-ec86562/results";out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location("paired",source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
xml=mod.driver_snapshot();root=ET.fromstring(xml["stdout"])
assert root.tag=="nvidia_smi_log" and len(root.findall("gpu"))==8
(out/"nvidia-smi.xml").write_text(xml["stdout"])
gpu=mod.gpu()
assert gpu["exit"]==0 and "sw_power_cap" in gpu["stdout"]
(out/"gpu.csv").write_text(gpu["stdout"])
records={}
env=json.loads((base/"paired-depth-untraced4/launch-environment.json").read_text())
env.update(FLASH_KV_HOST="1",FLASH_SOCK="/tmp/flash-bench-host4.sock")
live=Path("/home/nate/claude/flash-sass-next/kern")
records["host"]=dict(argv=[str(live/"flashcore"),"/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",str(base/"issue-10/kern/libengine.so"),*[f"/tmp/hold-h{i}.manifest" for i in range(6)]],launch_environment=env,engine_reload_commands=[],limitations=["Launch argv/environment reconstructed from the retained job4 setup script and launch record after that owned core exited. This is a supplemental operator record, not metadata captured before the original experiment."])
pid=1023383
proc=Path(f"/proc/{pid}")
argv=[s.decode() for s in (proc/"cmdline").read_bytes().split(b"\0") if s]
env={}
for raw in (proc/"environ").read_bytes().split(b"\0"):
    if raw and b"=" in raw:
        k,v=raw.decode().split("=",1)
        if k.startswith(("FLASH_","CUDA_","SPLIT_","ATTN_","DIE_")):env[k]=v
records["vram"]=dict(argv=argv,launch_environment=env,engine_reload_commands=[f"reload {base}/issue-10/kern/libengine.so FLASH_CUBIN={base}/issue-10/kern/die.cubin FLASH_TRACE=0 FLASH_ENGINE_MAXT=8192 FLASH_KV_HOST=0 FLASH_MAXIMG=1"],limitations=["Launch-only environment read from /proc after the experiment; process identity confirms the core that ran job4. /proc environment does not reflect all reload-time setenv values.", "Core construction capacity differs from the private host core; both effective engine capacities were 8192. Core-level PLE staging allocation can differ.", "This record does not recover the unrecorded original process launch environment of the earlier traced job3."])
path=out/"job4-supplemental-core-records.json";mod.save(path,records);validated=mod.core_records(path)
report=dict(status="PASS",scope="Current runner metadata-query and launch-record parser smoke only; no model forward or performance acceptance claimed.",runner_source_revision="ec86562bdbdec08f2f79e2f39aaa87eefd1c467f",runner=mod.identity(source),driver_query=dict(argv=xml["argv"],exit=xml["exit"],xml_gpu_count=len(root.findall("gpu"))),gpu_query=dict(argv=gpu["argv"],exit=gpu["exit"]),current_core_pid=pid,record_keys=sorted(validated),files={p.name:mod.identity(p) for p in out.iterdir() if p.is_file()})
mod.save(out/"result.json",report)
print(json.dumps({k:report[k] for k in ("status","scope","runner_source_revision","driver_query","gpu_query","current_core_pid","record_keys")}))
