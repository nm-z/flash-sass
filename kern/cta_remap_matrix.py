#!/usr/bin/env python3
"""Compare exact parent/remap cubins on one exclusively locked M60; keep raw evidence."""
import argparse, fcntl, hashlib, json, os, pathlib, statistics, subprocess, sys, time

def record(path, data):
    pathlib.Path(path).write_text(json.dumps(data, indent=2) + "\n")
def run(args, **kw):
    r = subprocess.run(args, text=True, capture_output=True, **kw)
    return {"argv": args, "exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
def image(path):
    p = pathlib.Path(path).resolve()
    return {"path":str(p), "sha256":hashlib.sha256(p.read_bytes()).hexdigest(), "bytes":p.stat().st_size}
def gpu(uuid):
    return run(["nvidia-smi", "-i", uuid, "--query-gpu=uuid,name,driver_version,pci.bus_id,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current", "--format=csv"])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True); p.add_argument("--new", required=True)
    p.add_argument("--old-library", required=True); p.add_argument("--new-library", required=True)
    p.add_argument("--old-source", default="c6d3de8180d8b87caf0d2f61fa52edc9247daded")
    p.add_argument("--new-source", default="66725432f86cc193b75cd51230216aed44c60d3f")
    p.add_argument("--harness", default="./attnpartn_regress")
    p.add_argument("--harness-source", required=True)
    p.add_argument("--gpu", required=True); p.add_argument("--out", required=True)
    p.add_argument("--keys", default="1,1023,1024,1025,2047,2048,2049,2050,2051,2052,4095,4096,4097,8191,8192,8193")
    p.add_argument("--seeds", default="0,77,4242"); p.add_argument("--repeats", type=int, default=32)
    p.add_argument("--instrumentation", default="none", help="literal build flags/patch identity if tracing")
    a = p.parse_args()
    counts, seeds = [int(x) for x in a.keys.split(",")], [int(x) for x in a.seeds.split(",")]
    if min(counts)<1 or max(counts)>16384 or a.repeats<4 or a.repeats%2:
        p.error("keys 1..16384 and an even repetition count >=4 required")
    out=pathlib.Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=a.gpu)
    lockpath=pathlib.Path("/home/nate/codex/asm-port")/(a.gpu+".lock")
    rows=[]
    with lockpath.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest={"arguments":vars(a),"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "old_cubin":image(a.old),"new_cubin":image(a.new),"old_library":image(a.old_library),"new_library":image(a.new_library),
            "harness":image(a.harness),"gpu_before":gpu(a.gpu),"compiler":run(["cc","--version"]),
            "ptxas":run(["/opt/cuda-11.4/bin/ptxas","--version"]),
            "environment":{k:v for k,v in env.items() if k.startswith(("CUDA_","ATTN_","DIE_"))},
            "timing":"64 positions per persistent launch, four warmup pairs, balanced alternating old/new order; one-phase partial and two-phase partial+merge programs; GPU events exclude allocation/upload and CPU checks",
            "input":"finite fixed-seed FP32 values; seed0 has zero query; padding finite zero"}
        record(out/"manifest.json",manifest)
        try:
            for n in counts:
                for seed in seeds:
                    args=[str(pathlib.Path(a.harness).resolve()),str(pathlib.Path(a.old).resolve()),str(pathlib.Path(a.new).resolve()),str(n),str(seed),str(a.repeats)]
                    r=run(args,env=env,timeout=180)
                    r["gpu_after"]=gpu(a.gpu)
                    record(out/f"keys-{n}-seed-{seed}.json",r)
                    if r["exit"]: raise RuntimeError(f"GPU comparison failed: n={n}, seed={seed}, exit={r['exit']}")
                    value=json.loads(r["stdout"])
                    if not value["bit_exact"] or value["partial_values"]!=49536 or value["final_values"]!=6144 or value["steps_per_sample"]!=64:
                        raise RuntimeError("missing comparison or timing coverage")
                    rows.append(value)
                    print(json.dumps({"keys":n,"seed":seed,"exact":True,"old_us":value["old"]["event"]["median_us"],"new_us":value["new"]["event"]["median_us"]}),flush=True)
            summary=[]
            for n in counts:
                cases=[r for r in rows if r["keys"]==n]
                old=statistics.median(r["old"]["event"]["median_us"] for r in cases)
                new=statistics.median(r["new"]["event"]["median_us"] for r in cases)
                summary.append({"keys":n,"seeds":len(cases),"old_partial_merge_us":old,"new_partial_merge_us":new,"delta_us":new-old,"speedup":old/new})
            record(out/"summary.json",summary); record(out/"status.json",{"status":"PASS","cases":len(rows)})
        except Exception as exc:
            record(out/"status.json",{"status":"FAIL","completed":len(rows),"error":str(exc)})
            raise
    return 0
if __name__=="__main__": sys.exit(main())
