#!/usr/bin/env python3
"""Measure matched host/VRAM resident cores with fixed tokens and raw events.
Synthetic token prompts exercise the real model; they are not quality tests.
Only connects to supplied cores. Does not reload cores or weight holders.
Live traced measurements require explicit successful lifecycle metadata.
Historical analysis accepts older missing metadata as unknown, never inferred success.
"""
import argparse,csv,gzip,hashlib,json,math,os,pathlib,random,re,socket,statistics,subprocess,sys,time
from collections import defaultdict
def save(path,x):pathlib.Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def run(args):
    p=subprocess.run(args,text=True,capture_output=True)
    return dict(argv=args,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def identity(path):
    p=pathlib.Path(path).resolve();a=p.stat();h=sha(p);b=p.stat()
    if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):raise ValueError("file changed while hashing")
    return dict(path=str(p),bytes=b.st_size,mtime_ns=b.st_mtime_ns,sha256=h)
def p95(x):return sorted(x)[math.ceil(.95*len(x))-1]
def gpu():
    return run(["nvidia-smi","--query-gpu=index,uuid,driver_version,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.sw_thermal_slowdown,power.draw,power.limit","--format=csv"])
def driver_snapshot():
    # Driver 470 supports the full XML dump, but not the PCIE -d selector.
    result=run(["nvidia-smi","-q","-x"])
    if result["exit"] or "<nvidia_smi_log>" not in result["stdout"]:
        raise ValueError("driver metadata query failed: "+json.dumps(result))
    return result
def core_records(path):
    data=json.loads(pathlib.Path(path).read_text())
    if not isinstance(data,dict) or set(data)!={"host","vram"}:
        raise ValueError("core records must name exactly host and vram")
    for name,record in data.items():
        if not isinstance(record,dict):raise ValueError("invalid core record "+name)
        argv=record.get("argv");env=record.get("launch_environment")
        if not isinstance(argv,list) or not argv or any(not isinstance(x,str) or not x for x in argv):
            raise ValueError("missing core argv "+name)
        if not isinstance(env,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in env.items()):
            raise ValueError("invalid launch environment "+name)
        if not isinstance(record.get("engine_reload_commands"),list):
            raise ValueError("engine reload commands must be an explicit list "+name)
        for key in ("FLASH_MAXT","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_ORDER"):
            if key not in env:raise ValueError("missing launch setting "+name+"."+key)
        if not isinstance(record.get("limitations"),list):raise ValueError("explicit limitations list required "+name)
    return data
def fields(line):return dict(x.split("=",1) for x in line.split()[1:] if "=" in x)
class Core:
    def __init__(self,path,timeout):
        self.s=socket.socket(socket.AF_UNIX);self.s.settimeout(timeout);self.s.connect(path);self.f=self.s.makefile("r")
    def send(self,x):self.s.sendall((x+"\n").encode())
    def line(self):
        x=self.f.readline()
        if not x:raise RuntimeError("core disconnected")
        if x.startswith("error"):raise RuntimeError(x.strip())
        return x.rstrip("\n")
    def one(self,x):self.send(x);return self.line()
    def close(self):self.f.close();self.s.close()
def matched(a,b,allow):
    diff={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k) and k not in allow}
    if diff:raise ValueError("unexplained manifest differences: "+json.dumps(diff,sort_keys=True))
def trace_lifecycle(meta, *, require_current=False, expected_requested=None, allow_ended=True):
    """Historical missing fields remain unknown; live evidence requires all fields."""
    required={"status","aborted","ended","completed","requested"}
    present=required.intersection(meta)
    if not present:
        if require_current:raise ValueError("live trace lacks lifecycle metadata")
        return dict(reported=False,status="legacy_unreported")
    if present!=required:raise ValueError("partial trace lifecycle metadata")
    if meta["aborted"] not in ("0","1") or meta["ended"] not in ("0","1"):
        raise ValueError("invalid trace lifecycle boolean")
    for key in ("completed","requested"):
        if not isinstance(meta[key],str) or not re.fullmatch(r"[0-9]+",meta[key]):
            raise ValueError("invalid trace lifecycle count "+key)
    completed,requested=int(meta["completed"]),int(meta["requested"])
    ended=int(meta["ended"])
    if meta["status"]!="completed" or meta["aborted"]!="0":
        raise ValueError("unsuccessful trace request: "+str(meta["status"]))
    if not 1<=completed<=requested:raise ValueError("impossible trace lifecycle counts")
    if not ended and completed!=requested:raise ValueError("incomplete trace request")
    if ended and not allow_ended:raise ValueError("stopped trace is not a fixed-continuation benchmark")
    if expected_requested is not None and requested!=expected_requested:
        raise ValueError("trace belongs to different requested positions")
    if "first" in meta and "count" in meta:
        first,count=int(meta["first"]),int(meta["count"])
        if first<0 or count<1 or first+count>completed:
            raise ValueError("trace window exceeds completed positions")
    return dict(reported=True,status="completed",aborted=0,ended=ended,
                completed=completed,requested=requested)

def phase_summary(rows,meta):
    """Partition local-die elapsed time, without adding overlapping die work."""
    lifecycle=trace_lifecycle(meta)
    nph,count,first=(int(meta[x]) for x in ("nph","count","first"))
    if nph<1 or count<1 or int(meta["ctas"])!=1:raise ValueError("bad trace dimensions")
    seen=0;previous=0;tokens=[];ops=defaultdict(lambda:[0]*5);keys=set();opcounts=defaultdict(int)
    for row in rows:
        pos,phase=int(row["position"]),int(row["phase"])
        if (pos,phase)!=(first+seen//nph,seen%nph):raise ValueError("missing/duplicate/out-of-order phase")
        z=[int(row[x]) for x in ("start_ns","wait_start_ns","wait_end_ns","body_end_ns","end_ns")]
        if not z[0] or z!=sorted(z) or z[0]<previous:raise ValueError("nonmonotonic trace")
        ps,pe,pub=(int(row[x]) for x in ("ple_wait_start_ns","ple_wait_end_ns","publication_ns"))
        if (ps or pe) and not(z[2]<=ps<=pe<=z[3]):raise ValueError("bad PLE wait interval")
        if phase==0:tokens.append(dict(position=pos,start_ns=z[0],phase_ns=0))
        tok=tokens[-1];tok["phase_ns"]+=z[4]-z[0];tok["end_ns"]=z[4]
        if phase==nph-1:
            if pub<z[4]:raise ValueError("missing publication")
            tok["publication_ns"]=pub
        elif pub:raise ValueError("publication on a nonterminal phase")
        if pos>first:
            v=ops[row["opcode"]];parts=[z[1]-z[0],z[2]-z[1],z[3]-z[2]-(pe-ps),z[4]-z[3],pe-ps]
            for i,t in enumerate(parts):v[i]+=t
            opcounts[row["opcode"]]+=1
        keys.add((pos,int(row["selected_keys"])));previous=z[-1];seen+=1
    if seen!=count*nph:raise ValueError("incomplete trace")
    for t in tokens:
        t["between_phase_ns"]=t["end_ns"]-t["start_ns"]-t["phase_ns"]
        t["publication_footer_ns"]=t["publication_ns"]-t["end_ns"]
    intervals=[tokens[i]["publication_ns"]-tokens[i-1]["publication_ns"] for i in range(1,count)]
    if any(x<=0 for x in intervals):raise ValueError("bad device interval")
    labels=("dispatch","exchange_wait","body_excluding_ple_wait","barrier_send","ple_wait")
    return dict(metadata=meta,lifecycle=lifecycle,tokens=tokens,intervals_ns=intervals,decode_ms=sum(intervals)/1e6,
                op_mean_us={op:{name:value/1000/max(1,count-1) for name,value in zip(labels,v)} for op,v in ops.items()},
                op_counts=dict(opcounts),selected_keys=sorted(keys),
                note="Local clock per die; these die totals overlap and cannot be summed.")
def capture(core,command,prefix,kind):
    core.send(command);meta=core.line()
    if not meta.startswith(kind+"_begin "):raise ValueError("bad trace response "+meta)
    path=pathlib.Path(str(prefix)+".csv.gz")
    with gzip.open(path,"wt",newline="") as f:
        while (x:=core.line())!=kind+"_end":f.write(x+"\n")
    return path,fields(meta)
def host_summary(rows,meta):
    trace_lifecycle(meta)
    data=list(rows);first,count=int(meta["first"]),int(meta["count"])
    if len(data)!=count or meta["clock"]!="CLOCK_MONOTONIC":raise ValueError("bad host trace dimensions")
    previous=0
    for i,row in enumerate(data):
        if int(row["position"])!=first+i:raise ValueError("bad host position")
        ob,cs,ce=(int(row[x]) for x in ("observed_ns","callback_start_ns","callback_end_ns"))
        if ob<previous or not(0<ob<=cs<=ce):raise ValueError("bad host callback ordering")
        ps,pe,ready=(int(row[x]) for x in ("ple_staging_start_ns","ple_staging_end_ns","ple_ready_ns"))
        if (ps or pe or ready) and not(0<ps<=pe<=ready):raise ValueError("bad host staging order")
        if int(row["ple_short_reads"]):raise ValueError("PLE short read invalidates benchmark")
        previous=ob
    return data
def request(core,ids,outputs,prefix,trace,dies,expected_resume):
    core.one(f"snap {len(ids)-1}");text="chat "+str(outputs-1)+" "+" ".join(map(str,ids))
    started=time.monotonic_ns();core.send(text);raw=[];tokens=[];observed=[]
    while True:
        line=core.line();now=time.monotonic_ns();raw.append(line)
        if line.startswith("tok "):tokens.append(int(line.split()[1]));observed.append(now)
        elif line.startswith("done "):
            x=line.split();done=dict(generated=int(x[1]),ended=int(x[2]),prompt=int(x[3]),run_ms=float(x[4]),first_ms=float(x[5]),resumed=int(x[6]),snapshot=int(x[7]));break
        else:raise ValueError("unexpected request response "+line)
    row=dict(request=text,request_start_ns=started,request_end_ns=now,generated_ids=tokens,client_observed_ns=observed,raw=raw,done=done)
    save(str(prefix)+".json",row)
    if len(tokens)!=outputs or done["generated"]!=outputs or done["ended"] or done["prompt"]!=len(ids):raise ValueError("incomplete or stopped request")
    if expected_resume is not None and done["resumed"]!=expected_resume:raise ValueError("prefix snapshot was not restored")
    intervals=[b-a for a,b in zip(observed,observed[1:])]
    if any(x<=0 for x in intervals):raise ValueError("invalid client token times")
    row.update(intervals_ns=intervals,decode_intervals=len(intervals),client_decode_ms=sum(intervals)/1e6,
               request_ttft_ms=(observed[0]-started)/1e6,
               request_completion_ms=(now-started)/1e6,reported_post_first_ms=done["run_ms"]-done["first_ms"])
    if trace:
        traces=[];request_id=None
        for die in range(dies):
            path,meta=capture(core,f"trace {die}",str(prefix)+f"-die-{die}","trace")
            trace_lifecycle(meta,require_current=True,expected_requested=len(ids)+outputs-1,allow_ended=False)
            if int(meta["count"])!=outputs or int(meta["first"])!=len(ids)-1 or int(meta["die"])!=die:raise ValueError("wrong trace request dimensions")
            if request_id is None:request_id=meta["request"]
            if meta["request"]!=request_id:raise ValueError("different die trace requests")
            with gzip.open(path,"rt") as f:summary=phase_summary(csv.DictReader(f),meta)
            save(str(path)+".summary.json",summary);traces.append(summary)
        path,meta=capture(core,"hosttrace",str(prefix)+"-host","hosttrace")
        row["lifecycle"]=trace_lifecycle(meta,require_current=True,expected_requested=len(ids)+outputs-1,allow_ended=False)
        if meta["request"]!=request_id or int(meta["count"])!=outputs:raise ValueError("different host trace request")
        with gzip.open(path,"rt") as f:host=host_summary(csv.DictReader(f),meta)
        row["device_publication_decode_ms_by_die"]=[t["decode_ms"] for t in traces]
        row["engine_observed_decode_ms"]=(int(host[-1]["observed_ns"])-int(host[0]["observed_ns"]))/1e6
        row["client_minus_engine_interval_ms"]=row["client_decode_ms"]-row["engine_observed_decode_ms"]
        row["engine_minus_gpu0_interval_ms"]=row["engine_observed_decode_ms"]-traces[0]["decode_ms"]
    row["gpu_after"]=gpu();row["status"]="complete";save(str(prefix)+".json",row);return row
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host-socket",required=True);p.add_argument("--vram-socket",required=True);p.add_argument("--out",required=True)
    p.add_argument("--depths",default="2048,2051,2052,3100,8000,65000,262000")
    p.add_argument("--outputs",type=int,default=129);p.add_argument("--pairs",type=int,default=5);p.add_argument("--warmup-outputs",type=int,default=9)
    p.add_argument("--trace",action="store_true");p.add_argument("--timeout",type=float,default=18000)
    p.add_argument("--source-sha",required=True);p.add_argument("--dirty-patch");p.add_argument("--artifact",action="append",required=True)
    p.add_argument("--weight-manifest",required=True);p.add_argument("--build-record",required=True)
    p.add_argument("--core-records",required=True,help="JSON: per-core launch argv/environment, reload commands and limitations")
    a=p.parse_args()
    launch=core_records(a.core_records)
    if not re.fullmatch("[0-9a-f]{40}",a.source_sha):p.error("source-sha must be a full SHA")
    if a.outputs<2 or a.outputs>256 or a.pairs<5 or a.warmup_outputs<2:p.error("outputs2..256, pairs>=5, warmup>=2")
    depths=[int(x) for x in a.depths.split(",")]
    if min(depths)<1 or depths!=sorted(set(depths)):p.error("unique ascending positive depths required")
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(4242);vocab=[198,220,11,13,279,315,323,374,499,264,311,389]
    ids=[rng.choice(vocab) for _ in range(max(depths))];save(out/"prompt-ids.json",ids)
    weights=json.loads(pathlib.Path(a.weight_manifest).read_text())
    for w in weights:
        st=pathlib.Path(w["path"]).stat()
        if st.st_size!=w["bytes"] or st.st_mtime_ns!=w["mtime_ns"]:raise ValueError("stale weight identity")
    manifest=dict(argv=sys.argv,source_sha=a.source_sha,dirty_patch=pathlib.Path(a.dirty_patch).read_text() if a.dirty_patch else None,
                  runner=identity(__file__),artifacts=[identity(x) for x in a.artifact],weights=weights,
                  build=json.loads(pathlib.Path(a.build_record).read_text()),core_launch_records=launch,core_records_identity=identity(a.core_records),prompt=identity(out/"prompt-ids.json"),
                  sampling=dict(seed=0,temperature=0,stop_ids=[]),warmup_outputs=a.warmup_outputs,pairs=a.pairs,outputs=a.outputs,
                  fixture="synthetic fixed token IDs; real checkpoint forward; no answer-quality acceptance",
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),gpu_before=gpu(),host=run(["uname","-a"]),
                  environment={k:v for k,v in os.environ.items() if k.startswith(("FLASH_","CUDA_","SPLIT_","ATTN_","DIE_"))},
                  driver_counters_before=driver_snapshot())
    cores={};results=[];exitcode=1
    try:
        configs={}
        for name,sock in (("host",a.host_socket),("vram",a.vram_socket)):
            c=Core(sock,a.timeout);cores[name]=c;configs[name]=fields(c.one("config"));c.one("stop");c.one("sample 0 0 0 1")
        matched(configs["host"],configs["vram"],{"kv_host"})
        if configs["host"]["kv_host"]!="1" or configs["vram"]["kv_host"]!="0":raise ValueError("wrong realized KV placement")
        cfg=configs["host"]
        if cfg["layers"]!="48/48" or cfg["nblk"]!="512":raise ValueError("requires full48-layer budget512 QSA")
        if a.trace and int(cfg.get("trace_tokens",0))<a.outputs:raise ValueError("insufficient trace capacity")
        manifest["configs"]=configs;save(out/"manifest.json",manifest)
        cap=int(cfg["maxt"]);dies=int(cfg["dies"]);prefixes={"host":0,"vram":0}
        for depth in depths:
            if depth+a.outputs-1>cap:
                row=dict(depth=depth,status="capacity_skip",capacity=cap);results.append(row);print(json.dumps(row),flush=True);continue
            for name in ("host","vram"):
                r=request(cores[name],ids[:depth],a.warmup_outputs,out/f"warmup-{depth}-{name}",False,dies,None)
                save(out/f"warmup-{depth}-{name}.json",r);prefixes[name]=depth-1
            for pair in range(a.pairs):
                order=("host","vram") if pair%2==0 else ("vram","host");paired={}
                for name in order:
                    prefix=out/f"depth-{depth}-pair-{pair}-{name}"
                    row=request(cores[name],ids[:depth],a.outputs,prefix,a.trace,dies,depth-1)
                    row.update(depth=depth,pair=pair,placement=name,order=list(order));save(str(prefix)+".json",row)
                    results.append(row);paired[name]=row
                    print(json.dumps(dict(depth=depth,pair=pair,placement=name,tok_s=1000*row["decode_intervals"]/row["client_decode_ms"],resumed=row["done"]["resumed"])),flush=True)
                if paired["host"]["generated_ids"]!=paired["vram"]["generated_ids"]:raise ValueError("matched placements generated different tokens")
                save(out/"results.json",results)
        summary=[]
        for depth in depths:
            for name in ("host","vram"):
                rr=[r for r in results if r.get("depth")==depth and r.get("placement")==name and r["status"]=="complete"]
                if rr:
                    raw=[x/1e6 for r in rr for x in r["intervals_ns"]]
                    means=[r["client_decode_ms"]/r["decode_intervals"] for r in rr]
                    summary.append(dict(depth=depth,placement=name,runs=len(rr),intervals=len(raw),median_run_ms_per_token=statistics.median(means),p95_interval_ms=p95(raw),median_tok_s=1000/statistics.median(means)))
        manifest["driver_counters_after"]=driver_snapshot();save(out/"manifest.json",manifest)
        save(out/"summary.json",summary);save(out/"status.json",dict(status="PASS",runs=sum(r.get("status")=="complete" for r in results)))
        exitcode=0
    except Exception as e:
        save(out/"FAILED.json",dict(error=repr(e)));raise
    finally:
        for c in cores.values():c.close()
        save(out/"exit.json",dict(exit=exitcode));save(out/"results.json",results)
if __name__=="__main__":main()
