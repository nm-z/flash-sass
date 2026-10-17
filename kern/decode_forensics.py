#!/usr/bin/env python3
"""Summarize matched decode records and reconcile one die's elapsed timeline.
All GPU epochs remain local to a die. Distinct attention-layer operation work
is reported separately from the overlapping receiver waits.
"""
import argparse,csv,gzip,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
PARTS=("dispatch","exchange_wait","body_excluding_ple_wait","barrier_send","ple_wait")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
    text=json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if str(p).endswith(".gz"):
        with Path(p).open("wb") as f:
            with gzip.GzipFile(filename="",fileobj=f,mode="wb",mtime=0) as g:g.write(text.encode())
    else:Path(p).write_text(text)
def p95(v):return sorted(v)[math.ceil(.95*len(v))-1]
def timeline(s):
    t=s["tokens"];n=len(t)-1
    if n!=128:raise ValueError("expected 129 publication events")
    parts={k:sum(v[k] for v in s["op_mean_us"].values()) for k in PARTS}
    parts["between_phase"]=sum(x["between_phase_ns"] for x in t[1:])/n/1000
    parts["publication_footer"]=sum(x["publication_footer_ns"] for x in t[1:])/n/1000
    parts["between_token"]=sum(t[i]["start_ns"]-t[i-1]["publication_ns"] for i in range(1,len(t)))/n/1000
    exact=(t[-1]["publication_ns"]-t[0]["publication_ns"])/n/1000
    residual=exact-sum(parts.values())
    if abs(residual)>1e-5:raise ValueError(f"unreconciled timeline {residual} us")
    return {"components_us_per_token":parts,"device_us_per_token":exact,"reconciliation_residual_us":residual}
def process(root,out,tag,traced):
    root=Path(root);manifest=load(root/"manifest.json");results=load(root/"results.json")
    status=load(root/"status.json");exit_status=load(root/"exit.json")
    if status.get("status")!="PASS" or exit_status.get("exit")!=0:raise ValueError("capture did not finish successfully")
    if manifest["pairs"]!=5 or manifest["outputs"]!=129:raise ValueError("unexpected repetition or output count")
    configs=manifest["configs"]
    if set(configs)!={"host","vram"} or configs["host"]["kv_host"]!="1" or configs["vram"]["kv_host"]!="0":
        raise ValueError("missing paired KV placements")
    if {k:v for k,v in configs["host"].items() if k!="kv_host"}!={k:v for k,v in configs["vram"].items() if k!="kv_host"}:
        raise ValueError("unmatched realized configurations")
    for v in configs.values():
        if int(v["nblk"])!=512 or v["layers"]!="48/48":raise ValueError("unexpected model configuration")
    inputs={};runs=[];phase=[];groups=defaultdict(list);skips=[];seen=set()
    for row in results:
        if row["status"]!="complete":
            if row["status"]!="capacity_skip" or row["depth"]+128<=row["capacity"]:raise ValueError("unexpected rejected sample")
            skips.append(row);continue
        key=(row["depth"],row["pair"],row["placement"])
        if key in seen or row["pair"] not in range(5) or row["placement"] not in configs:raise ValueError("duplicate or invalid pair")
        seen.add(key)
        observed=row["client_observed_ns"]
        if len(observed)!=129 or row["intervals_ns"]!=[b-a for a,b in zip(observed,observed[1:])]:
            raise ValueError("client intervals do not match raw observations")
        if any(x<=0 for x in row["intervals_ns"]) or abs(sum(row["intervals_ns"])/1e6-row["client_decode_ms"])>1e-9:
            raise ValueError("invalid client decode duration")
        if row["done"]["generated"]!=129 or row["done"]["prompt"]!=row["depth"]:raise ValueError("wrong completion dimensions")
        if row["decode_intervals"]!=128 or len(row["generated_ids"])!=129 or row["done"]["ended"]:raise ValueError("invalid decode sample")
        if row["done"]["resumed"]!=row["depth"]-1:raise ValueError("missing prefix reuse")
        k=(row["depth"],row["pair"]);ids=row["generated_ids"]
        if k in inputs and inputs[k]!=ids:raise ValueError("paired continuations differ")
        inputs[k]=ids
        short={k:v for k,v in row.items() if k not in ("request","gpu_after","raw")}
        short["raw_done"]=row["raw"][-1]
        short["gpu_after"]=row["gpu_after"]
        short["client_ms_per_token"]=row["client_decode_ms"]/128
        if traced:
            ds=[]
            for die in range(6):
                f=root/f"depth-{row['depth']}-pair-{row['pair']}-{row['placement']}-die-{die}.csv.gz.summary.json"
                s=load(f);rec=timeline(s)
                rec.update(die=die,op_mean_us=s["op_mean_us"],op_counts=s["op_counts"],selected_keys=s["selected_keys"],metadata=s["metadata"],source_summary_sha256=sha(f))
                ds.append(rec)
            for op in ("attnqkq","idxqk","idxscore","kvgather","attnpartn","attnmerge"):
                if sum(s["op_counts"].get(op,0) for s in ds)!=12*128:raise ValueError(f"wrong executed opcode count {op}")
            if sum(s["op_counts"].get("idxsel",0) for s in ds)!=12*128*5:raise ValueError("wrong selection pass count")
            short["die0_timeline"]=ds[0]
            short["distinct_attention_layer_work_us"]={op:sum(sum(s["op_mean_us"].get(op,{}).values()) for s in ds) for op in ("attnqkq","idxqk","idxscore","idxsel","kvgather","attnpartn","attnmerge")}
            short["client_minus_device0_us_per_token"]=1000*short["client_ms_per_token"]-ds[0]["device_us_per_token"]
            phase.append({"depth":row["depth"],"pair":row["pair"],"placement":row["placement"],"dies":ds})
        runs.append(short);groups[(row["depth"],row["placement"])].append(short)
    if status["runs"]!=len(runs):raise ValueError("capture count does not match results")
    summary=[]
    for (depth,placement),rows in sorted(groups.items()):
        if len(rows)!=5 or len(groups.get((depth,"vram" if placement=="host" else "host"),[]))!=5:
            raise ValueError("incomplete placement pairs")
        vals=[r["client_ms_per_token"] for r in rows]
        summary.append(dict(depth=depth,placement=placement,runs=len(rows),median_ms_per_token=statistics.median(vals),median_tok_s=1000/statistics.median(vals),p95_interval_ms=p95([n/1e6 for r in rows for n in r["intervals_ns"]]),per_run_ms_per_token=vals))
    result=dict(schema="flash-sass.decode-forensics.v1",tag=tag,traced=traced,manifest=manifest,summary=summary,capacity_skips=skips,runs=runs,note="Full model forward on fixed synthetic token IDs. No language/task quality claim. GPU epochs are local; do not sum complete die timelines.")
    save(out/(tag+".json.gz"),result)
    if traced:save(out/(tag+"-all-die-phase-totals.json.gz"),phase)
    save(out/(tag+"-summary.json"),summary)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--traced",required=True);p.add_argument("--untraced");p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    trace=process(a.traced,out,"traced",True)
    if a.untraced:process(a.untraced,out,"untraced",False)
    # Pair 2 is an illustrative matched pair; all five pairs remain published.
    witness={};rows={(r["depth"],r["placement"],r["pair"]):r for r in trace["runs"]}
    for place in ("host","vram"):
        s=rows[(3100,place,2)];e=rows[(8000,place,2)]
        da=s["die0_timeline"]["components_us_per_token"];db=e["die0_timeline"]["components_us_per_token"]
        witness[place]={"pair":2,"client_delta_us_per_token":1000*(e["client_ms_per_token"]-s["client_ms_per_token"]),"die0_delta_us_per_token":e["die0_timeline"]["device_us_per_token"]-s["die0_timeline"]["device_us_per_token"],"timeline_delta_us_per_token":{k:db[k]-da[k] for k in da},"distinct_attention_work_delta_us":{k:e["distinct_attention_layer_work_us"][k]-s["distinct_attention_layer_work_us"][k] for k in s["distinct_attention_layer_work_us"]}}
    save(out/"matched-phase-deltas.json",witness)
    (out/"IDENTITIES.json").write_text(json.dumps({str(x.relative_to(out)):{"bytes":x.stat().st_size,"sha256":sha(x)} for x in sorted(out.rglob("*")) if x.is_file() and x!=out/"IDENTITIES.json"},indent=2)+"\n")
    print(json.dumps({"summary":trace["summary"],"witness":witness},indent=2))
if __name__=="__main__":main()
