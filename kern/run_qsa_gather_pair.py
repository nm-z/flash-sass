#!/usr/bin/env python3
import argparse,csv,hashlib,json,os,random,struct,subprocess,sys
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def put(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',default='./qsa_gather_bench');p.add_argument('--out',required=True);p.add_argument('--nv',type=int,default=5000);p.add_argument('--iterations',type=int,default=32);p.add_argument('--seed',type=int,default=4242);p.add_argument('--matrix',action='store_true');a=p.parse_args()
    if not 1<=a.nv<=65536 or not 1<=a.iterations<=10000 or not 0<=a.seed<=0xffffffff:p.error('invalid nv/iterations/seed')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);binary=Path(a.binary).resolve();here=Path(__file__).resolve().parent;cubin=Path(os.environ.get('DIE_CUBIN','die.cubin')).resolve()
    identities={str(f):sha(f) for f in [binary,cubin,here/'qsa_gather_bench.c',here/'qsa_test.c',here/'../ref/qsa.c',here/'../ref/attn.c',here/'../ref/fmath.c']}
    smi=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.free','--format=csv'],capture_output=True,text=True)
    (out/'nvidia-smi.txt').write_text(smi.stdout+smi.stderr)
    cases=[(a.nv,1 if 4*a.nv+1<=262144 else 0,'changing')]
    if a.matrix:cases=[(nv,t,pat) for nv in sorted(set([128,a.nv])) for t in range(4) for pat in ['clustered','even','random','changing','captured'] if 4*nv+t<=262144]
    report={'identities_sha256':identities,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'seed':a.seed,'iterations':a.iterations,'hardware_counter_gate':'not_run','true_dram_bandwidth_validated':False,'pairs':[]}
    put(out/'provenance.json',report)
    for nv,tail,pat in cases:
        k=min(nv,512);tag=f'nv{nv}-tail{tail}-{pat}';env=os.environ.copy();env['DIE_CUBIN']=str(cubin);env.pop('GATHER_IDS',None)
        if pat=='captured':
            capture=out/(tag+'.ids.u32');rng=random.Random(a.seed)
            with capture.open('xb') as f:
                for _ in range(a.iterations):
                    row=sorted(rng.sample(range(nv),k));f.write(struct.pack('<'+'I'*k,*row))
            env['GATHER_IDS']=str(capture.resolve())
        arms={};rows={}
        for host in [0,1]:
            raw=out/(tag+f'-host{host}.csv');cmd=[str(binary),str(nv),str(tail),str(a.iterations),pat,str(host),str(raw.resolve()),str(a.seed)]
            proc=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=180);(out/(tag+f'-host{host}.log')).write_text(proc.stdout+proc.stderr)
            if proc.returncode:raise RuntimeError(f'benchmark failed exit={proc.returncode}: {cmd}; see log')
            result=json.loads(proc.stdout)
            if not result['preflight_verified_all']:raise RuntimeError('missing preflight verification')
            with raw.open() as f:rows[host]=list(csv.DictReader(f))
            if len(rows[host])!=a.iterations:raise RuntimeError('missing raw timing rows')
            arms[host]={'command':cmd,'result':result,'raw_csv_sha256':sha(raw)}
        for field in ['cache_fnv1a64','selections_fnv1a64','logical_keys','physical_gathered_slots','selection_changes','verified_output_signature_changes']:
            if arms[0]['result'][field]!=arms[1]['result'][field]:raise RuntimeError(f'host/VRAM input or output identity differs: {field}')
        for x,y in zip(rows[0],rows[1]):
            for field in ['iteration','selection_fnv1a64','verified_output_fnv1a64']:
                if x[field]!=y[field]:raise RuntimeError(f'raw pair mismatch: {field}')
        den=arms[0]['result']['gpu_us_p50'];pair={'case':tag,'arms':arms,'host_to_vram_p50_ratio':arms[1]['result']['gpu_us_p50']/den if den else None}
        if pat=='captured':pair['captured_ids_sha256']=sha(out/(tag+'.ids.u32'));pair['captured_ids_origin']='generated fixture, not model capture'
        report['pairs'].append(pair);put(out/'report.json',report);print(tag,'verified',pair['host_to_vram_p50_ratio'],flush=True)
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,KeyError,subprocess.TimeoutExpired) as exc:print(str(exc),file=sys.stderr);raise SystemExit(1)
