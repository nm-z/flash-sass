#!/usr/bin/env python3
"""Run reproducible synthetic-history QSA component checks on one reserved GPU."""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def save(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',default='./qsa_direct_test');p.add_argument('--out',required=True)
    p.add_argument('--quick',action='store_true');p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    binary=Path(a.binary).resolve();cubin=Path(os.environ.get('DIE_CUBIN','die.cubin')).resolve()
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('QSA_'):env.pop(key)
    env['DIE_CUBIN']=str(cubin)
    cases=[]
    lengths=[1,2047,2052,3100] if a.quick else [1,2047,2051,2055,3100,8000,65000,262000,262144]
    for host in (0,1):
        for visible in lengths:
            cases.append((visible,512,min(4,262145-visible),4242,host,'hash',False,None))
        cases.append((65,8,4,4242,host,'hash',True,None))
        for pattern in ('oldest','newest','clustered','scattered','equal','adjacent'):
            cases.append((3100,512,4,4242,host,pattern,False,None))
    if not a.quick:
        for seed in (0,77):
            for visible in (2052,8000,65000):cases.append((visible,512,4,seed,0,'hash',False,None))
    for injection in ('selection','gather','output'):
        cases.append((2052,512,1,4242,0,'hash',False,injection))
    info={'schema':'flash-sass.qsa-direct-matrix.v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary':str(binary),'binary_sha256':digest(binary),'cubin':str(cubin),'cubin_sha256':digest(cubin),'CUDA_VISIBLE_DEVICES':env.get('CUDA_VISIBLE_DEVICES'),'scope':'synthetic-history component parity, not whole-model or checkpoint quality','cases':[],'passed':False}
    smi=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,temperature.gpu,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True)
    (out/'gpu-before.csv').write_text(smi.stdout+smi.stderr);save(out/'results.json',info)
    for i,(visible,budget,steps,seed,host,pattern,replay,injection) in enumerate(cases):
        tag=f'{i:03d}-n{visible}-b{budget}-s{seed}-h{host}-{pattern}'+('-replay' if replay else '')+('-inject-'+injection if injection else '')
        runenv=env|{'QSA_DIRECT_SEED':str(seed),'QSA_HOSTKV':str(host),'QSA_PATTERN':pattern,'QSA_REPLAY_PREFIX':str(int(replay))}
        if injection:runenv['QSA_DIRECT_INJECT']=injection
        cmd=[str(binary),str(visible),str(budget),str(steps)];started=time.monotonic()
        try:
            r=subprocess.run(cmd,env=runenv,capture_output=True,text=True,timeout=a.timeout)
            stdout,stderr,rc=r.stdout,r.stderr,r.returncode
        except subprocess.TimeoutExpired as exc:
            stdout=exc.stdout or b'';stderr=exc.stderr or b''
            if isinstance(stdout,bytes):stdout=stdout.decode(errors='replace')
            if isinstance(stderr,bytes):stderr=stderr.decode(errors='replace')
            stderr+='\nRUNNER TIMEOUT\n';rc=124
        log=out/(tag+'.log');log.write_text(stdout+stderr)
        rows=re.findall(r'^visible=(\d+) complete=(\d+) tail=(\d+) selected=(\d+) keys=(\d+) guarded_allocations=(\d+) exclusion=(yes|no) (EXACT|FAIL)$',stdout,re.M)
        passed=(rc==(3 if injection else 0) and len(rows)==steps)
        for j,row in enumerate(rows):
            n,complete,tail,selected,keys,guards=map(int,row[:6]);excluded,status=row[6:]
            passed=passed and n==visible+j and complete==n//4 and tail==n%4 and selected==min(n//4,budget) and keys==4*selected+tail and guards>0 and (excluded=='yes')==(n//4>budget) and status==('FAIL' if injection else 'EXACT')
        record={'case':tag,'command':cmd,'environment':{k:v for k,v in runenv.items() if k.startswith('QSA_')},'exit_code':rc,'expected_exit_code':3 if injection else 0,'seconds':time.monotonic()-started,'parsed_positions':len(rows),'log_sha256':digest(log),'passed':bool(passed)}
        info['cases'].append(record);save(out/'results.json',info);print(tag,'PASS' if passed else 'FAIL',flush=True)
        if not passed:return 1
    info['passed']=True;info['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(out/'results.json',info)
    return 0
if __name__=='__main__':raise SystemExit(main())
