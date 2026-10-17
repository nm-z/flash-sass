#!/usr/bin/env python3
import argparse, datetime, fcntl, hashlib, json, os, pathlib, statistics, subprocess, time
P=pathlib.Path
p=argparse.ArgumentParser()
p.add_argument('mode',choices=['padding','quant'])
p.add_argument('--out',required=True)
a=p.parse_args()
out=P(a.out);out.mkdir(parents=True,exist_ok=True)
uuid='GPU-431371bd-b482-96b2-4b70-04f83c6d075b'
parent=P('/tmp/flash-resolve-issue16-20261017/kern')
candidate=P('/tmp/flash-resolve-'+('issue18-final' if a.mode=='padding' else 'issue19')+'-20261017/kern')
def sha(f):return hashlib.sha256(P(f).read_bytes()).hexdigest()
def record(cmd,cwd,env,name,expect=0,timeout=60):
    start=time.monotonic();merged=os.environ.copy()
    for k in list(merged):
        if k.startswith(('QSA_','ATTN_')) or k=='DIE_CUBIN':merged.pop(k)
    merged.update(env);merged['CUDA_VISIBLE_DEVICES']=uuid
    with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
        try:rc=subprocess.run(cmd,cwd=cwd,env=merged,stdout=so,stderr=se,timeout=timeout).returncode
        except subprocess.TimeoutExpired:rc=124
    row={'name':name,'command':list(map(str,cmd)),'cwd':str(cwd),'env':dict(env,CUDA_VISIBLE_DEVICES=uuid),'exit_code':rc,'expected_exit':expect,'seconds':time.monotonic()-start}
    row['passes']=expect is None or rc==expect
    if cmd and P(cmd[0]).is_file():row['executable_sha256']=sha(cmd[0])
    for leaf in ('attnpart.cubin','attnmerge.cubin','die.cubin'):
        f=P(env.get('DIE_CUBIN',str(P(cwd)/leaf))) if leaf=='die.cubin' else P(cwd)/leaf
        if f.is_file():row[leaf+'_sha256']=sha(f)
    rows.append(row);(out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(name,rc,'PASS' if row['passes'] else 'FAIL',flush=True)
    return row
rows=[];manifest={'mode':a.mode,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':os.uname().nodename,'gpu_uuid':uuid,'candidate_build':json.loads((candidate.parent/'evidence/build.json').read_text()),'parent_build':json.loads((parent.parent/'evidence/build.json').read_text()),'script_sha256':sha(__file__),'results':rows}
if a.mode=='padding':
    r=record(['make','CUDA=/opt/cuda','attnpart.cubin','attnmerge.cubin'],parent,{},'parent-build',0,90)
    if not r['passes']:raise SystemExit(1)
lock=open('/home/nate/codex/asm-port/'+uuid+'.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
record(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm','--format=csv,noheader'],candidate,{},'gpu-before',None,15)
if a.mode=='quant':
    record([str(candidate/'quant_range_test')],candidate,{},'cpu-range')
    for host in (False,True):
        e={'QSA_HOSTKV':'1'} if host else {};suffix='host' if host else 'vram'
        record([str(candidate/'qsa_test'),'kv','16','2'],candidate,dict(e,QSA_KV_EXTREMES='1',DIE_CUBIN=str(parent/'die.cubin')),'parent-extremes-'+suffix,1)
        record([str(candidate/'qsa_test'),'kv','16','2'],candidate,dict(e,QSA_KV_EXTREMES='1'),'fixed-extremes-'+suffix)
        record([str(candidate/'qsa_test'),'kv','16','2'],candidate,e,'fixed-normal-'+suffix)
else:
    for part in ('K','V'):
        # K and V are controlled separately. A passing parent is recorded, not invented as a failure.
        e={'ATTN_POISON_K':'off','ATTN_POISON_V':'off'};e['ATTN_POISON_'+part]='nan'
        record([str(candidate/'attn_test'),'1','0'],parent,e,'parent-dense-'+part,3 if part=='V' else None)
        e={'QSA_POISON_K':'off','QSA_POISON_V':'off','DIE_CUBIN':str(parent/'die.cubin')};e['QSA_POISON_'+part]='nan'
        record([str(candidate/'qsa_test'),'kv','13','2'],candidate,e,'parent-sparse-'+part,1 if part=='V' else None)
    sizes=list(range(1,33))+[33,65,1023,1024,1025]+list(range(2048,2056))+[8191,8192,8193]
    for n in sizes:
        for k,v in [('nan','off'),('off','nan'),('inf','-inf'),('max','max')]:
            record([str(candidate/'attn_test'),str(n),'0'],candidate,{'ATTN_POISON_K':k,'ATTN_POISON_V':v},f'dense-{n}-{k}-{v}')
    for host in (False,True):
        suffix='host' if host else 'vram';e={'QSA_HOSTKV':'1'} if host else {}
        for budget in (1,2,8):
            for n in (1,4*budget+1,4*budget+2,4*budget+3,4*budget+4,4*budget+5):
                for k,v in [('nan','off'),('off','nan'),('inf','-inf'),('max','max')]:
                    record([str(candidate/'qsa_test'),'kv',str(n),str(budget)],candidate,dict(e,QSA_POISON_K=k,QSA_POISON_V=v),f'sparse-{suffix}-{budget}-{n}-{k}-{v}')
        for n in (2048,2051,2052,2055):
            record([str(candidate/'qsa_test'),'kv',str(n),'512'],candidate,dict(e,QSA_POISON_PADDING='nan',QSA_KV_LAST_ONLY='1'),f'sparse-production-{suffix}-{n}',0,120)
    # Fixed valid inputs, same standalone partition/merge shape; launch-loop component timing, not model throughput.
    for repeat in range(5):
        for label,cwd in ([('parent',parent),('fixed',candidate)] if repeat%2==0 else [('fixed',candidate),('parent',parent)]):
            record([str(candidate/'attn_test'),'2051','200'],cwd,{},f'timing-{repeat}-{label}')
record(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm','--format=csv,noheader'],candidate,{},'gpu-after',None,15)
manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
manifest['status']='PASS' if all(x['passes'] for x in rows) else 'FAIL'
manifest['cases']=len(rows);manifest['failed']=[x['name'] for x in rows if not x['passes']]
(out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('status','cases','failed')},indent=2),flush=True)
raise SystemExit(0 if manifest['status']=='PASS' else 1)
