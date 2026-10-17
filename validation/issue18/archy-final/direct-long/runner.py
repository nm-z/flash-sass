import datetime, fcntl, hashlib, json, os, pathlib, re, subprocess, time
P=pathlib.Path
out=P('/tmp/flash-resolve-issue18-final-20261017/evidence/direct-long');out.mkdir(parents=True,exist_ok=True)
binary=P('/tmp/flash-resolve-issue17-20261017/kern/qsa_direct_test')
cubin=P('/tmp/flash-resolve-issue18-final-20261017/kern/die.cubin')
uuid='GPU-431371bd-b482-96b2-4b70-04f83c6d075b'
lock=open('/home/nate/codex/asm-port/'+uuid+'.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
j={'binary':str(binary),'binary_sha256':sha(binary),'driver_source_revision':'eee917c5adc919ef7689ecfec7dd8f6390e09ffe','cubin':str(cubin),'cubin_sha256':sha(cubin),'kernel_source_revision':'58508ed3d4c544454ff54911d24773e0a924d624','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Synthetic-history complete QSA chain with allocation canaries; not a whole-model test','cases':[]}
for host in (0,1):
    for n in (3100,8000,65000,262000,262144):
        steps=min(4,262145-n);tag=f'h{host}-n{n}'
        e={k:v for k,v in os.environ.items() if not k.startswith(('QSA_','ATTN_'))}
        extra={'CUDA_VISIBLE_DEVICES':uuid,'DIE_CUBIN':str(cubin),'QSA_HOSTKV':str(host),'QSA_PATTERN':'hash','QSA_DIRECT_SEED':'4242'}
        e.update(extra);cmd=[str(binary),str(n),'512',str(steps)];start=time.monotonic()
        r=subprocess.run(cmd,cwd=binary.parent,env=e,text=True,capture_output=True,timeout=180)
        (out/(tag+'.log')).write_text(r.stdout+r.stderr)
        rows=re.findall(r'^visible=(\d+) complete=(\d+) tail=(\d+) selected=(\d+) keys=(\d+) guarded_allocations=(\d+) exclusion=(yes|no) (EXACT|FAIL)$',r.stdout,re.M)
        ok=r.returncode==0 and len(rows)==steps and all(int(row[0])==n+i and row[-1]=='EXACT' and int(row[5])>0 for i,row in enumerate(rows))
        j['cases'].append({'name':tag,'command':cmd,'environment':extra,'exit_code':r.returncode,'seconds':time.monotonic()-start,'positions':len(rows),'passed':ok})
        (out/'results.json').write_text(json.dumps(j,indent=2)+'\n');print(tag,ok,flush=True)
        if not ok:raise SystemExit(1)
j['passed']=True;j['positions']=sum(x['positions'] for x in j['cases'])
(out/'results.json').write_text(json.dumps(j,indent=2)+'\n')
