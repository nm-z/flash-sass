#!/usr/bin/env python3
"""Invalid CLI cases must exit2 before cuInit or cuLaunchKernel."""
import argparse,hashlib,json,os,pathlib,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
binaries={k:str(pathlib.Path(k+'_test').resolve()) for k in ('qsa','attn')}
qcases=[[],['unknown'],['embimg','extra'],['kv','1','512','extra'],['idxsel','0','512'],['idxsel','700','512','0','6'],['idxscore','65537'],['idxfull','0'],['gathertime','2']]
bad=['0','-1','+1','',' 1','1 ','1x','1.0','1e3','0x10','262145','4294967295','9999999999999999999999']
qcases += [['kv',x,'512'] for x in bad]+[['kv','1',x] for x in ['0','-1','4097']]
acases=[[x] for x in bad]+[['1',x] for x in ['-1','+1','1x','262145']]+[['1','0','extra']]
rows=[]
with tempfile.TemporaryDirectory() as d:
 d=pathlib.Path(d);src=d/'guard.c';so=d/'guard.so'
 src.write_text('#include <unistd.h>\nint cuInit(unsigned x){(void)x;_exit(99);}\nint cuLaunchKernel(){_exit(99);}\n')
 subprocess.run(['cc','-shared','-fPIC',str(src),'-o',str(so)],check=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',LD_PRELOAD=str(so),DIE_CUBIN='/__must_not_load__.cubin')
 for name,cases in [('qsa',qcases),('attn',acases)]:
  for case in cases:
   r=subprocess.run([binaries[name],*case],env=env,capture_output=True,text=True,timeout=10)
   rows.append(dict(binary=name,args=case,exit=r.returncode,stdout=r.stdout,stderr=r.stderr,passed=r.returncode==2))
report=dict(binaries={k:dict(path=v,sha256=hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest()) for k,v in binaries.items()},guard='CUDA entry points exit99',cases=rows,status='PASS' if all(r['passed'] for r in rows) else 'FAIL')
(out/'cli-report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],len(rows),'invalid-argument cases');raise SystemExit(0 if report['status']=='PASS' else 1)
