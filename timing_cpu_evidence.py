from pathlib import Path
import json,subprocess,hashlib,time
base=Path('/home/nate/claude/flash-sass-prs');src=base/'timing-build-c3ea695/kern';out=base/'timing-validation-c3ea695/cpu-c3ea695';out.mkdir(exist_ok=False)
cmds=[[str(src/'timings_test')],[str(src/'trace_session_test')],['perl',str(src/'flashd_timing_test.pl'),str(src/'flashd_timing_test'),str(src/'timings_test'),'/tmp/timing-http-c3ea695']]
rows=[]
for i,cmd in enumerate(cmds):
 t=time.monotonic()
 with (out/f'command{i}.stdout').open('w') as f,(out/f'command{i}.stderr').open('w') as g:r=subprocess.run(cmd,cwd=src,stdout=f,stderr=g,timeout=240)
 rows.append(dict(argv=cmd,exit=r.returncode,seconds=time.monotonic()-t))
 (out/'results.json').write_text(json.dumps(dict(status='PASS' if len(rows)==3 and all(x['exit']==0 for x in rows) else 'RUNNING_OR_FAILED',source='c3ea6950b91382d40a8d74fea46675594d6ded3a',commands=rows,files={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (src/'timings_test',src/'trace_session_test',src/'flashd_timing_test',src/'flashd_timing_test.pl')}),indent=2)+'\n')
 print(rows[-1],flush=True);r.check_returncode()
import shutil
shutil.copytree('/tmp/timing-http-c3ea695',out/'http-responses',ignore=shutil.ignore_patterns('engine.sock'))
print('PASS timing CPU and 18 HTTP fixture cases')
