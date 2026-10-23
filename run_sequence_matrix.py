import subprocess,json;from pathlib import Path;previous=json.loads(Path('/home/nate/claude/flash-sass-prs/trace-validation-7262065/master-paired-healthy-probe/results.json').read_text())["cores"][0]["pid"];assert not Path("/proc/"+str(previous)).exists();
for c in ['/home/nate/claude/flash-sass-prs/slots2-kv0-q1-b1.json', '/home/nate/claude/flash-sass-prs/slots2-kv1-q8-b512.json']:
 r=subprocess.run(["python3",'/home/nate/claude/flash-sass-prs/shadow_validation_job.py',c]);print("CONFIG",c,"EXIT",r.returncode,flush=True);r.check_returncode()
