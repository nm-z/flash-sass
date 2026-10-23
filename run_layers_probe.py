import subprocess,json;from pathlib import Path;previous=json.loads(Path('/home/nate/claude/flash-sass-prs/ple-diagnostic-nofuse-validation-6835416/nofuse-probe/results.json').read_text())["cores"][0]["pid"];assert not Path("/proc/"+str(previous)).exists();
for c in ['/home/nate/claude/flash-sass-prs/nofuse-layers1.json', '/home/nate/claude/flash-sass-prs/nofuse-layers2.json', '/home/nate/claude/flash-sass-prs/nofuse-layers4.json']:
 r=subprocess.run(["python3",'/home/nate/claude/flash-sass-prs/shadow_validation_job.py',c]);print("CONFIG",c,"EXIT",r.returncode,flush=True);r.check_returncode()
