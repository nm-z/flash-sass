import subprocess;
for c in ['/home/nate/claude/flash-sass-prs/nofuse-layers40.json', '/home/nate/claude/flash-sass-prs/nofuse-layers44.json', '/home/nate/claude/flash-sass-prs/nofuse-layers46.json']:
 r=subprocess.run(["python3",'/home/nate/claude/flash-sass-prs/shadow_validation_job.py',c]);print("CONFIG",c,"EXIT",r.returncode,flush=True);r.check_returncode()
