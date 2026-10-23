import subprocess;
for c in ['/home/nate/claude/flash-sass-prs/scalar-identical-prefixes.json', '/home/nate/claude/flash-sass-prs/paired-fresh-prefix32.json']:
 r=subprocess.run(["python3",'/home/nate/claude/flash-sass-prs/shadow_validation_job.py',c]);print("CONFIG",c,"EXIT",r.returncode,flush=True);r.check_returncode()
