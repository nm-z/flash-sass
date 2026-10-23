#!/usr/bin/env python3
import hashlib,json,os,pathlib,time
base=pathlib.Path("/home/nate/claude/flash-sass-prs")
rows=[]
for part in (1,2):
    path=pathlib.Path(f"/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-{part:05d}-of-00002.gguf")
    before=path.stat(); started=time.monotonic(); h=hashlib.sha256(); done=0
    with path.open("rb") as f:
        while True:
            b=f.read(8<<20)
            if not b: break
            h.update(b); done+=len(b)
            if done % (1<<30) < len(b):
                (base/"weight-hash-progress.json").write_text(json.dumps({"pid":os.getpid(),"part":part,"bytes":done,"total":before.st_size,"elapsed_s":time.monotonic()-started})+"\n")
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError("weight file changed while hashing")
    row={"path":str(path),"bytes":before.st_size,"mtime_ns":before.st_mtime_ns,"sha256":h.hexdigest(),"elapsed_s":time.monotonic()-started}
    rows.append(row); print(json.dumps(row),flush=True)
    (base/f"weight-{part}-sha256.json").write_text(json.dumps(row,indent=2)+"\n")
(base/"weights-sha256.json").write_text(json.dumps(rows,indent=2)+"\n")
