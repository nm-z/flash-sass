import json,pathlib,socket
base=pathlib.Path("/home/nate/claude/flash-sass-prs/issue-10")
out=base/"smoke-trace"; out.mkdir(exist_ok=True)
original="/home/nate/claude/flash-sass-next/kern"
def connect():
    s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.settimeout(180); s.connect("/tmp/flash.sock")
    return s,s.makefile("r")
def single(command):
    s,f=connect()
    try:
        s.sendall((command+"\n").encode()); line=f.readline().strip(); print(line,flush=True)
        if not line: raise RuntimeError("empty core response")
        return line
    finally: f.close(); s.close()
records=[]
try:
    records.append(single(f"reload {base}/kern/libengine.so FLASH_CUBIN={base}/kern/die_trace.cubin FLASH_TRACE=3 FLASH_ENGINE_MAXT=230000 FLASH_KV_HOST=0"))
    records.append(single("config"))
    s,f=connect()
    try:
        for cmd in ["stop","sample 0 0 0 1"]:
            s.sendall((cmd+"\n").encode()); records.append(f.readline().strip())
        s.sendall(b"chat 2 42 248046 198\n")
        while True:
            line=f.readline().strip(); records.append(line)
            if not line or line.startswith(("done ","error")): break
        if not line.startswith("done 3 "): raise RuntimeError("short trace model request failed: "+line)
        for d in range(6):
            s.sendall(f"trace {d}\n".encode()); meta=f.readline().strip()
            if not meta.startswith("trace_begin "): raise RuntimeError(meta)
            lines=[]
            while True:
                x=f.readline().strip()
                if x=="trace_end": break
                if not x or x.startswith("error"): raise RuntimeError("trace incomplete: "+x)
                lines.append(x)
            (out/f"die-{d}.csv").write_text("\n".join(lines)+"\n"); records.append(meta)
        s.sendall(b"hosttrace\n"); meta=f.readline().strip(); records.append(meta); lines=[]
        if not meta.startswith("hosttrace_begin"): raise RuntimeError(meta)
        while True:
            x=f.readline().strip()
            if x=="hosttrace_end": break
            if not x: raise RuntimeError("hosttrace incomplete")
            lines.append(x)
        (out/"host.csv").write_text("\n".join(lines)+"\n")
    finally: f.close(); s.close()
    records.append("PASS: six complete die traces and host events")
finally:
    records.append(single(f"reload {original}/libengine.so FLASH_CUBIN={original}/die.cubin FLASH_TRACE=0 FLASH_ENGINE_MAXT=230000 FLASH_KV_HOST=0"))
    (out/"protocol.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(records,indent=2),flush=True)
