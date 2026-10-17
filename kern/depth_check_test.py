#!/usr/bin/env python3
"""Offline HTTP fixtures for depth.sh; no model or GPU required."""
import copy, http.server, json, os, pathlib, re, socket, subprocess, tempfile, threading
ROOT=pathlib.Path(__file__).resolve().parent

def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix="depth-check-test-") as tmp:
        base=pathlib.Path(tmp)
        class Handler(http.server.BaseHTTPRequestHandler):
            mode="success"
            def log_message(self,*args): pass
            def do_POST(self):
                raw=self.rfile.read(int(self.headers["Content-Length"]))
                request=json.loads(raw)
                code=re.search(r"vault is ([0-9a-f]{6})",request["messages"][0]["content"]).group(1)
                body={"choices":[{"message":{"content":code+"\n1 2 3"}}],
                      "timings":{"prompt_n":100,"prompt_ms":2000.0,"prompt_per_second":50.0,
                                 "predicted_n":11,"predicted_ms":400.0,"predicted_per_second":25.0,
                                 "decode_intervals":10,"decode_ms":400.0}}
                status=200
                if self.mode=="wrong": body["choices"][0]["message"]["content"]="wrong code"
                elif self.mode=="missing-answer": del body["choices"]
                elif self.mode=="api-error": body={"error":{"message":"fixture error"}}
                elif self.mode=="missing-timings": del body["timings"]
                elif self.mode=="bad-rate": body["timings"]["predicted_per_second"]=123.0
                elif self.mode=="http-error": status=503; body={"error":"unavailable"}
                payload=b"{broken" if self.mode=="malformed" else json.dumps(body).encode()
                self.send_response(status); self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(payload))); self.end_headers(); self.wfile.write(payload)
        server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        url=f"http://127.0.0.1:{server.server_port}"
        try:
            for mode,wanted in [("success",0),("wrong",1),("missing-answer",2),("api-error",2),
                                ("missing-timings",2),("bad-rate",2),("malformed",2),("http-error",2)]:
                Handler.mode=mode; out=base/mode
                r=subprocess.run(["bash",str(ROOT/"depth.sh"),"2","50",url],capture_output=True,text=True,
                                 env=dict(os.environ,DEPTH_OUT=str(out),DEPTH_TIMEOUT="3"),timeout=10)
                assert r.returncode==wanted,(mode,r.returncode,r.stdout,r.stderr)
                assert (out/"request.json").is_file() and (out/"response.json").is_file()
                if wanted==0:
                    assert "prompt 100 tok" in r.stdout and "10 intervals in 400.0 ms" in r.stdout and "PASS" in r.stdout
                if wanted==1: assert "FAIL" in r.stdout and "PASS" not in r.stdout
                row={"case":mode,"exit":r.returncode,"expected":wanted,"stdout":r.stdout.strip()}
                results.append(row); print(json.dumps(row),flush=True)
        finally:
            server.shutdown(); server.server_close(); thread.join()
        # A closed, locally allocated port exercises curl's transport failure.
        r=subprocess.run(["bash",str(ROOT/"depth.sh"),"2","50",url],capture_output=True,text=True,
                         env=dict(os.environ,DEPTH_OUT=str(base/"transport"),DEPTH_TIMEOUT="1"),timeout=5)
        assert r.returncode==2 and "transport failed" in r.stderr,(r.returncode,r.stderr)
        print(json.dumps({"case":"transport","exit":r.returncode,"expected":2}),flush=True)
    print("PASS: nine response/HTTP/transport cases")
if __name__=="__main__": main()
