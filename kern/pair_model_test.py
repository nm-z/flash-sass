#!/usr/bin/env python3
"""Compare isolated engine processes borrowing the existing archy IPC weights.

This never connects to the resident server. Acquire compatible GPU locks before
running it. A failed child is terminated; the holders and resident core are left
running. --chunk compares chunked prefill with the full-head scalar reference.
"""
import argparse
import hashlib
import fcntl
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("rootdir", type=Path)
ap.add_argument("outdir", type=Path)
ap.add_argument("--chunk", type=int, action="append")
ap.add_argument("--timeout", type=float, default=90)
ap.add_argument("--sha", default="unknown")
ap.add_argument("--kv-host", choices=["0", "1"], default="0")
a = ap.parse_args()
a.outdir.mkdir(parents=True, exist_ok=True)
binpath = a.rootdir.resolve()
if (binpath / "kern" / "flashcore").exists():
    binpath /= "kern"
model = Path("/mnt/sentry-nfs/flash-next-q2")
ple = model / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
argv = [str(binpath / "flashcore"), str(ple), str(binpath / "libengine.so")]
argv += [f"/tmp/hold-h{i}.manifest" for i in range(6)]
env = {k: v for k, v in os.environ.items() if not k.startswith("FLASH_")}
env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5",
           SPLIT_SHARES="112,97,81,116,97,97",
           ATTN_MAP="1,1,1,1,4,4,4,4,5,5,5,5", ATTN_DIE="1",
           FLASH_MAXT="128", FLASH_ENGINE_MAXT="128", FLASH_MAXIMG="0",
           FLASH_KV_HOST=a.kv_host, FLASH_PLE_DISK="1", FLASH_PLE_OFF="192",
           FLASH_NBLK="4", FLASH_WATCHDOG_S="15", FLASH_ANCHOR_DIE="0",
           FLASH_CUBIN=str(binpath / "die.cubin"), FLASH_LAYERS="48")

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

report = {"hostname": socket.gethostname(), "head_sha": a.sha, "argv": argv,
          "cwd": str(binpath), "environment": {
              k: v for k, v in env.items()
              if k.startswith(("FLASH_", "CUDA_", "ATTN_", "SPLIT_"))},
          "sha256": {f: digest(binpath / f)
                     for f in ["flashcore", "libengine.so", "die.cubin"]},
          "runs": {}, "comparisons": []}

class Core:
    def __init__(self, label, pair, chunk):
        self.label, self.q = label, queue.Queue()
        self.end = time.monotonic() + a.timeout
        self.stdout = open(a.outdir / f"{label}.stdout.log", "w", buffering=1)
        self.stderr = open(a.outdir / f"{label}.stderr.log", "w", buffering=1)
        self.commands = open(a.outdir / f"{label}.commands.log", "w", buffering=1)
        settings = dict(env, FLASH_TWO_POSITION=str(pair), FLASH_PREFILL_CHUNK=str(chunk))
        self.p = subprocess.Popen(argv, cwd=binpath, env=settings, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)
        def pump():
            try:
                for line in self.p.stdout:
                    self.stdout.write(line)
                    self.q.put(line.rstrip("\n"))
            finally:
                self.q.put(None)
        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()

    def until(self, pred):
        lines = []
        while True:
            remaining = self.end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.label}: command deadline {a.timeout}s")
            try:
                line = self.q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"{self.label}: waiting for output; see stderr log")
            if line is None:
                raise RuntimeError(f"{self.label}: EOF exit={self.p.poll()}; see stderr log")
            lines.append(line)
            if line.startswith("error:"):
                raise RuntimeError(f"{self.label}: {line}")
            if pred(line):
                return lines

    def command(self, text, pred):
        self.end = time.monotonic() + a.timeout
        self.commands.write(text + "\n")
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()
        return self.until(pred)

    def ack(self, text):
        return self.command(text, lambda l: l.startswith("ok:"))

    def case(self, label, ids, nnew=1):
        cmd = f"gen {nnew} " + " ".join(map(str, ids))
        t0 = time.monotonic()
        lines = self.command(cmd, lambda l: l.startswith("time:"))
        wall = time.monotonic() - t0
        toks = [l for l in lines if l.startswith("tokens:")]
        if len(toks) != 1:
            raise RuntimeError(f"{self.label}/{label}: expected one tokens line: {lines}")
        tokens = list(map(int, toks[0].split(":", 1)[1].split()))
        if not tokens:
            raise RuntimeError(f"{self.label}/{label}: empty generated output")
        state = self.command("statehash", lambda l: l.startswith("statehash done"))
        summary = next((l for l in state if "positions" in l), "")
        m = re.search(r"\bpositions=(\d+)", summary)
        if not m:
            raise RuntimeError(f"{self.label}: missing statehash summary: {state[:3]}")
        body = [l for l in state if l != summary and not l.startswith("statehash done")]
        if not body:
            raise RuntimeError(f"{self.label}: empty statehash body")
        print(f"{self.label} {label}: tokens={tokens} {lines[-1]}", flush=True)
        return {"command": cmd, "tokens": tokens, "positions": int(m.group(1)),
                "state_summary": summary, "state": body, "time": lines[-1], "wall_s": wall}

    def close(self):
        if self.p.poll() is None:
            try:
                self.commands.write("quit\n")
                self.p.stdin.write("quit\n")
                self.p.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.terminate()
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait(timeout=5)
        self.thread.join(timeout=2)
        self.stdout.close()
        self.stderr.close()
        self.commands.close()
        return self.p.returncode

def prompt(n):
    return [100 + i for i in range(n)]

def real_prompt():
    path = a.outdir / "prompt.txt"
    path.write_text("<|im_start|>user\nExplain why a GPU needs a memory barrier.\n<|im_end|>\n<|im_start|>assistant\n")
    shard = model / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
    p = subprocess.run([str(binpath / "tok_test"), str(shard), str(path)],
                       check=True, text=True, capture_output=True, timeout=30)
    ids = json.loads(p.stdout)
    if not 1 < len(ids) < 120:
        raise RuntimeError(f"Unexpected real prompt token count: {len(ids)}")
    report["real_prompt_ids"] = ids
    return ids

def run(label, pair, chunk, text_ids, stop_token=None):
    c = Core(label, pair, chunk)
    result = {"pair": pair, "chunk": chunk, "cases": {}}
    report["runs"][label] = result
    try:
        c.until(lambda l: l == "ready")
        result["config"] = c.command("config", lambda l: l.startswith("config "))[-1]
        c.ack("stop")
        c.ack("sample 0 0 1 1")
        for n in [1, 2, 3, 5, 17, 33]:
            result["cases"][f"p{n}"] = c.case(f"p{n}", prompt(n))
        result["cases"]["real_prompt"] = c.case("real_prompt", text_ids, 3)
        c.ack("sample 1234567 0.7 8 0.9")
        result["cases"]["sampled"] = c.case("sampled", prompt(17), 4)
        c.ack("sample 0 0 1 1")
        c.ack("snap 5")
        result["cases"]["snapshot"] = c.case("snapshot", prompt(17))
        result["cases"]["resume"] = c.case("resume", prompt(19))
        if "5 from the snapshot" not in result["cases"]["resume"]["time"]:
            raise RuntimeError(f"{label}: snapshot was not reused")
        if stop_token is None:
            stop_token = result["cases"]["p1"]["tokens"][0]
        c.ack(f"stop {stop_token}")
        result["cases"]["stop"] = c.case("stop", prompt(1), 3)
        if len(result["cases"]["stop"]["tokens"]) != 1:
            raise RuntimeError(f"{label}: stop token did not stop generation")
        c.ack("stop")
        result["stop_token"] = stop_token
    finally:
        result["exit_code"] = c.close()
    if result["exit_code"] != 0:
        raise RuntimeError(f"{label}: core exit {result['exit_code']}")
    return result

# IPC weight holders use shared locks. Keep compatible shared locks for the
# whole comparison so exclusive GPU jobs cannot begin during this borrower.
locks = []
try:
    gpu_ids = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    for line in gpu_ids.splitlines():
        index, uuid = [x.strip() for x in line.split(",")]
        if int(index) < 6:
            lock = open(Path.home() / "codex/asm-port" / (uuid + ".lock"), "a")
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            locks.append(lock)
    if len(locks) != 6:
        raise RuntimeError("Need all six holder GPU locks")
    text_ids = real_prompt()
    base = run("scalar", 0, 0, text_ids)
    modes = [("paired", 1, 0)] if not a.chunk else [(f"chunk{n}", 1, n) for n in a.chunk]
    for label, pair, chunk in modes:
        test = run(label, pair, chunk, text_ids, base["stop_token"])
        for name, ref in base["cases"].items():
            got = test["cases"][name]
            fields = {k: got[k] == ref[k] for k in ["tokens", "positions", "state"]}
            comparison = {"mode": label, "case": name, **fields, "pass": all(fields.values())}
            report["comparisons"].append(comparison)
            if not fields["state"]:
                print("STATE MISMATCH", label, name,
                      [(x, y) for x, y in zip(ref["state"], got["state"]) if x != y][:6], flush=True)
        if chunk != 1 and not any(re.search(r"paired_positions=[1-9]", r["state_summary"])
                                  for r in test["cases"].values()):
            raise RuntimeError(f"{label}: no reported paired execution")
    report["pass"] = all(x["pass"] for x in report["comparisons"])
except Exception as e:
    report["pass"] = False
    report["error"] = repr(e)
(a.outdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"pass": report["pass"], "error": report.get("error"),
                  "comparisons": report["comparisons"]}, indent=2), flush=True)
raise SystemExit(0 if report["pass"] else 1)
