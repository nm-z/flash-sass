#!/usr/bin/env python3
"""Measure fixed-length real-model decode at exact token depths through flashcore.
The generated fixture is a deterministic synthetic token prompt, not a quality test.
Does not reload a core, change GPU clocks, or stop weight holders.
"""
import argparse, csv, hashlib, json, math, os, pathlib, random, shlex, socket
import statistics, subprocess, sys, time
from collections import defaultdict

def dump(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2) + "\n")

def command(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return {"argv": args, "exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr}

def identity(path):
    p = pathlib.Path(path)
    before = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"artifact changed while hashing: {p}")
    return {"path": str(p.resolve()), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": h.hexdigest()}

def percentile(values, q):
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]

def gpu():
    return command(["nvidia-smi", "--query-gpu=index,uuid,driver_version,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current", "--format=csv"])

class Core:
    def __init__(self, path, timeout):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.settimeout(timeout)
        self.s.connect(path)
        self.f = self.s.makefile("r", encoding="utf-8")
    def send(self, text):
        self.s.sendall((text + "\n").encode())
    def line(self):
        s = self.f.readline()
        if not s:
            raise RuntimeError("core disconnected before completing the request")
        if s.startswith("error"):
            raise RuntimeError(s.strip())
        return s.rstrip("\n")
    def one(self, text):
        self.send(text)
        return self.line()
    def close(self):
        self.f.close()
        self.s.close()

def trace_summary(path, metadata):
    token = {}
    op = defaultdict(list)
    expected_first, expected_count, nph = (int(metadata[k]) for k in ("first", "count", "nph"))
    if expected_count < 2 or nph < 1 or int(metadata["ctas"]) != 1:
        raise RuntimeError("invalid trace dimensions")
    seen, previous_end = 0, 0
    with pathlib.Path(path).open() as f:
        for row in csv.DictReader(f):
            t = int(row["position"])
            if (t, int(row["phase"])) != (expected_first + seen // nph, seen % nph):
                raise RuntimeError("missing, duplicate, or out-of-order trace phase")
            times = [int(row[k]) for k in ("start_ns", "wait_start_ns", "wait_end_ns", "body_end_ns", "end_ns")]
            if times != sorted(times) or not times[0] or times[0] < previous_end:
                raise RuntimeError("nonmonotonic or empty phase trace")
            previous_end = times[-1]
            seen += 1
            v = token.setdefault(t, {"start_ns": times[0], "end_ns": times[-1], "phase_ns": 0, "ops": defaultdict(lambda: [0, 0, 0, 0])})
            v["end_ns"] = times[-1]
            v["phase_ns"] += times[-1] - times[0]
            z = v["ops"][row["opcode"]]
            for i in range(4):
                z[i] += times[i+1] - times[i]
    if seen != expected_count * nph:
        raise RuntimeError("trace record count differs from the declared dimensions")
    order = sorted(token)
    for t in order[1:]:
        for name, parts in token[t]["ops"].items():
            op[name].append(parts)
    operations = {}
    for name, rows in op.items():
        operations[name] = {label: {"median_us": statistics.median(x[i] for x in rows)/1000,
                                   "p95_us": percentile([x[i] for x in rows], .95)/1000}
                            for i, label in enumerate(("dispatch", "peer_wait", "body", "barrier_and_send"))}
    return {"tokens": len(order),
            "decode_intervals": len(order)-1,
            "decode_ms": (token[order[-1]]["end_ns"]-token[order[0]]["end_ns"])/1e6,
            "per_token": {t: {"start_ns": token[t]["start_ns"], "end_ns": token[t]["end_ns"],
                              "phase_ns": token[t]["phase_ns"],
                              "between_phase_ns": token[t]["end_ns"]-token[t]["start_ns"]-token[t]["phase_ns"]}
                          for t in order},
            "op_components": operations,
            "note": "CTA 0 samples; body is its completion time, barrier includes waiting for other CTAs. Components partition elapsed time and must not be added again as PCIe costs."}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket", default="/tmp/flash.sock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="2048,2051,2052,3100,8000,65000,262000")
    ap.add_argument("--outputs", type=int, default=129)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=18000)
    ap.add_argument("--ids-file", help="JSON array; otherwise create and save deterministic fixture IDs")
    ap.add_argument("--trace", action="store_true", help="requires matching DIE_TRACE=1 cubin and FLASH_TRACE >= outputs")
    ap.add_argument("--source-sha", required=True, help="exact checkout commit, use --dirty-patch for uncommitted source")
    ap.add_argument("--dirty-patch")
    ap.add_argument("--artifact", action="append", default=[], help="library/cubin/core/executable path to SHA-256")
    ap.add_argument("--weight", action="append", default=[], help="GGUF path to SHA-256; potentially slow")
    ap.add_argument("--weight-manifest", help="reuse prior weight hashes only when size and mtime still match")
    a = ap.parse_args()
    if a.outputs < 2 or a.repeats < 1:
        ap.error("outputs >= 2 and repeats >= 1")
    depths = [int(x) for x in a.depths.split(",")]
    if not depths or min(depths) < 1:
        ap.error("depths must be positive")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.ids_file:
        ids = json.loads(pathlib.Path(a.ids_file).read_text())
    else:
        r = random.Random(4242)
        choices = [198, 220, 11, 13, 279, 315, 323, 374, 499, 264, 311, 389]
        ids = [r.choice(choices) for _ in range(max(depths))]
    if len(ids) < max(depths) or any(not isinstance(x, int) or x < 0 or x >= 248320 for x in ids):
        ap.error("fixture must contain valid token IDs for the longest requested depth")
    dump(out/"prompt-ids.json", ids)
    weights = []
    if a.weight_manifest:
        weights = json.loads(pathlib.Path(a.weight_manifest).read_text())
        for w in weights:
            st = pathlib.Path(w["path"]).stat()
            if st.st_size != w["bytes"] or st.st_mtime_ns != w["mtime_ns"]:
                raise RuntimeError("weight manifest is stale: "+w["path"])
    weights += [identity(p) for p in a.weight]
    manifest = {"argv": sys.argv, "command": shlex.join(sys.argv), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_sha": a.source_sha, "dirty_patch": pathlib.Path(a.dirty_patch).read_text() if a.dirty_patch else None,
                "artifacts": [identity(p) for p in a.artifact], "weights": weights,
                "prompt": identity(out/"prompt-ids.json"), "gpu_before": gpu(),
                "host": command(["uname", "-a"]),
                "environment": {k:v for k,v in os.environ.items() if k.startswith(("FLASH_", "CUDA_", "SPLIT_", "ATTN_", "DIE_"))},
                "processes": command(["pgrep", "-xa", "flashcore"]),
                "mode": "real checkpoint forward, deterministic token prompt, greedy decode with stop IDs disabled"}
    core = Core(a.socket, a.timeout)
    try:
        config = core.one("config")
        manifest["config"] = config
        cfg = dict(x.split("=", 1) for x in config.split()[1:] if "=" in x)
        info = core.one("info").split()
        capacity = int(info[1])
        dies = int(cfg["dies"])
        if cfg.get("layers") != "48/48":
            raise RuntimeError("benchmark requires the full 48-layer model")
        if a.trace and int(cfg.get("trace_tokens", 0)) < a.outputs:
            raise RuntimeError("trace buffer cannot cover the requested output interval")
        dump(out/"manifest.json", manifest)
        core.one("stop")
        core.one("sample 0 0 0 1")
        results = []
        for depth in depths:
            if depth + a.outputs - 1 > capacity:
                row = {"depth": depth, "status": "capacity_skip", "capacity": capacity}
                results.append(row)
                print(json.dumps(row), flush=True)
                continue
            for repeat in range(a.repeats):
                prefix = out/f"depth-{depth}-run-{repeat}"
                core.one(f"snap {depth-1}")
                request = f"chat {a.outputs-1} " + " ".join(map(str, ids[:depth]))
                started = time.perf_counter_ns()
                core.send(request)
                generated, observed, raw = [], [], []
                while True:
                    line = core.line()
                    timestamp = time.perf_counter_ns()
                    raw.append(line)
                    if line.startswith("tok "):
                        generated.append(int(line.split()[1]))
                        observed.append(timestamp)
                    elif line.startswith("done "):
                        fields = line.split()
                        done = {"generated": int(fields[1]), "ended": int(fields[2]), "prompt": int(fields[3]),
                                "run_ms": float(fields[4]), "first_ms": float(fields[5]),
                                "resumed": int(fields[6]), "snapshot": int(fields[7])}
                        break
                    else:
                        raise RuntimeError("unexpected response: "+line)
                row = {"depth": depth, "repeat": repeat, "done": done, "generated_ids": generated,
                       "host_observed_ns": observed, "request_start_ns": started, "raw": raw,
                       "gpu_after": gpu()}
                dump(str(prefix)+".json", row)
                if len(generated) != a.outputs or done["generated"] != a.outputs or done["ended"] or done["prompt"] != depth:
                    raise RuntimeError(f"incomplete run at depth {depth}: {done}")
                if repeat and done["resumed"] != depth-1:
                    raise RuntimeError("repeat unexpectedly lost its prefix snapshot")
                dt = (observed[-1]-observed[0])/1e6
                row.update(decode_intervals=a.outputs-1, observed_decode_ms=dt,
                           observed_tok_s=(a.outputs-1)*1000/dt,
                           request_ttft_ms=(observed[0]-started)/1e6,
                           reported_decode_ms=done["run_ms"]-done["first_ms"], status="complete")
                traces = []
                if a.trace:
                    for die in range(dies):
                        core.send(f"trace {die}")
                        meta = core.line()
                        if not meta.startswith("trace_begin "):
                            raise RuntimeError("missing trace metadata: "+meta)
                        path = pathlib.Path(str(prefix)+f"-die-{die}.csv")
                        with path.open("w") as f:
                            while (line := core.line()) != "trace_end":
                                f.write(line+"\n")
                        trace_meta = dict(x.split("=", 1) for x in meta.split()[1:])
                        summary = trace_summary(path, trace_meta)
                        summary["metadata"] = meta
                        if summary["decode_intervals"] != a.outputs-1:
                            raise RuntimeError("trace did not cover the full measured interval")
                        dump(str(path)+".summary.json", summary)
                        traces.append(summary["decode_ms"])
                    row["gpu_decode_ms_by_die"] = traces
                    row["host_minus_gpu0_decode_ms"] = dt-traces[0]
                dump(str(prefix)+".json", row)
                results.append(row)
                print(json.dumps({k:row[k] for k in ("depth", "repeat", "status", "observed_tok_s", "request_ttft_ms", "reported_decode_ms")}), flush=True)
                dump(out/"results.json", results)
        summary = []
        for depth in depths:
            rows = [r for r in results if r["depth"] == depth and r["status"] == "complete"]
            if rows:
                rates = [r["observed_tok_s"] for r in rows]
                ms = [r["observed_decode_ms"]/(a.outputs-1) for r in rows]
                summary.append({"depth": depth, "runs": len(rows), "median_tok_s": statistics.median(rates),
                                "median_ms_per_token": statistics.median(ms), "p95_ms_per_token": percentile(ms, .95),
                                "cold_prefill_request_ttft_ms": rows[0]["request_ttft_ms"],
                                "cached_repeats": sum(r["done"]["resumed"] == depth-1 for r in rows)})
        dump(out/"summary.json", summary)
        print(json.dumps({"summary": summary}), flush=True)
    except Exception as exc:
        dump(out/"FAILED.json", {"error": str(exc)})
        raise
    finally:
        core.close()

if __name__ == "__main__":
    main()
