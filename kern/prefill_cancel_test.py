#!/usr/bin/env python3
"""Cancel an owned socket core's prefill request and verify its next response.

Configured for archy's existing IPC holders. The reference must be a successful
full-model comparison's results.json at the same source revision and binaries;
see PAIR_TESTING.md for its scalar-reference case and state comparison protocol.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import select
import socket
import subprocess
import tempfile
import time
from pathlib import Path


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def fields(line):
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def read_command(client, command, terminator, records):
    entry = {"command": command, "response": []}
    records.append(entry)
    client.sendall((command + "\n").encode())
    deadline = time.monotonic() + 90
    with client.makefile("rb") as stream:
        for _ in range(10000):
            left = deadline - time.monotonic()
            check(left > 0, "response deadline exceeded")
            client.settimeout(left)
            raw = stream.readline(1024 * 1024)
            check(raw and len(raw) < 1024 * 1024, "EOF or oversized response")
            line = raw.decode().rstrip("\r\n")
            entry["response"].append(line)
            check(not line.startswith("error:"), line)
            if terminator(line):
                return entry["response"]
    raise RuntimeError("response line limit exceeded")


def connect(path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(90)
    try:
        client.connect(path)
    except Exception:
        client.close()
        raise
    return client


def exchange(path, command, terminator, records):
    with connect(path) as client:
        return read_command(client, command, terminator, records)


def state(lines):
    summaries = [x for x in lines if x.startswith("statehash positions=")]
    check(len(summaries) == 1, "missing or duplicate statehash summary")
    result = {k: int(v) for k, v in fields(summaries[0]).items()}
    check("positions" in result and "paired_positions" in result, "missing counters")
    result["state"] = [x for x in lines
                       if x != summaries[0] and x != "statehash done"]
    check(result["state"], "empty statehash body")
    return result


def stop_child(child, path):
    if child.poll() is None:
        try:
            with connect(path) as client:
                client.sendall(b"quit\n")
            child.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    child.stdout.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rootdir", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--kv-host", choices=["0", "1"], default="0")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    root = a.rootdir.resolve()
    binpath = root / "kern" if (root / "kern/flashcore").is_file() else root
    report = {"hostname": socket.gethostname(), "head_sha": a.sha, "pass": False,
              "commands": [], "reference": str(a.reference.resolve())}
    child = None
    try:
        check(re.fullmatch(r"[0-9a-f]{40}", a.sha), "supply a full commit SHA")
        ref = json.loads(a.reference.read_text())
        check(ref["head_sha"] == a.sha and ref.get("pass"), "reference revision/pass mismatch")
        scalar = ref["runs"]["scalar"]
        expected = {"rev": a.sha, "kv_host": a.kv_host, "ple_disk": "1",
                    "maxt": "128", "layers": "48/48", "nblk": "4", "dies": "6"}
        refcfg = fields(scalar["config"])
        check(all(refcfg.get(k) == v for k, v in expected.items()), "reference config mismatch")
        oracle = scalar["cases"]["p17"]
        report["sha256"] = {f: digest(binpath / f)
                           for f in ["flashcore", "libengine.so", "die.cubin"]}
        check(report["sha256"] == ref["sha256"], "reference binaries differ")
        report["reference_sha256"] = digest(a.reference)
        argv = [str(binpath / "flashcore"),
                "/mnt/sentry-nfs/flash-next-q2/"
                "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
                str(binpath / "libengine.so")]
        argv += [f"/tmp/hold-h{i}.manifest" for i in range(6)]
        report["argv"] = argv
        with contextlib.ExitStack() as resources:
            gpu_text = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True, timeout=15)
            gpus = {int(i): u.strip() for i, u in
                    (line.split(",") for line in gpu_text.splitlines()) if int(i) < 6}
            check(set(gpus) == set(range(6)), "need GPU indices 0 through 5")
            report["gpu_uuids"] = gpus
            for i in range(6):
                lockpath = Path.home() / "codex/asm-port" / (gpus[i] + ".lock")
                lock = resources.enter_context(open(lockpath, "a"))
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            private = resources.enter_context(
                tempfile.TemporaryDirectory(prefix="prefill-cancel-", dir="/tmp"))
            sockpath = str(Path(private) / "core.sock")
            settings = dict(
                CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5",
                SPLIT_SHARES="112,97,81,116,97,97",
                ATTN_MAP="1,1,1,1,4,4,4,4,5,5,5,5", ATTN_DIE="1",
                FLASH_MAXT="128", FLASH_ENGINE_MAXT="128", FLASH_MAXIMG="0",
                FLASH_KV_HOST=a.kv_host, FLASH_PLE_DISK="1", FLASH_PLE_OFF="192",
                FLASH_NBLK="4", FLASH_WATCHDOG_S="15", FLASH_ANCHOR_DIE="0",
                FLASH_LAYERS="48", FLASH_CUBIN=str(binpath / "die.cubin"),
                FLASH_PREFILL_CHUNK="4", FLASH_SOCK=sockpath)
            env = {k: v for k, v in os.environ.items() if not k.startswith("FLASH_")}
            env.update(settings)
            report["environment"] = settings
            stderr = resources.enter_context(open(a.outdir / "stderr.log", "wb"))
            stdout = resources.enter_context(open(a.outdir / "stdout.log", "wb"))
            child = subprocess.Popen(argv, cwd=binpath, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
            resources.callback(stop_child, child, sockpath)
            report["pid"] = child.pid
            deadline, pending = time.monotonic() + 90, b""
            while b"ready\n" not in pending or not Path(sockpath).exists():
                check(child.poll() is None, "core exited before ready")
                check(time.monotonic() < deadline, "core startup deadline exceeded")
                if select.select([child.stdout], [], [], 0.05)[0]:
                    data = os.read(child.stdout.fileno(), 65536)
                    check(data, "EOF before ready")
                    stdout.write(data)
                    stdout.flush()
                    pending += data
            records = report["commands"]
            config = exchange(sockpath, "config", lambda x: x.startswith("config "), records)[-1]
            report["config"] = config
            actual = fields(config)
            expected.update(two_position="1", prefill_chunk="4")
            check(all(actual.get(k) == v for k, v in expected.items()), "core config mismatch")
            with connect(sockpath) as client:
                read_command(client, "snap 5", lambda x: x.startswith("ok:"), records)
                command = "chat 1 " + " ".join(map(str, range(100, 161)))
                records.append({"command": command, "action": "close_after_send"})
                client.sendall((command + "\n").encode())
            cancelled = state(exchange(sockpath, "statehash",
                                      lambda x: x == "statehash done", records))
            report["cancelled"] = cancelled
            positions, paired = cancelled["positions"], cancelled["paired_positions"]
            check(0 <= positions <= 4, f"cancellation exceeded one chunk: {positions}")
            check(0 <= paired <= positions and paired % 2 == 0, "invalid paired counter")
            check(child.poll() is None, "core exited after cancellation")
            recovery = "gen 1 " + " ".join(map(str, range(100, 117)))
            lines = exchange(sockpath, recovery, lambda x: x.startswith("time:"), records)
            token_lines = [x for x in lines if x.startswith("tokens:")]
            check(len(token_lines) == 1, "missing or duplicate recovery tokens")
            tokens = list(map(int, token_lines[0].split(":", 1)[1].split()))
            got = state(exchange(sockpath, "statehash",
                                 lambda x: x == "statehash done", records))
            report["recovery"] = {"tokens": tokens, "time": lines[-1], **got}
            check("0 from the snapshot" in lines[-1], "incomplete snapshot was reused")
            check(tokens == oracle["tokens"], "recovery tokens differ from scalar")
            check(got["positions"] == oracle["positions"], "recovery positions differ")
            check(got["state"] == oracle["state"], "recovery state differs from scalar")
        check(child.returncode == 0, f"core exit {child.returncode}")
        report["pass"] = True
    except Exception as e:
        report["error"] = repr(e)
    report["child_exit_code"] = child.poll() if child is not None else None
    (a.outdir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "error": report.get("error"),
                      "results": str(a.outdir / "results.json")}), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
