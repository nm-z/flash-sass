#!/usr/bin/env python3
"""Check real PLE failures through a private flashd and an owned shadow core.

Use a shadow engine built with FLASH_PLE_IO_TEST and matching die.cubin,
the real disk PLE table, all 48 layers, and a 512-position context.
This script does not reload or quit the core and terminates only its flashd.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


class Core:
    def __init__(self, path):
        self.path = path
        self.protocol = []

    def request(self, command, terminal=None, timeout=240):
        if terminal is None:
            terminal = lambda line: True
        lines = []
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(self.path)
            self.protocol.append({"direction": "in", "line": command})
            sock.sendall((command + "\n").encode())
            with sock.makefile("rb") as reader:
                while True:
                    raw = reader.readline()
                    assert raw, ("engine closed connection", command, lines)
                    line = raw.decode("utf-8").rstrip("\r\n")
                    self.protocol.append({"direction": "out", "line": line})
                    lines.append(line)
                    if terminal(line):
                        return lines

    def control(self, mode="none", position=0, timeout=240):
        line = self.request(
            "ple_io_test %s %d 3 17" % (mode, position),
            timeout=timeout,
        )[0]
        assert line.startswith("ok: ple_io_test "), line

    def status(self):
        line = self.request("ple_io_test status")[0]
        assert line.startswith("ple_io_test "), line
        result = {
            key: int(value)
            for key, value in re.findall(
                r"(launches|failed|pos|head|snapshot)=(-?\d+)", line
            )
        }
        assert set(result) == {
            "launches", "failed", "pos", "head", "snapshot"
        }, line
        return result

    def fresh(self):
        self.control()
        lines = self.request(
            "chat 0 1 2",
            terminal=lambda line: (
                line.startswith("done ") or line.startswith("error")
            ),
        )
        assert lines[-1].startswith("done "), lines
        state = self.status()
        assert state["snapshot"] == 0 and state["failed"] == 0, state
        return state


def sse_events(body):
    events = [
        line[5:].strip()
        for line in body.splitlines()
        if line.startswith("data:")
    ]
    assert events and events[-1] == "[DONE]", body
    return [json.loads(event) for event in events[:-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flashd", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    core_path = str(Path(args.core).resolve())
    assert core_path != str(Path("/tmp/flash.sock").resolve()), "Use an owned shadow core."
    flashd = str(Path(args.flashd).resolve())
    gguf = str(Path(args.gguf).resolve())
    assert Path(flashd).is_file() and Path(gguf).is_file()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    core = Core(core_path)
    results = []
    run_status = "FAIL"
    proc = None
    log = (out / "flashd.log").open("w")
    config = None
    try:
        config = core.request("config")[0]
        assert "ple_disk=1" in config and "layers=48/48" in config, config
        info = core.request("info")[0].split()
        assert info[0] == "info" and int(info[1]) >= 512, info
        core.fresh()
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        base = "http://127.0.0.1:%d" % port
        env = dict(os.environ, FLASHD_GREEDY="1", FLASHD_THINKING="0")
        proc = subprocess.Popen(
            [flashd, gguf, core_path, "127.0.0.1:%d" % port],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, "flashd exited; see flashd.log"
            try:
                with urllib.request.urlopen(base + "/health", timeout=1) as response:
                    assert response.status == 200
                break
            except (urllib.error.URLError, OSError):
                assert time.monotonic() < deadline, "flashd startup timeout"
                time.sleep(0.05)

        def post(label, stream=False, long_prompt=False):
            content = " ".join(["x"] * 180) if long_prompt else "Return the single word hello."
            request_body = {
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4, "stream": stream, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(request_body).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                response = urllib.request.urlopen(request, timeout=240)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                code = response.code
                headers = dict(response.headers)
                body = response.read().decode("utf-8")
            (out / (label + ".txt")).write_text(body)
            (out / (label + ".meta.json")).write_text(
                json.dumps({"status": code, "headers": headers,
                            "request": request_body}, indent=2) + "\n")
            return code, body

        code, body = post("baseline")
        assert code == 200, (code, body)
        baseline = json.loads(body)
        assert "error" not in baseline and baseline["choices"], baseline
        assert baseline["choices"][0]["finish_reason"] in ("stop", "length"), baseline

        for mode in ("eof", "eio"):
            for position in (0, 160, 161):
                for stream in (False, True):
                    label = "%s_pos%d_%s" % (mode, position, "sse" if stream else "json")
                    core.fresh()
                    core.control(mode, position)
                    before = core.status()
                    code, body = post(label, stream=stream, long_prompt=position >= 160)
                    after = core.status()
                    assert after["failed"] == 1, (label, after, code, body)
                    assert after["pos"] == position and after["head"] == 3, (label, after)
                    assert after["snapshot"] == 0, (label, after)
                    if position == 0:
                        assert after["launches"] == before["launches"], (label, before, after)
                    else:
                        assert after["launches"] > before["launches"], (label, before, after)
                    if stream:
                        assert code == 200, (label, code, body)
                        events = sse_events(body)
                        assert any("error" in event for event in events), (label, events)
                        assert not any(
                            choice.get("finish_reason") is not None
                            for event in events for choice in event.get("choices", [])
                        ), (label, events)
                    else:
                        assert code == 502, (label, code, body)
                        payload = json.loads(body)
                        assert "error" in payload and "choices" not in payload, (label, payload)
                    core.control()
                    healthy_code, healthy_body = post(label + "_recovery")
                    assert healthy_code == 200, (label, healthy_code, healthy_body)
                    healthy = json.loads(healthy_body)
                    assert "error" not in healthy, (label, healthy)
                    assert healthy["choices"] == baseline["choices"], (label, baseline, healthy)
                    recovered = core.status()
                    assert recovered["failed"] == 0, (label, recovered)
                    results.append({
                        "case": label, "status": "PASS", "http_status": code,
                        "before": before, "after": after, "recovered": recovered,
                    })
                    print("PASS", label, "and deterministic healthy reuse", flush=True)
        run_status = "PASS"
        print("PASS: %d real-PLE HTTP failure/recovery cases" % len(results), flush=True)
    finally:
        try:
            core.control(timeout=5)
        except Exception as exc:
            print("Could not clear shadow fault injection:", repr(exc), flush=True)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log.close()
        (out / "protocol.json").write_text(json.dumps(core.protocol, indent=2) + "\n")
        (out / "result.json").write_text(json.dumps({
            "status": run_status, "kind": "real-PLE real-flashd failure/recovery",
            "config": config, "core": core_path, "flashd": flashd, "gguf": gguf,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
