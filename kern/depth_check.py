#!/usr/bin/env python3
"""Validate one depth.sh response. Exit 0: answer passed; 1: answer failed; 2: protocol/timing failure."""
import json, math, pathlib, sys

def validate(response):
    if not isinstance(response, dict) or "error" in response:
        raise ValueError("API error or non-object response")
    try:
        answer = response["choices"][0]["message"]["content"]
        timing = response["timings"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("missing answer or timings")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(timing, dict):
        raise ValueError("missing answer text or timings")
    for key in ("prompt_n", "predicted_n", "decode_intervals"):
        value = timing.get(key)
        if type(value) is not int or value < 0:
            raise ValueError("invalid timing count: " + key)
    if timing["prompt_n"] < 1 or timing["predicted_n"] < 1:
        raise ValueError("request did not process a prompt and generate an answer")
    if timing["decode_intervals"] != max(0, timing["predicted_n"] - 1):
        raise ValueError("decode interval count differs from generated outputs minus one")
    for key in ("prompt_ms", "prompt_per_second", "predicted_ms", "predicted_per_second", "decode_ms"):
        value = timing.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid timing value: " + key)
    if timing["decode_intervals"]:
        if timing["decode_ms"] <= 0 or timing["predicted_per_second"] <= 0:
            raise ValueError("nonpositive decode measurement")
        # Existing HTTP fields are rounded to 0.1 ms / 0.01 tok/s.
        rate = 1000 * timing["decode_intervals"] / timing["decode_ms"]
        allowance = .011 + rate * .051 / max(timing["decode_ms"] - .051, .001)
        if abs(rate - timing["predicted_per_second"]) > allowance:
            raise ValueError("decode rate does not match intervals / milliseconds")
    elif timing["predicted_per_second"] != 0:
        raise ValueError("a single output cannot measure a decode rate")
    return answer, timing

def main():
    try:
        response = json.loads(pathlib.Path(sys.argv[1]).read_text())
        answer, t = validate(response)
    except (ValueError, OSError, IndexError) as exc:
        print("depth: invalid response: " + str(exc), file=sys.stderr)
        return 2
    passed = sys.argv[2] in answer
    print(f"{sys.argv[3]} paragraphs  prompt {t['prompt_n']} tok  prefill {t['prompt_per_second']} tok/s  "
          f"decode {t['predicted_per_second']} tok/s ({t['decode_intervals']} intervals in {t['decode_ms']} ms)  "
          + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())
