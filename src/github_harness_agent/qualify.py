"""Matched issue replay and blind adjudication. Run with python -m github_harness_agent.qualify."""

import argparse
import hashlib
import json
import secrets
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ARMS = ("incumbent", "plain", "harness2")


def replay(spec, output):
    output.mkdir(exist_ok=False, parents=True)
    (output / "patches").mkdir()
    ledger = []
    for case in spec["issues"]:
        arms = list(ARMS)
        secrets.SystemRandom().shuffle(arms)
        for arm in arms:
            code = secrets.token_hex(8)
            start = time.monotonic()
            # Trusted operator wrappers receive only the frozen task input, on stdin.
            # Each wrapper provisions a disposable worker without evaluation files.
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    result = subprocess.run(
                        spec["commands"][arm],
                        cwd=tmp,
                        input=json.dumps(case["input"]).encode(),
                        capture_output=True,
                        timeout=spec["timeout_seconds"],
                    )
                    patch = result.stdout if result.returncode == 0 else b""
                    status = "produced" if patch else "failed"
                except subprocess.TimeoutExpired:
                    patch, status = b"", "timeout"
            (output / "patches" / f"{code}.patch").write_bytes(patch)
            ledger.append(
                dict(
                    case=case["id"],
                    arm=arm,
                    code=code,
                    status=status,
                    digest=hashlib.sha256(patch).hexdigest(),
                    seconds=round(time.monotonic() - start, 2),
                )
            )
            (output / "private-ledger.json").write_text(json.dumps(ledger, indent=2))
    (output / "blind-scorecard.json").write_text(
        json.dumps(
            [
                dict(
                    code=row["code"],
                    case=row["case"],
                    accepted=None,
                    correction_minutes=None,
                    review_rounds=None,
                    quota_units=None,
                    fallback=None,
                )
                for row in ledger
            ],
            indent=2,
        )
    )


def report(ledger, scores):
    keyed = {r["code"]: r for r in scores}
    if len(keyed) != len(scores) or set(keyed) != {r["code"] for r in ledger}:
        raise ValueError("Score each run exactly once")
    cases = {r["case"] for r in ledger}
    if len(ledger) != len(cases) * 3 or {(r["case"], r["arm"]) for r in ledger} != {
        (c, a) for c in cases for a in ARMS
    }:
        raise ValueError("Incomplete paired comparison")
    summary = {}
    for arm in ARMS:
        runs = [r for r in ledger if r["arm"] == arm]
        observations = [keyed[r["code"]] for r in runs]
        if any(
            type(s["accepted"]) is not bool
            or type(s["correction_minutes"]) not in (float, int)
            or s["correction_minutes"] < 0
            for s in observations
        ):
            raise ValueError("Complete the blind scorecard before unblinding")
        if any(
            s["accepted"] and r["status"] != "produced"
            for r, s in zip(runs, observations)
        ):
            raise ValueError("Failed executions cannot be accepted")
        summary[arm] = dict(
            accepted=sum(s["accepted"] for s in observations),
            median_minutes=statistics.median(
                s["correction_minutes"] for s in observations
            ),
            median_seconds=statistics.median(r["seconds"] for r in runs),
        )
    reference = summary["incumbent"]
    for arm in ARMS[1:]:
        result = summary[arm]
        result["passes_replay_gate"] = (
            len(cases) >= 20
            and result["accepted"] >= reference["accepted"]
            and result["median_minutes"] <= reference["median_minutes"]
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["replay", "report"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.action == "replay":
        replay(json.loads(args.spec.read_text()), args.directory)
    else:
        def read(name):
            return json.loads((args.directory / name).read_text())
        print(
            json.dumps(
                report(read("private-ledger.json"), read("blind-scorecard.json")),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
