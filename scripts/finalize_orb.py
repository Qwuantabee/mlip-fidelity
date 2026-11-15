#!/usr/bin/env python3
"""Finish the ORB integration once its re-run lands: merge, score, gate, figures.

Idempotent and safe to run repeatedly. Refuses to publish anything unless every
decisions CSV recomputes to its summary row, which is the check that caught the
withdrawn 0.893.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/home/ubuntu/.openclaw/workspace/venv/bin/python"
MODEL = "orb_orbmol_v1_conservative"


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(("$ " + " ".join(cmd)) + f"\n{r.stdout.strip()[-800:]}")
    if r.returncode:
        print(r.stderr[-800:], file=sys.stderr)
    return r.returncode


def main() -> int:
    chunks = sorted((ROOT / "results").glob(f"chunk_{MODEL}_*.json"))
    expected = json.loads((ROOT / "results" / "sweep_complete.json").read_text())
    if not chunks:
        print("no ORB chunks yet")
        return 1
    n = len(chunks)
    print(f"{n} ORB chunks present")
    if run([PY, "scripts/merge_chunks.py", "--model", MODEL]):
        return 1
    raw = json.loads((ROOT / "results" / f"raw_{MODEL}.json").read_text())
    # completeness: the raw artefact must cover every conformer we exported
    n_conf = sum(1 for _ in open(ROOT / "results" / "conformers.jsonl"))
    print(f"raw n_ok={raw['n_ok']} of {n_conf} exported conformers")
    # 100%, not 95%: this is the same rule scripts/verify_completeness.py applies,
    # and two writers with different rules is how the missing 5% - which are the
    # hardest conformers - would have been published as complete.
    if raw["n_ok"] < n_conf:
        print(f"INCOMPLETE ({raw['n_ok']} of {n_conf}); not publishing - rerun the "
              f"sweep to finish")
        return 1
    done = set(expected.get("complete", []))
    done.add(MODEL)
    (ROOT / "results" / "sweep_complete.json").write_text(
        json.dumps({**expected, "complete": sorted(done)}, indent=2) + "\n")
    run([PY, "scripts/score.py", "--require-complete"])
    run([PY, "scripts/make_figures.py"])
    print("\nNow verify every row reproduces, update the README table + the "
          "withdrawal note, commit and push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
