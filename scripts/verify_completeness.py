#!/usr/bin/env python3
"""Decide which models have a COMPLETE sweep, from artefacts only.

`results/sweep_complete.json` was hand-written, which makes it a claim rather than
a check: the file named orb_orbmol_v1_conservative as incomplete while a full
run existed, and earlier it was the reverse. A completeness gate that a human
edits is not a gate, so this recomputes it from the raw artefacts and the
conformer set every time, and refuses to mark anything complete that does not
cover every conformer exactly once.

Usage: python scripts/verify_completeness.py [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def conformer_ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines()
            if line.strip()]


def coverage_problem(raw: dict, want: list[str]) -> str | None:
    """None if the raw artefact covers `want` exactly, else a reason string."""
    ids = raw.get("ids") or []
    if not ids:
        return "no ids"
    if len(ids) != len(set(ids)):
        return f"{len(ids) - len(set(ids))} duplicated conformer ids"
    missing = set(want) - set(ids)
    extra = set(ids) - set(want)
    if missing and missing == set(want):
        return "artefact covers none of the conformers"
    if missing or extra:
        return f"{len(missing)} conformers missing, {len(extra)} unexpected"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--conformers", default=str(ROOT / "results" / "conformers.jsonl"))
    ap.add_argument("--write", action="store_true",
                    help="write results/sweep_complete.json")
    args = ap.parse_args()

    res = Path(args.results)
    want = conformer_ids(Path(args.conformers))
    complete, partial = [], {}
    for p in sorted(res.glob("raw_*.json")):
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError:
            partial[p.stem] = "unparseable"
            continue
        if not raw.get("ok"):
            partial[raw.get("model", p.stem)] = raw.get("error", "failed")[:120]
            continue
        problem = coverage_problem(raw, want)
        if problem:
            partial[raw.get("model", p.stem)] = problem
        else:
            complete.append(raw["model"])

    out = {
        "complete": sorted(complete),
        "partial": partial,
        "n_conformers": len(want),
        "note": ("Recomputed from raw_*.json by scripts/verify_completeness.py. "
                 "A model is complete only if its artefact covers every conformer "
                 "in conformers.jsonl exactly once; nothing here is hand-edited."),
    }
    print(json.dumps(out, indent=2))
    if args.write:
        (res / "sweep_complete.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {res / 'sweep_complete.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
