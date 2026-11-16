#!/usr/bin/env python3
"""Decide which models have a COMPLETE sweep, from artefacts only.

`results/sweep_complete.json` was hand-written, which makes it a claim rather than
a check: the file named orb_orbmol_v1_conservative as incomplete while a full
run existed, and earlier it was the reverse. A completeness gate that a human
edits is not a gate, so this recomputes it from the raw artefacts and the
conformer set every time, and refuses to mark anything complete that does not
cover every conformer exactly once.

A model is COMPLETE only if its raw_<model>.json:
  * has ok: true;
  * has unique ids equal to the id set of conformers.jsonl (no missing, no
    extra, no duplicated ids);
  * has len(energies_ev) == len(ids);
  * has an energy block that is neither all-NaN nor all exactly 0.0 (a crashed
    or stubbed runner produces both, and both used to pass);
  * when it carries n_ok / n_failed: n_ok equals the finite-energy count and
    n_ok + n_failed == len(ids);
  * has coverage = finite energies / ids >= --min-coverage (default 0.95).
    The committed artefacts sit well above it: MACE-OFF has 319 NaN of 15,232
    by design (unsupported Na/Li/K/Mg/Ca, coverage 0.979), MMFF 469 (0.969),
    ORB 0 (1.000). 0.95 lets those documented failures through and rejects a
    sweep that silently lost a chunk's worth (2,000 of 15,232 = 13%) of energies.

Exit status: 0 only if at least one model was checked and EVERY model checked is
complete; 1 otherwise, so a shell or CI step can gate on it. --write still
records only the complete models.

raw_<model>.smoke.json files (from run_sweep --limit) are not production
artefacts and are skipped.

Usage: python scripts/verify_completeness.py [--write] [--min-coverage F]
                                             [--results DIR] [--conformers PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIN_COVERAGE = 0.95


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


def _finite(e) -> bool:
    return isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e)


def energy_problem(raw: dict, min_coverage: float = DEFAULT_MIN_COVERAGE) -> str | None:
    """None if the energy block is usable, else a reason string."""
    ids = raw.get("ids") or []
    ens = raw.get("energies_ev")
    if not isinstance(ens, list):
        return "no energies_ev list"
    if len(ens) != len(ids):
        return f"len(energies_ev)={len(ens)} != len(ids)={len(ids)}"
    n_finite = sum(1 for e in ens if _finite(e))
    if n_finite == 0:
        return "energy block is all NaN / non-finite"
    if all(e == 0 for e in ens if _finite(e)):
        return "energy block is all exactly zero"
    if raw.get("n_ok") is not None and int(raw["n_ok"]) != n_finite:
        return f"n_ok={raw['n_ok']} but {n_finite} finite energies"
    if (raw.get("n_ok") is not None and raw.get("n_failed") is not None
            and int(raw["n_ok"]) + int(raw["n_failed"]) != len(ids)):
        return (f"n_ok + n_failed = {int(raw['n_ok']) + int(raw['n_failed'])} "
                f"!= len(ids)={len(ids)}")
    cov = n_finite / len(ids)
    if cov < min_coverage:
        return f"coverage {cov:.4f} ({n_finite}/{len(ids)}) below {min_coverage}"
    return None


def model_problem(raw: dict, want: list[str],
                  min_coverage: float = DEFAULT_MIN_COVERAGE) -> str | None:
    return coverage_problem(raw, want) or energy_problem(raw, min_coverage)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--conformers", default=str(ROOT / "results" / "conformers.jsonl"))
    ap.add_argument("--write", action="store_true",
                    help="write results/sweep_complete.json")
    ap.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                    help="minimum fraction of conformers with a finite energy "
                         f"(default {DEFAULT_MIN_COVERAGE})")
    args = ap.parse_args()

    res = Path(args.results)
    want = conformer_ids(Path(args.conformers))
    complete, partial = [], {}
    for p in sorted(res.glob("raw_*.json")):
        if p.name.endswith(".smoke.json"):
            continue
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError:
            partial[p.stem] = "unparseable"
            continue
        if not raw.get("ok"):
            partial[raw.get("model", p.stem)] = raw.get("error", "failed")[:120]
            continue
        problem = model_problem(raw, want, args.min_coverage)
        if problem:
            partial[raw.get("model", p.stem)] = problem
        else:
            complete.append(raw["model"])

    out = {
        "complete": sorted(complete),
        "partial": partial,
        "n_conformers": len(want),
        "min_coverage": args.min_coverage,
        "note": ("Recomputed from raw_*.json by scripts/verify_completeness.py. "
                 "A model is complete only if its artefact covers every conformer "
                 "in conformers.jsonl exactly once, with one energy per id, a "
                 "non-degenerate (not all-NaN, not all-zero) energy block, "
                 "consistent n_ok/n_failed and finite-energy coverage >= "
                 "min_coverage; nothing here is hand-edited."),
    }
    print(json.dumps(out, indent=2))
    if args.write:
        (res / "sweep_complete.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {res / 'sweep_complete.json'}")
    if partial or not complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
