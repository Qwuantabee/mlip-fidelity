#!/usr/bin/env python3
"""Turn raw model energies into decision fidelity, per model and subset.

Reads `results/raw_<model>.json` (written by scripts/run_sweep.py) plus the pair
manifest, and writes:
  results/decisions_<model>.csv   one row per pair: qm gap, model gap, agree?
  results/summary.csv             one row per (model, subset scope)
  results/summary.json            the same, with the null control per model

All energies are converted to a common zero by using the PAIR DIFFERENCE, never
the absolute energy: different models use different reference energies (isolated
atoms vs a fitted baseline), so an absolute comparison is meaningless and only
differences are transferable. That is also why the decision framing is the right
one for this question and an energy-MAE framing would not be.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlipfid.decisions import (  # noqa: E402
    DECISION_BAR_KCAL, decision_scores, null_control)

HA2EV = 27.211386245988


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def energy_map(raw: dict) -> dict[str, float]:
    """id -> energy in eV, dropping failures (nan)."""
    out = {}
    for cid, e in zip(raw["ids"], raw["energies_ev"]):
        if e == e:  # not nan
            out[cid] = float(e)
    return out


def score_model(raw_path: Path, manifest: dict, scope: str) -> tuple[list[dict], dict]:
    raw = json.loads(raw_path.read_text())
    model = raw["model"]
    emap = energy_map(raw)
    rows, qm_list, ml_list = [], [], []
    for p in manifest["pairs"]:
        if scope == "bio" and not p["is_bio"]:
            continue
        ea, eb = emap.get(p["id_a"]), emap.get(p["id_b"])
        if ea is None or eb is None:
            continue
        # model gap in eV -> kcal/mol, sign preserved: E(B) - E(A)
        gap_ev = eb - ea
        gap_kcal = gap_ev * HA2EV / HA2EV * 0.0  # placeholder replaced below
        gap_kcal = (gap_ev / HA2EV) * 627.5094740631
        agree = int(np.sign(gap_kcal) == np.sign(p["qm_gap_kcal"]))
        rows.append({
            "model": model, "scope": scope, "molecule": p["molecule"],
            "subset": p["subset"], "id_a": p["id_a"], "id_b": p["id_b"],
            "qm_gap_kcal": p["qm_gap_kcal"], "model_gap_kcal": round(gap_kcal, 6),
            "error_kcal": round(gap_kcal - p["qm_gap_kcal"], 6),
            "agree": agree,
        })
        qm_list.append(p["qm_gap_kcal"])
        ml_list.append(gap_kcal)
    qm = np.array(qm_list, dtype=float)
    ml = np.array(ml_list, dtype=float)
    stats = decision_scores(qm, ml) if len(qm) else {}
    if len(qm) >= 5:
        stats.update({f"null_{k}": v for k, v in
                      null_control(ml, qm, n_shuffle=200).items()})
    stats.update({"model": model, "scope": scope,
                  "family": raw.get("meta", {}).get("family", "?"),
                  "licence": raw.get("meta", {}).get("licence", "?"),
                  "wall_seconds": raw.get("wall_seconds"),
                  "n_conformers_scored": raw.get("n_ok"),
                  "n_conformers_failed": raw.get("n_failed")})
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "results" / "pairs_manifest.json"))
    ap.add_argument("--raw-glob", default=str(ROOT / "results" / "raw_*.json"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    out_dir = Path(args.out)
    raw_files = sorted(Path().glob(args.raw_glob)) or \
        sorted(Path(args.raw_glob.rsplit("/", 1)[0]).glob(
            args.raw_glob.rsplit("/", 1)[1]))
    if not raw_files:
        raise SystemExit(f"no raw files matched {args.raw_glob}")

    summary = []
    for raw_path in raw_files:
        raw = json.loads(raw_path.read_text())
        if not raw.get("ok"):
            summary.append({"model": raw.get("model", raw_path.stem),
                            "scope": "all", "error": raw.get("error", "failed")})
            continue
        for scope in ("all", "bio"):
            rows, stats = score_model(raw_path, manifest, scope)
            if not stats or not stats.get("n_pairs"):
                continue
            stem = raw.get("model", "model")
            with open(out_dir / f"decisions_{stem}_{scope}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            summary.append(stats)

    if summary:
        keys = sorted({k for s in summary for k in s})
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for s in summary:
        if "error" in s:
            print(f"{s['model']:20s} FAILED {s['error'][:60]}")
            continue
        print(f"{s['model']:20s} {s['scope']:4s} n={s['n_decided']:5d} "
              f"acc={s['accuracy']:.3f} "
              f"[{s['accuracy_lo']:.3f},{s['accuracy_hi']:.3f}] "
              f"medErr={s['median_abs_err_kcal']:.3f} kcal  "
              f"nullP={s.get('null_p_value', float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
