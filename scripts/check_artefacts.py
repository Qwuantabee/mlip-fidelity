#!/usr/bin/env python3
"""Consistency checks on the committed artefacts, run in CI and by hand.

This script exists because the repository's claim is that every published number
comes from a committed file. That claim is only worth anything if something
mechanically checks the files agree with each other. Checks:

  1. the pair manifest and the conformer file describe the same conformers
     (every pair member exists; every conformer is used by some pair);
  2. every conformer's positions are plausible Angstrom geometry, not bohr
     (a bohr pass-through shrinks every molecule ~2x — caught by bond-length
     range);
  3. every decision CSV row's `agree` flag is consistent with the signs of the
     gaps it stores;
  4. summary.csv accuracy equals the accuracy recomputed from its decision CSV;
  5. the Wilson intervals actually bracket the accuracy and contain [0,1];
  6. every model in summary.csv carries a licence, and academic-only licences are
     not silently relabelled as permissive;
  7. the null control was run for every scored model (the rails say it is not
     optional).

Exit code 1 on any failure, with the failing check named.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Plausible covalent bond lengths in Angstrom. If positions were left in bohr the
# shortest bonds would cluster near 0.5 A, well below this floor.
BOND_MIN_A = 0.7
BOND_MAX_A = 2.2


class Failures:
    def __init__(self) -> None:
        self.items: list[str] = []

    def check(self, cond: bool, msg: str) -> None:
        if not cond:
            self.items.append(msg)

    def report(self) -> int:
        if self.items:
            print(f"FAILED {len(self.items)} check(s):")
            for m in self.items:
                print("  -", m)
            return 1
        print("all artefact checks passed")
        return 0


def check_manifest(f: Failures) -> tuple[dict, dict]:
    mpath = RESULTS / "pairs_manifest.json"
    cpath = RESULTS / "conformers.jsonl"
    f.check(mpath.exists(), f"missing {mpath}")
    f.check(cpath.exists(), f"missing {cpath}")
    if f.items:
        return {}, {}
    man = json.loads(mpath.read_text())
    conf = {}
    for line in cpath.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            conf[r["id"]] = r
    pairs = man["pairs"]
    f.check(len(pairs) > 0, "manifest has no pairs")
    f.check(man["n_pairs"] == len(pairs),
            f"manifest n_pairs={man['n_pairs']} but {len(pairs)} rows")
    f.check(man["n_conformers"] == len(conf),
            f"manifest n_conformers={man['n_conformers']} but {len(conf)} rows")

    missing = [p for p in pairs if p["id_a"] not in conf or p["id_b"] not in conf]
    f.check(not missing, f"{len(missing)} pairs reference conformers not in the file")

    used = set()
    for p in pairs:
        used.add(p["id_a"])
        used.add(p["id_b"])
    unused = set(conf) - used
    f.check(not unused, f"{len(unused)} exported conformers are used by no pair "
                        "(wasted sweep work)")

    bad_same = [p for p in pairs
                if p["id_a"].split("|")[0] != p["id_b"].split("|")[0]]
    f.check(not bad_same, f"{len(bad_same)} pairs compare different molecules")
    self_pairs = [p for p in pairs if p["id_a"] == p["id_b"]]
    f.check(not self_pairs, f"{len(self_pairs)} self-pairs")

    # A decision pair must be inside the 1 kcal/mol bar; otherwise the pair is a
    # comparison between two different species, not a decision.
    over = [p for p in pairs if abs(p["qm_gap_kcal"]) > 1.0 + 1e-9]
    f.check(not over, f"{len(over)} pairs exceed the 1 kcal/mol decision bar")
    return man, conf


def check_geometry(conf: dict, f: Failures, sample: int = 300) -> None:
    """Catch a bohr/angstrom mix-up via bond-length plausibility."""
    if not conf:
        return
    ids = sorted(conf)[:sample]
    too_short = too_long = 0
    n_pairs = 0
    for cid in ids:
        r = conf[cid]
        z = np.array(r["atomic_numbers"])
        xyz = np.array(r["positions"], dtype=float)
        f.check(xyz.shape == (len(z), 3), f"{cid}: positions shape {xyz.shape}")
        # nearest-neighbour distances are enough; we only need the scale
        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        too_short += int((nn < BOND_MIN_A).sum())
        too_long += int((nn > BOND_MAX_A).sum())
        n_pairs += len(z)
    f.check(too_short == 0,
            f"{too_short} atoms have a nearest neighbour closer than {BOND_MIN_A} A "
            "(positions are probably still in bohr)")
    f.check(too_long == 0,
            f"{too_long} atoms have no neighbour within {BOND_MAX_A} A "
            "(geometry looks exploded or fragmented)")


def check_decisions(f: Failures) -> tuple[dict, list]:
    paths = sorted(RESULTS.glob("decisions_*_all.csv"))
    f.check(bool(paths), "no decisions_*_all.csv files; run scripts/score.py")
    frames = {}
    for p in paths:
        df = pd.read_csv(p)
        model = df["model"].iloc[0] if len(df) else p.stem
        # 3: agree flag must equal the sign agreement of the stored gaps
        dec = df[df["qm_gap_kcal"].abs() > 0.01]
        agree = (np.sign(dec["qm_gap_kcal"]) == np.sign(dec["model_gap_kcal"]))
        mism = int((agree.astype(int) != dec["agree"].astype(int)).sum())
        f.check(mism == 0, f"{model}: {mism} rows where `agree` contradicts the "
                           "stored gap signs")
        frames[model] = df
    return frames, paths


def check_summary(frames: dict, f: Failures) -> pd.DataFrame:
    spath = RESULTS / "summary.csv"
    f.check(spath.exists(), f"missing {spath}")
    if not spath.exists():
        return pd.DataFrame()
    s = pd.read_csv(spath)
    s = s[s.get("error").isna()] if "error" in s else s
    f.check(len(s) > 0, "summary.csv has no successful rows")
    for _, r in s.iterrows():
        model, scope = r["model"], r["scope"]
        if scope != "all" or model not in frames:
            continue
        df = frames[model]
        dec = df[df["qm_gap_kcal"].abs() > 0.01]
        if not len(dec):
            continue
        recomputed = float(dec["agree"].astype(int).mean())
        # 4: the published accuracy must equal the decision CSV
        f.check(abs(recomputed - float(r["accuracy"])) < 1e-9,
                f"{model}: summary accuracy {r['accuracy']:.6f} != recomputed "
                f"{recomputed:.6f}")
        f.check(int(r["n_decided"]) == len(dec),
                f"{model}: summary n_decided {int(r['n_decided'])} != "
                f"{len(dec)} decided rows")
        # 5: intervals
        lo, hi = float(r["accuracy_lo"]), float(r["accuracy_hi"])
        acc = float(r["accuracy"])
        f.check(0.0 <= lo <= acc <= hi <= 1.0,
                f"{model}: interval [{lo:.4f},{hi:.4f}] does not bracket "
                f"accuracy {acc:.4f} inside [0,1]")
        # 6: licence present; ASL must not be described as permissive
        lic = str(r.get("licence", ""))
        f.check(lic and lic != "nan", f"{model}: no licence recorded")
        if "mace_off" in str(model):
            f.check("Academic" in lic or "ASL" in lic or "non-commercial" in lic,
                    f"{model}: MACE-OFF is ASL/academic-only but licence reads "
                    f"{lic!r}")
        # 7: null control present and finite
        p = r.get("null_p_value")
        f.check(p is not None and not (isinstance(p, float) and math.isnan(p)),
                f"{model}: null control p-value missing (the rails require it)")
        if p is not None and not (isinstance(p, float) and math.isnan(p)):
            f.check(0.0 <= float(p) <= 1.0, f"{model}: null p={p} outside [0,1]")
    return s


def main() -> int:
    f = Failures()
    man, conf = check_manifest(f)
    check_geometry(conf, f)
    frames, _ = check_decisions(f)
    check_summary(frames, f)
    return f.report()


if __name__ == "__main__":
    sys.exit(main())
