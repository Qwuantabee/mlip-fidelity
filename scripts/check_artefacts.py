#!/usr/bin/env python3
"""Correctness checks on the committed main-sweep artefacts. Run in CI
(.github/workflows/tests.yml, after pytest) and by hand:

    python scripts/check_artefacts.py [--results results/]

Exit code 1 on any failure, with every failing check named; 0 otherwise.

WHAT IT CHECKS (inputs: pairs_manifest.json, conformers.jsonl, raw_<model>.json,
decisions_<model>_{all,bio}.csv, summary.csv in the --results directory):

  1. manifest <-> conformers: every pair member exists, every conformer is used,
     pairs are same-molecule, non-self, inside the 1 kcal/mol bar; counts match.
  2. reference gap: every manifest qm_gap_kcal is recomputed from the two
     conformers' energy_hartree as (E_B - E_A) * HARTREE2KCAL and must agree
     within 1e-5 kcal/mol (the manifest stores 6 dp).
  3. geometry, on a sample stratified by subset (up to GEOM_PER_SUBSET
     conformers from EVERY subset, evenly spaced over sorted ids): no atom has a
     neighbour closer than 0.7 A, and every hydrogen has a neighbour within
     1.7 A. A bohr pass-through stretches C-H to ~2.06 A, a double conversion
     shrinks bonds to ~0.5 A. Sampled conformers only, not every conformer.
  4. raw bookkeeping, per raw_<model>.json: ok is true; len(energies_ev) ==
     len(ids); ids unique and equal to the conformer ids; n_ok + n_failed ==
     len(ids); n_failed == number of NaN energies.
  5. raw -> CSV recompute, per model and scope (all, bio): the CSV rows are
     exactly the manifest pairs (in manifest order, bio filtered by is_bio) whose
     two raw energies are finite; model_gap_kcal recomputed from the raw eV
     energies as (E_B - E_A) / HA2EV * HARTREE2KCAL agrees within 1e-6 kcal/mol;
     error_kcal == model_gap - qm_gap within 1e-6; qm_gap_kcal equals the
     manifest within 1e-9 and the Hartree recompute within 1e-5; `agree` equals
     the sign agreement for every row.
  6. summary.csv, per (model, scope): n_pairs, n_decided, n_ties, n_agree,
     accuracy, Wilson interval, bias, median and p90 absolute error,
     median_err_in_bars, n_beyond_bar and null_real_accuracy recomputed from the
     decision CSV; n_conformers_scored/failed equal the raw n_ok/n_failed;
     n_pairs_absent_from_raw is 0 and n_pairs_nan_energy equals the manifest
     pairs in scope minus the CSV rows;
     licence present (MACE-OFF must read academic/ASL); null p-value in [0,1].

WHAT IT CANNOT SEE: whether a raw energy is the energy the named model actually
produces for that geometry (it trusts raw_<model>.json as the model's output;
only re-running the model checks that); whether conformers.jsonl matches the
SPICE archive (the archive is not in the repo); geometry of conformers outside
the stratified sample; whether the null-control p-value is right (only its
presence and range); the held-out artefacts (results/heldout_*), the figures
and the README numbers - none of those are read here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlipfid.decisions import (  # noqa: E402
    DECISION_BAR_KCAL, HARTREE2KCAL, wilson_interval)

# Same CODATA value score.py uses. Written out here, not imported from score.py,
# so a wrong constant in the scorer is caught by an independent recompute.
HA2EV = 27.211386245988
TIE_KCAL = 0.01

# Plausible distances in Angstrom. A double bohr->A conversion shrinks bonds to
# ~0.5 A (below the floor); a missing conversion stretches them 1.89x, so a C-H
# bond (1.09 A) lands at ~2.06 A, beyond the hydrogen ceiling. The ceiling is on
# HYDROGEN only: every H in SPICE is covalently bound (measured max nearest
# neighbour 1.42 A over all 15,232 conformers), while heavy atoms legitimately
# sit beyond 2.2 A from everything: Na+/K+/Cl- in des370k and ion_pairs, and
# stretched C-I bonds in pubchem/des_monomers (647 conformers). The old
# all-atom 2.2 A ceiling passed only because its sample was 300 pubchem ids.
BOND_MIN_A = 0.7
H_NEIGHBOUR_MAX_A = 1.7
GEOM_PER_SUBSET = 60

TOL_MODEL_GAP = 1e-6   # kcal/mol, raw eV -> CSV
TOL_QM_GAP = 1e-5      # kcal/mol, Hartree -> manifest (stored to 6 dp)


class Failures:
    def __init__(self) -> None:
        self.items: list[str] = []

    def check(self, cond: bool, msg: str) -> bool:
        if not cond:
            self.items.append(msg)
        return bool(cond)

    def report(self) -> int:
        if self.items:
            print(f"FAILED {len(self.items)} check(s):")
            for m in self.items:
                print("  -", m)
            return 1
        print("all artefact checks passed")
        return 0


def check_manifest(results: Path, f: Failures) -> tuple[dict, dict]:
    mpath = results / "pairs_manifest.json"
    cpath = results / "conformers.jsonl"
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

    # 2: the reference gap must be the Hartree energies' difference, not just a
    # number the manifest carries.
    ok = [p for p in pairs if p["id_a"] in conf and p["id_b"] in conf]
    if ok:
        ea = np.array([conf[p["id_a"]]["energy_hartree"] for p in ok], dtype=float)
        eb = np.array([conf[p["id_b"]]["energy_hartree"] for p in ok], dtype=float)
        stored = np.array([p["qm_gap_kcal"] for p in ok], dtype=float)
        dev = np.abs((eb - ea) * HARTREE2KCAL - stored)
        n_bad = int((~(dev <= TOL_QM_GAP)).sum())
        f.check(n_bad == 0,
                f"qm_gap: {n_bad} manifest pairs differ from (E_B - E_A) * "
                f"{HARTREE2KCAL} of conformers.jsonl by > {TOL_QM_GAP} kcal/mol "
                f"(max {np.nanmax(dev):.3g})")
    return man, conf


def check_geometry(conf: dict, f: Failures,
                   per_subset: int = GEOM_PER_SUBSET) -> list[str]:
    """Catch a bohr/angstrom mix-up via bond-length plausibility, on a sample
    that contains conformers from every subset (sorted(ids)[:300] used to be
    300 pubchem conformers and nothing else)."""
    if not conf:
        return []
    by_sub: dict[str, list[str]] = {}
    for cid, r in conf.items():
        by_sub.setdefault(r["subset"], []).append(cid)
    ids = []
    for sub in sorted(by_sub):
        s = sorted(by_sub[sub])
        idx = np.unique(np.linspace(0, len(s) - 1, min(per_subset, len(s))).astype(int))
        ids += [s[i] for i in idx]
    too_short = too_long = 0
    for cid in ids:
        r = conf[cid]
        z = np.array(r["atomic_numbers"])
        xyz = np.array(r["positions"], dtype=float)
        if not f.check(xyz.shape == (len(z), 3), f"{cid}: positions shape {xyz.shape}"):
            continue
        # nearest-neighbour distances are enough; we only need the scale
        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        too_short += int((nn < BOND_MIN_A).sum())
        too_long += int((nn[z == 1] > H_NEIGHBOUR_MAX_A).sum())
    f.check(too_short == 0,
            f"geometry: {too_short} atoms have a nearest neighbour closer than "
            f"{BOND_MIN_A} A (positions are probably still in bohr)")
    f.check(too_long == 0,
            f"geometry: {too_long} hydrogens have no neighbour within "
            f"{H_NEIGHBOUR_MAX_A} A (positions are probably still in bohr, or the "
            "geometry is exploded)")
    print(f"geometry sample subsets: {','.join(sorted(by_sub))} "
          f"({len(ids)} conformers)")
    return sorted(by_sub)


def check_raw(results: Path, conf: dict, f: Failures) -> dict[str, dict]:
    """4: bookkeeping of every raw_<model>.json. Returns model -> id->energy eV."""
    paths = sorted(results.glob("raw_*.json"))
    f.check(bool(paths), f"no raw_*.json in {results}")
    out: dict[str, dict] = {}
    for p in paths:
        raw = json.loads(p.read_text())
        model = raw.get("model", p.stem)
        tag = f"{p.name} ({model})"
        if not f.check(bool(raw.get("ok")),
                       f"{tag}: ok is not true; a published artefact must come "
                       "from a complete run"):
            continue
        ids, en = raw.get("ids", []), raw.get("energies_ev", [])
        if not f.check(len(en) == len(ids),
                       f"{tag}: len(energies_ev)={len(en)} != len(ids)={len(ids)}"):
            continue
        e = np.array([np.nan if v is None else v for v in en], dtype=float)
        n_nan = int(np.isnan(e).sum())
        f.check(len(set(ids)) == len(ids),
                f"{tag}: {len(ids) - len(set(ids))} duplicate ids")
        if conf:
            f.check(set(ids) == set(conf),
                    f"{tag}: ids differ from conformers.jsonl "
                    f"({len(set(conf) - set(ids))} missing, "
                    f"{len(set(ids) - set(conf))} extra)")
        n_ok, n_failed = raw.get("n_ok"), raw.get("n_failed")
        f.check(n_ok is not None and n_failed is not None
                and n_ok + n_failed == len(ids),
                f"{tag}: n_ok + n_failed = {n_ok} + {n_failed} != len(ids)={len(ids)}")
        f.check(n_failed == n_nan,
                f"{tag}: n_failed={n_failed} but {n_nan} NaN energies")
        out[model] = dict(zip(ids, e.tolist()))
    return out


def check_decisions(results: Path, man: dict, conf: dict, emaps: dict[str, dict],
                    f: Failures) -> dict[tuple[str, str], pd.DataFrame]:
    """5: every decision CSV row is recomputed from the raw and QM energies."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    csvs = sorted(results.glob("decisions_*.csv"))
    f.check(bool(csvs), "no decisions_*.csv files; run scripts/score.py")
    expected_names = {f"decisions_{m}_{s}.csv" for m in emaps for s in ("all", "bio")}
    orphans = sorted(p.name for p in csvs if p.name not in expected_names)
    f.check(not orphans, f"decision CSVs with no ok raw file: {orphans}")
    pairs = man.get("pairs", [])
    for model, emap in sorted(emaps.items()):
        model_ok = True
        for scope in ("all", "bio"):
            path = results / f"decisions_{model}_{scope}.csv"
            tag = f"{model} {scope}"
            if not f.check(path.exists(), f"{tag}: missing {path.name}"):
                model_ok = False
                continue
            df = pd.read_csv(path)
            frames[(model, scope)] = df
            exp = [p for p in pairs
                   if (scope == "all" or p["is_bio"])
                   and math.isfinite(emap.get(p["id_a"], math.nan))
                   and math.isfinite(emap.get(p["id_b"], math.nan))]
            got = list(zip(df["id_a"], df["id_b"]))
            if not f.check(got == [(p["id_a"], p["id_b"]) for p in exp],
                           f"{tag}: CSV rows ({len(got)}) are not the {len(exp)} "
                           "manifest pairs with two finite raw energies, in order"):
                model_ok = False
                continue
            if not exp:
                continue
            ea = np.array([emap[p["id_a"]] for p in exp])
            eb = np.array([emap[p["id_b"]] for p in exp])
            gap = (eb - ea) / HA2EV * HARTREE2KCAL
            qm_man = np.array([p["qm_gap_kcal"] for p in exp], dtype=float)
            qm_csv = df["qm_gap_kcal"].to_numpy(dtype=float)
            mg = df["model_gap_kcal"].to_numpy(dtype=float)
            er = df["error_kcal"].to_numpy(dtype=float)
            checks = [
                (np.abs(mg - gap), TOL_MODEL_GAP,
                 "model_gap_kcal != (E_B - E_A)[raw eV] / HA2EV * HARTREE2KCAL"),
                (np.abs(er - (gap - qm_csv)), TOL_MODEL_GAP,
                 "error_kcal != recomputed model gap - qm gap"),
                (np.abs(qm_csv - qm_man), 1e-9, "qm_gap_kcal != manifest"),
            ]
            if conf:
                ha = np.array([(conf[p["id_b"]]["energy_hartree"]
                                - conf[p["id_a"]]["energy_hartree"]) * HARTREE2KCAL
                               for p in exp])
                checks.append((np.abs(qm_csv - ha), TOL_QM_GAP,
                               "qm_gap_kcal != (E_B - E_A)[Hartree] * HARTREE2KCAL"))
            for dev, tol, what in checks:
                n_bad = int((~(dev <= tol)).sum())
                if not f.check(n_bad == 0, f"{tag}: {n_bad} rows {what} beyond "
                                           f"{tol} kcal/mol (max dev "
                                           f"{np.nanmax(dev):.4g})"):
                    model_ok = False
            dec = np.abs(qm_csv) > TIE_KCAL
            agree = (np.sign(qm_csv) == np.sign(gap)).astype(int)
            mism = int((agree[dec] != df["agree"].to_numpy(dtype=int)[dec]).sum())
            if not f.check(mism == 0, f"{tag}: {mism} decided rows where `agree` "
                                      "contradicts the recomputed signs"):
                model_ok = False
        if model_ok:
            print(f"raw-to-CSV recompute ok: {model}")
    return frames


def _close(a, b, tol=1e-9) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def check_summary(results: Path, frames: dict, results_raw: dict,
                  f: Failures, man: dict | None = None) -> None:
    """6: every summary.csv number is recomputed from its decision CSV."""
    spath = results / "summary.csv"
    if not f.check(spath.exists(), f"missing {spath}"):
        return
    s = pd.read_csv(spath)
    if "error" in s:
        s = s[s["error"].isna()]
    f.check(len(s) > 0, "summary.csv has no successful rows")
    seen = set()
    for _, r in s.iterrows():
        model, scope = r["model"], r["scope"]
        tag = f"{model} {scope}: summary"
        seen.add((model, scope))
        if not f.check((model, scope) in frames,
                       f"{tag} row has no checked decision CSV"):
            continue
        df = frames[(model, scope)]
        qm = df["qm_gap_kcal"].to_numpy(dtype=float)
        ml = df["model_gap_kcal"].to_numpy(dtype=float)
        dec = np.abs(qm) > TIE_KCAL
        n_dec = int(dec.sum())
        n_agree = int((np.sign(qm[dec]) == np.sign(ml[dec])).sum())
        lo, hi = wilson_interval(n_agree, n_dec)
        ae = np.abs(ml - qm)
        acc = n_agree / n_dec if n_dec else float("nan")
        want = {
            "n_pairs": len(qm), "n_decided": n_dec, "n_ties": len(qm) - n_dec,
            "n_agree": n_agree, "accuracy": acc,
            "accuracy_lo": lo, "accuracy_hi": hi,
            "bias_kcal": float(np.mean(ml - qm)),
            "median_abs_err_kcal": float(np.median(ae)),
            "p90_abs_err_kcal": float(np.percentile(ae, 90)),
            "median_err_in_bars": float(np.median(ae / DECISION_BAR_KCAL)),
            "n_beyond_bar": int((ae > DECISION_BAR_KCAL).sum()),
            "null_real_accuracy": acc,
        }
        if man:
            n_scope = sum(1 for p in man["pairs"] if scope == "all" or p["is_bio"])
            want["n_pairs_absent_from_raw"] = 0
            want["n_pairs_nan_energy"] = n_scope - len(qm)
        raw = results_raw.get(model)
        if raw is not None:
            want["n_conformers_scored"] = raw["n_ok"]
            want["n_conformers_failed"] = raw["n_failed"]
        for k, v in want.items():
            f.check(k in r and _close(r[k], v),
                    f"{tag} {k}={r.get(k)!r} != {v!r} recomputed from the CSV")
        # Wilson intervals bracket the accuracy inside [0,1]
        f.check(0.0 <= lo <= acc <= hi <= 1.0,
                f"{tag} interval [{lo:.4f},{hi:.4f}] does not bracket {acc:.4f}")
        # licence present; ASL must not be described as permissive
        lic = str(r.get("licence", ""))
        f.check(bool(lic) and lic != "nan", f"{tag}: no licence recorded")
        if "mace_off" in str(model):
            f.check("Academic" in lic or "ASL" in lic or "non-commercial" in lic,
                    f"{tag}: MACE-OFF is ASL/academic-only but licence reads {lic!r}")
        # null control present and in range (its value is not recomputed)
        p = r.get("null_p_value")
        if f.check(p is not None and not (isinstance(p, float) and math.isnan(p)),
                   f"{tag}: null control p-value missing (the rails require it)"):
            f.check(0.0 <= float(p) <= 1.0, f"{tag}: null p={p} outside [0,1]")
    missing = sorted(set(frames) - seen)
    f.check(not missing, f"decision CSVs with no summary.csv row: {missing}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"),
                    help="directory holding the artefacts (default: results/)")
    args = ap.parse_args(argv)
    results = Path(args.results)
    f = Failures()
    man, conf = check_manifest(results, f)
    check_geometry(conf, f)
    emaps = check_raw(results, conf, f)
    raws = {json.loads(p.read_text()).get("model"): json.loads(p.read_text())
            for p in sorted(results.glob("raw_*.json"))}
    frames = check_decisions(results, man, conf, emaps, f)
    check_summary(results, frames, raws, f, man)
    return f.report()


if __name__ == "__main__":
    sys.exit(main())
