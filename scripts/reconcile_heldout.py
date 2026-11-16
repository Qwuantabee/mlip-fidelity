"""Reconcile two held-out benchmarks that disagree about MACE-OFF23.

Published claim (msg 36789, README): "MACE-OFF23's sub-kcal accuracy collapses out
of sample: 0.760 -> 0.467, p = 0.0008." That came from MPCONF196's 30 sub-kcal
pairs. A second held-out set (Folmsbee, 22,379 sub-kcal pairs from 690 compounds)
puts MACE at 0.773 -- no drop at all.

Three candidate explanations, each tested here:

  H1 PSEUDO-REPLICATION. The 30 MPCONF196 pairs come from only 7 compounds, and
     26 of them from 3 linear tripeptides (FGG114, GGF01, GFA01 by formula;
     an earlier version of this note called them macrocycles). A binomial test over 30 pairs assumes 30
     independent observations; it does not have them. Test: cluster bootstrap
     over compounds.
  H2 MOLECULE SIZE. MPCONF196 is macrocycles and peptides (median 64 atoms);
     Folmsbee is drug-like (median 39). If MACE degrades with size, that is a
     property of the model, not of contamination. Test: stratify Folmsbee by
     heavy-atom count.
  H3 GAP DENSITY. SPICE pairs have median |gap| 0.55 kcal/mol; Folmsbee's
     sub-kcal slice has median 0.239 -- much nearer ties. If MACE were riding on
     training-set memory, it should do WORSE on the harder near-tie slice, not
     better. Test: accuracy vs |gap| for both sets on the same axis.

Whichever survives decides what the README is allowed to say.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
MODELS = {"orb_orbmol_v1_conservative": "ORB-omol",
          "mace_off_small": "MACE-OFF23",
          "rdkit_mmff": "MMFF94"}


def cluster_ci(df: pd.DataFrame, n_boot: int = 4000) -> tuple[float, float]:
    """Bootstrap over COMPOUNDS, resampling whole molecules."""
    comps = df.compound.unique()
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(comps, size=len(comps), replace=True)
        sub = pd.concat([df[df.compound == c] for c in pick], ignore_index=True)
        if len(sub):
            vals.append(sub.agree.mean())
    return tuple(np.percentile(vals, [2.5, 97.5]))


def load(tag: str, model: str) -> pd.DataFrame | None:
    """Load a decisions CSV, normalising the two column-naming conventions.

    SPICE uses `molecule` + `qm_gap_kcal`; the held-out runs use `compound` +
    `ref_kcal`. Renaming both to a common pair is what the first version of this
    helper failed to do, which raised KeyError on the SPICE table.
    """
    for cand in (f"{tag}_decisions_{model}.csv", f"decisions_{model}_all.csv",
                 f"heldout_decisions_{model}.csv"):
        p = ROOT / "results" / cand
        if p.exists():
            d = pd.read_csv(p)
            gap = "ref_kcal" if "ref_kcal" in d.columns else "qm_gap_kcal"
            mol = "compound" if "compound" in d.columns else "molecule"
            return d.rename(columns={gap: "gap", mol: "compound"})[
                ["compound", "gap", "agree"]]
    return None


def main() -> int:
    spice = pd.read_csv(ROOT / "results" / "summary.csv")
    spice = spice[spice.scope == "all"].set_index("model")

    print("=" * 78)
    print("H1  PSEUDO-REPLICATION: are the 30 MPCONF196 pairs 30 observations?")
    print("=" * 78)
    for m in MODELS:
        d = load("heldout", m)
        if d is None:
            out = pd.read_csv(ROOT / "results" / f"heldout_decisions_{m}.csv")
            d = out.rename(columns={"ref_kcal": "gap"})[["compound", "gap", "agree"]]
        sub = d[d.gap.abs() <= 1.0]
        lo, hi = cluster_ci(sub)
        top = sub.compound.value_counts()
        print(f"  {MODELS[m]:11s} {int(sub.agree.sum())}/{len(sub)} = "
              f"{sub.agree.mean():.3f}  clustered 95% CI [{lo:.2f}, {hi:.2f}]  "
              f"compounds={sub.compound.nunique()} (top3 hold "
              f"{top.head(3).sum()}/{len(sub)})")

    print()
    print("=" * 78)
    print("H2  MOLECULE SIZE: does MACE degrade with heavy-atom count? (Folmsbee)")
    print("=" * 78)
    bench = json.loads((ROOT / "results" / "heldout_folmsbee.json").read_text())
    size = {cid: len([z for z in c["atomic_numbers"] if z > 1])
            for cid, c in bench["conformers"].items()}
    for m in MODELS:
        p = ROOT / "results" / f"heldout_folmsbee_decisions_{m}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["heavy"] = d.id_a.map(size)
        d["bin"] = pd.cut(d.heavy, [0, 25, 35, 45, 60, 200],
                          labels=["<=25", "26-35", "36-45", "46-60", ">60"])
        agg = d.groupby("bin", observed=True).agree.agg(["mean", "size"])
        line = "  ".join(f"{b}:{r['mean']:.3f}(n={int(r['size'])})"
                         for b, r in agg.iterrows())
        print(f"  {MODELS[m]:11s} {line}")

    print()
    print("=" * 78)
    print("H3  GAP DENSITY: accuracy vs |gap|, SPICE vs Folmsbee")
    print("=" * 78)
    bins = [(0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.0)]
    for m in MODELS:
        sd = load("heldout_junk", m)          # resolves decisions_<m>_all.csv
        fd = load("heldout_folmsbee", m)
        if sd is None or fd is None:
            continue
        parts = []
        for lo, hi in bins:
            s = sd[(sd.gap.abs() >= lo) & (sd.gap.abs() < hi)]
            f = fd[(fd.gap.abs() >= lo) & (fd.gap.abs() < hi)]
            if len(s) < 40 or len(f) < 40:
                continue
            parts.append(f"[{lo},{hi}) SPICE {s.agree.mean():.3f}(n={len(s)}) "
                         f"vs Folmsbee {f.agree.mean():.3f}(n={len(f)})")
        print(f"  {MODELS[m]}")
        for p in parts:
            print(f"     {p}")

    print()
    print("=" * 78)
    print("SUMMARY: in-sample vs each held-out set, sub-kcal slice")
    print("=" * 78)
    fb_s = pd.read_csv(ROOT / "results" / "heldout_folmsbee_summary.csv").set_index("model")
    mp_s = pd.read_csv(ROOT / "results" / "heldout_summary.csv").set_index("model")
    print(f"{'model':11s} {'SPICE':>7s} {'Folmsbee':>9s} {'n':>7s} "
          f"{'MPCONF':>7s} {'n':>4s}")
    for m, lab in MODELS.items():
        ins = spice.loc[m, "accuracy"]
        print(f"{lab:11s} {ins:7.3f} {fb_s.loc[m,'acc_under_1kcal']:9.3f} "
              f"{int(fb_s.loc[m,'n_under_1kcal']):7d} "
              f"{mp_s.loc[m,'acc_under_1kcal']:7.3f} "
              f"{int(mp_s.loc[m,'n_under_1kcal']):4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
