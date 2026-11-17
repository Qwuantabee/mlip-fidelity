"""Every clustered interval and design effect the README prints, from committed CSVs.

Pairs from one molecule are not independent observations, so the README's
pair-level Wilson intervals are too narrow. This script produces the numbers that
say by how much, and the intervals that account for it. It reads only the
committed decision CSVs and writes `results/intervals.json`;
`tests/test_intervals.py` re-runs it and checks that every figure printed in the
README equals the JSON at the README's precision.

What is computed (all bootstraps resample whole molecules / compounds, with
replacement, N_BOOT draws, fixed seed, 95% percentile intervals):

  spice_pairwise   ORB-MACE and MACE-MMFF94 accuracy differences on the pairs all
                   three models decided, resampling those pairs' molecules.
  spice_clustering per model, the design effect (bootstrap variance / binomial
                   variance), its square root (how much too narrow the Wilson
                   interval is) and the implied intra-molecule correlation
                   ICC = (deff - 1) / (m0 - 1), m0 = sum(n_i^2) / sum(n_i).
  folmsbee_cluster per model, a compound-clustered interval for the held-out
                   accuracy (every sub-kcal pair, ties included, as in the README's
                   bold column) and its design effect.
  mmff_offset      (Folmsbee - SPICE shift of a model) - (MMFF94's shift), with a
                   paired bootstrap: one draw of SPICE molecules and one draw of
                   Folmsbee compounds is shared by all three models, the two
                   datasets drawn independently of each other.

Ties: SPICE accuracy excludes pairs with |reference gap| <= 0.01 kcal/mol, as in
the README's SPICE table; the Folmsbee accuracy counts every sub-kcal pair, as in
the README's bold held-out column.

Usage:  python scripts/intervals.py            (prints and writes the JSON)
        python scripts/intervals.py --check    (exits 1 if the JSON is stale)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "intervals.json"
MODELS = {"orb": "orb_orbmol_v1_conservative",
          "mace": "mace_off_small",
          "mmff": "rdkit_mmff"}
TIE_KCAL = 0.01
N_BOOT = 10_000
SEED = 20261117


def _spice(model: str) -> pd.DataFrame:
    d = pd.read_csv(RES / f"decisions_{model}_all.csv")
    d = d[d.qm_gap_kcal.abs() > TIE_KCAL]
    return d[["molecule", "id_a", "id_b", "agree"]].rename(columns={"molecule": "group"})


def _folmsbee(model: str) -> pd.DataFrame:
    d = pd.read_csv(RES / f"heldout_folmsbee_decisions_{model}.csv")
    return d[["compound", "id_a", "id_b", "agree"]].rename(columns={"compound": "group"})


def _per_group(d: pd.DataFrame, groups: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """(agreements, pairs) per group, aligned to `groups`; absent groups get 0, 0."""
    g = d.groupby("group").agree.agg(["sum", "size"]).reindex(groups, fill_value=0)
    return g["sum"].to_numpy(float), g["size"].to_numpy(float)


def _boot_idx(n_groups: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_groups, size=(N_BOOT, n_groups))


def _ci(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def _deff(acc_boot: np.ndarray, p: float, n: int) -> float:
    return float(np.var(acc_boot, ddof=1) / (p * (1 - p) / n))


def compute() -> dict:
    rng = np.random.default_rng(SEED)
    spice = {k: _spice(m) for k, m in MODELS.items()}
    folm = {k: _folmsbee(m) for k, m in MODELS.items()}
    out: dict = {"n_boot": N_BOOT, "seed": SEED, "tie_kcal": TIE_KCAL}

    # 1. pairwise differences on the pairs all three decided
    key = ["group", "id_a", "id_b"]
    common = spice["orb"][key].merge(spice["mace"][key]).merge(spice["mmff"][key])
    sub = {k: d.merge(common, on=key) for k, d in spice.items()}
    groups = pd.Index(sorted(common.group.unique()))
    sums = {k: _per_group(d, groups) for k, d in sub.items()}
    idx = _boot_idx(len(groups), rng)
    acc = {k: s[idx].sum(1) / n[idx].sum(1) for k, (s, n) in sums.items()}
    pt = {k: float(d.agree.mean()) for k, d in sub.items()}
    out["spice_pairwise"] = {
        "n_pairs": int(len(common)), "n_molecules": int(len(groups)),
        "accuracy": pt,
        "orb_minus_mace": {"diff": pt["orb"] - pt["mace"], "ci": _ci(acc["orb"] - acc["mace"])},
        "mace_minus_mmff": {"diff": pt["mace"] - pt["mmff"], "ci": _ci(acc["mace"] - acc["mmff"])},
    }

    # 2. SPICE clustering: design effect and ICC per model, on its own decided pairs
    sg = pd.Index(sorted(set().union(*(d.group for d in spice.values()))))
    s_sums = {k: _per_group(d, sg) for k, d in spice.items()}
    s_idx = _boot_idx(len(sg), rng)
    s_boot = {k: s[s_idx].sum(1) / n[s_idx].sum(1) for k, (s, n) in s_sums.items()}
    out["spice_clustering"] = {}
    for k, d in spice.items():
        p, n = float(d.agree.mean()), len(d)
        sizes = d.groupby("group").size().to_numpy(float)
        m0 = float((sizes ** 2).sum() / sizes.sum())
        deff = _deff(s_boot[k], p, n)
        out["spice_clustering"][k] = {
            "accuracy": p, "n_pairs": n, "n_molecules": int(len(sizes)),
            "cluster_ci": _ci(s_boot[k]), "design_effect": deff,
            "width_ratio": float(np.sqrt(deff)), "m0": m0,
            "icc": (deff - 1) / (m0 - 1)}

    # 3. Folmsbee compound-clustered intervals, ties included
    fg = pd.Index(sorted(set().union(*(d.group for d in folm.values()))))
    f_sums = {k: _per_group(d, fg) for k, d in folm.items()}
    f_idx = _boot_idx(len(fg), rng)
    f_boot = {k: s[f_idx].sum(1) / n[f_idx].sum(1) for k, (s, n) in f_sums.items()}
    out["folmsbee_cluster"] = {}
    for k, d in folm.items():
        p, n = float(d.agree.mean()), len(d)
        out["folmsbee_cluster"][k] = {
            "accuracy": p, "n_pairs": n, "n_compounds": int(d.group.nunique()),
            "cluster_ci": _ci(f_boot[k]), "design_effect": _deff(f_boot[k], p, n)}

    # 4. the MMFF94-offset estimate, paired across models within each dataset
    shift = {k: out["folmsbee_cluster"][k]["accuracy"] - out["spice_clustering"][k]["accuracy"]
             for k in MODELS}
    shift_boot = {k: f_boot[k] - s_boot[k] for k in MODELS}
    out["mmff_offset"] = {}
    for k in ("orb", "mace"):
        dist = shift_boot[k] - shift_boot["mmff"]
        out["mmff_offset"][k] = {
            "estimate": shift[k] - shift["mmff"], "ci": _ci(dist),
            "se": float(np.std(dist, ddof=1))}
    return out


def _fmt(o: dict) -> str:
    return json.dumps(o, indent=2, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if results/intervals.json differs from a recompute")
    a = ap.parse_args()
    res = compute()
    text = _fmt(res)
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is missing or stale", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} matches a recompute")
        return 0
    OUT.write_text(text)
    pw = res["spice_pairwise"]
    print(f"SPICE common pairs {pw['n_pairs']:,} from {pw['n_molecules']:,} molecules")
    for name in ("orb_minus_mace", "mace_minus_mmff"):
        r = pw[name]
        print(f"  {name}: {r['diff']:+.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")
    for k, r in res["spice_clustering"].items():
        print(f"SPICE {k}: deff {r['design_effect']:.2f} (x{r['width_ratio']:.2f} wider), "
              f"m0 {r['m0']:.1f}, ICC {r['icc']:.3f}")
    for k, r in res["folmsbee_cluster"].items():
        print(f"Folmsbee {k}: {r['accuracy']:.4f} cluster CI [{r['cluster_ci'][0]:.4f}, "
              f"{r['cluster_ci'][1]:.4f}] deff {r['design_effect']:.2f}")
    for k, r in res["mmff_offset"].items():
        print(f"offset {k}: {r['estimate']:+.4f} CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] "
              f"SE {r['se']:.4f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
