#!/usr/bin/env python3
"""Figures for the README. Every panel reads a committed CSV; no number is typed.

The six panels, in the order a reader needs them:

  a  decision_accuracy_vs_mae   headline: accuracy vs the MAE a benchmark would
                                have quoted, with Wilson intervals and the bar
  b  decision_error_distribution  the tail: |error| per model against 1 kcal/mol
  c  error_vs_gap                 where failures live (small gaps should be hardest)
  d  bio_subset                   the peptide / amino-acid slice only
  e  rank_preservation            Spearman between model gap and QM gap
  f  null_control                 shuffled-pair accuracy vs real, the sanity check

If an artefact is missing the panel is skipped with a message rather than
inventing a placeholder: a missing figure must look missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

C_GOOD, C_BAD, C_NEUTRAL = "C0", "C3", "0.55"
BAR = 1.0  # kcal/mol


def _summary() -> pd.DataFrame | None:
    p = RES / "summary.csv"
    if not p.exists():
        print(f"  (skipping: {p.name} missing)")
        return None
    return pd.read_csv(p)


def _footer(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, fontsize=6.5, color="0.35", ha="left", va="bottom")


def _save(fig, name: str, footer: str) -> None:
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _footer(fig, footer)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def fig_accuracy_vs_mae(s: pd.DataFrame):
    """Headline. x = median |error| (what an MAE report would quote, in kcal/mol),
    y = decision accuracy. A model in the top-left is honest; bottom-left is the
    dangerous quadrant: a small error that still ranks things wrongly."""
    d = s[(s["scope"] == "all") & s["median_abs_err_kcal"].notna()].copy()
    if d.empty:
        print("  (skipping accuracy_vs_mae: no 'all' rows)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, r in d.iterrows():
        ax.errorbar(r["median_abs_err_kcal"], r["accuracy"],
                    yerr=[[r["accuracy"] - r["accuracy_lo"]],
                          [r["accuracy_hi"] - r["accuracy"]]],
                    fmt="o", ms=7, capsize=4, lw=1.4,
                    color=C_GOOD if r["accuracy"] >= 0.5 else C_BAD)
        ax.annotate(str(r["model"]).replace("_", " "),
                    (r["median_abs_err_kcal"], r["accuracy"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=7)
    ax.axhline(0.5, ls="--", lw=1, color=C_NEUTRAL)
    ax.text(ax.get_xlim()[1], 0.505, "coin flip", fontsize=7, ha="right",
            color=C_NEUTRAL)
    ax.axvline(BAR, ls=":", lw=1, color="0.75")
    ax.set_xlabel("median |ΔE error| (kcal/mol) — what an MAE report would quote")
    ax.set_ylabel("decision accuracy\n(conformer pairs ordered correctly)")
    ax.set_ylim(0, 1)
    ax.set_title("Does a small energy error mean the chemistry is right?\n"
                 "Vertical axis is the question a design loop asks; the horizontal "
                 "axis is what benchmarks report", fontsize=9.5)
    _save(fig, "decision_accuracy_vs_mae.png",
          "Bars are 95% Wilson intervals. Pairs are conformers of one molecule "
          "with a quantum gap under 1 kcal/mol. Source: results/summary.csv")


def fig_error_distribution(s: pd.DataFrame):
    frames = []
    for p in sorted(RES.glob("decisions_*_all.csv")):
        frames.append(pd.read_csv(p))
    if not frames:
        print("  (skipping error_distribution: no decision CSVs)")
        return
    df = pd.concat(frames)
    df["abs_err"] = df["error_kcal"].abs()
    models = sorted(df["model"].unique())
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    data = [df.loc[df["model"] == m, "abs_err"].clip(upper=6) for m in models]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_GOOD)
        pc.set_alpha(0.45)
    ax.axhline(BAR, ls="--", lw=1.2, color=C_BAD)
    ax.text(0.99, BAR * 1.06, "1 kcal/mol: the bar a decision is made at",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color=C_BAD)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=7)
    ax.set_ylabel("|ΔE error| (kcal/mol, clipped at 6 for display)")
    ax.set_title("The error tail, not the mean\n"
                 "A model can sit under the bar on average and still be wrong "
                 "on the pairs that matter", fontsize=9.5)
    _save(fig, "decision_error_distribution.png",
          "Clipped at 6 kcal/mol for display only; the summary CSV carries the "
          "unclipped p90. Source: results/decisions_*_all.csv")


def fig_error_vs_gap():
    frames = []
    for p in sorted(RES.glob("decisions_*_all.csv")):
        frames.append(pd.read_csv(p))
    if not frames:
        print("  (skipping error_vs_gap: no decision CSVs)")
        return
    df = pd.concat(frames)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m, g in df.groupby("model"):
        g = g.assign(bin=pd.cut(g["qm_gap_kcal"].abs(),
                                bins=[0, .05, .1, .25, .5, 1.0]))
        agg = g.groupby("bin", observed=True).agg(
            gap=("qm_gap_kcal", lambda x: np.abs(x).mean()),
            err=("error_kcal", lambda x: np.median(np.abs(x))))
        ax.plot(agg["gap"], agg["err"], "o-", label=str(m).replace("_", " "),
                ms=5, lw=1.3)
    ax.axhline(BAR, ls="--", lw=1, color=C_NEUTRAL)
    ax.set_xlabel("quantum gap of the pair (kcal/mol, bin mean)")
    ax.set_ylabel("median |ΔE error| (kcal/mol)")
    ax.set_title("Where the failures live\nSmall gaps are chemically harder; a flat "
                 "line means the model is not resolving them at all", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, "error_vs_gap.png",
          "Binned by |QM gap| into 0-0.05, 0.05-0.1, 0.1-0.25, 0.25-0.5, 0.5-1.0 "
          "kcal/mol. Source: results/decisions_*_all.csv")


def fig_bio_subset(s: pd.DataFrame):
    d = s[(s["scope"] == "bio") & s["accuracy"].notna()]
    if d.empty:
        print("  (skipping bio_subset: no bio rows)")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    y = np.arange(len(d))
    ax.barh(y, d["accuracy"], color=C_GOOD, edgecolor="black", linewidth=0.5)
    ax.errorbar(d["accuracy"], y,
                xerr=[d["accuracy"] - d["accuracy_lo"],
                      d["accuracy_hi"] - d["accuracy"]],
                fmt="none", ecolor="black", capsize=3)
    for yi, (acc, n) in enumerate(zip(d["accuracy"], d["n_decided"])):
        ax.text(min(acc + 0.03, 0.95), yi, f"{acc:.2f}  (n={int(n)})",
                va="center", fontsize=8)
    ax.axvline(0.5, ls="--", lw=1, color=C_NEUTRAL)
    ax.set_yticks(y)
    ax.set_yticklabels([str(m).replace("_", " ") for m in d["model"]], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("decision accuracy on the biomolecular slice")
    ax.set_title("Peptides and solvated amino acids only\n"
                 "The slice a protein-engineering reader cares about", fontsize=9.5)
    _save(fig, "bio_subset.png",
          "Bars are 95% Wilson intervals; n is decided pairs, ties excluded. "
          "Source: results/summary.csv (scope=bio)")


def fig_rank_preservation():
    frames = [pd.read_csv(p) for p in sorted(RES.glob("decisions_*_all.csv"))]
    if not frames:
        print("  (skipping rank_preservation: no decision CSVs)")
        return
    df = pd.concat(frames)
    from scipy.stats import spearmanr
    models, rhos, ps = [], [], []
    for m, g in df.groupby("model"):
        if len(g) < 10:
            continue
        rho, p = spearmanr(g["model_gap_kcal"], g["qm_gap_kcal"])
        models.append(str(m).replace("_", " "))
        rhos.append(rho)
        ps.append(p)
    if not models:
        print("  (skipping rank_preservation: too few pairs)")
        return
    order = np.argsort(rhos)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh([models[i] for i in order], [rhos[i] for i in order],
            color=[C_GOOD if rhos[i] > 0 else C_BAD for i in order],
            edgecolor="black", linewidth=0.5)
    for k, i in enumerate(order):
        ax.text(rhos[i] + (0.02 if rhos[i] >= 0 else -0.02), k,
                f"ρ={rhos[i]:.2f} (p={ps[i]:.1e})",
                va="center", ha="left" if rhos[i] >= 0 else "right", fontsize=7)
    ax.axvline(0, lw=1, color="black")
    ax.set_xlabel("Spearman ρ between model ΔE and quantum ΔE")
    ax.set_title("Ranking preservation\nA design loop only needs the ORDER of "
                 "candidates, not their absolute energies", fontsize=9.5)
    ax.set_xlim(min(rhos) - 0.25, max(rhos) + 0.35)
    _save(fig, "rank_preservation.png",
          "Spearman over all decided pairs per model. Source: "
          "results/decisions_*_all.csv")


def fig_null(s: pd.DataFrame):
    d = s[(s["scope"] == "all") & s.get("null_p_value").notna()] \
        if "null_p_value" in s else pd.DataFrame()
    if d.empty:
        print("  (skipping null_control: no null columns)")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["accuracy"], width=0.4, label="real pairing",
           color=C_GOOD, edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, d["null_mean_accuracy"], width=0.4,
           label="shuffled pairing (null)", color=C_NEUTRAL,
           edgecolor="black", linewidth=0.5)
    for xi, p in zip(x, d["null_p_value"]):
        ax.text(xi, 1.01, f"p={p:.3g}", ha="center", fontsize=7)
    ax.axhline(0.5, ls="--", lw=1, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([str(m).replace("_", "\n") for m in d["model"]], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("decision accuracy")
    ax.set_title("The control that makes the numbers believable\n"
                 "Shuffling which model gap meets which quantum gap must destroy "
                 "the signal", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _save(fig, "null_control.png",
          "200 shuffles per model; p is one-sided (shuffles at least as accurate "
          "as the real pairing). Source: results/summary.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    s = _summary()
    jobs = {
        "accuracy_vs_mae": lambda: fig_accuracy_vs_mae(s) if s is not None else None,
        "error_distribution": fig_error_distribution,
        "error_vs_gap": fig_error_vs_gap,
        "bio_subset": lambda: fig_bio_subset(s) if s is not None else None,
        "rank_preservation": fig_rank_preservation,
        "null": lambda: fig_null(s) if s is not None else None,
    }
    names = args.only or list(jobs)
    for n in names:
        if n not in jobs:
            raise SystemExit(f"unknown figure {n}; choose from {list(jobs)}")
        jobs[n]()
    produced = sorted(p.name for p in FIG.glob("*.png"))
    print(f"\nfigures present: {len(produced)} -> {produced}")
    return 0 if len(produced) >= 6 else 1


if __name__ == "__main__":
    sys.exit(main())
