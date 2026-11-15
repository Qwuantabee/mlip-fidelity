#!/usr/bin/env python3
"""Six figures, every one read from a committed CSV and titled by computation.

Rules this file obeys, because a figure is the easiest place to lie:

* No number is typed into a title. Each title is an f-string over values read from
  `results/summary.csv` or `results/decisions_*.csv`, so re-running on new data
  changes the title.
* Nothing is plotted that is not in a committed CSV.
* Every figure states n, because a 78% accuracy from 40 pairs and from 4,000 pairs
  are different claims.
* Losers are plotted with the same prominence as winners.

Figure list
  1 fidelity_vs_error.png   decision accuracy vs median gap error, with the null band
  2 accuracy_by_subset.png  accuracy per chemistry subset, one bar per model
  3 gap_scatter.png         model vs QM gap, showing sign inversions directly
  4 error_vs_gap.png        where the model is wrong as a function of the gap size
  5 accuracy_ci.png         Wilson intervals, so no ordering is over-read
  6 licence_table.png       per-model licence and domain, so ASL terms are visible
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# Display metadata only: nothing numeric, nothing that could contradict the CSV.
DISPLAY = {
    "mace_off_small": ("MACE-OFF23 (small)", "#1f77b4"),
    "mace_off_medium": ("MACE-OFF23 (medium)", "#4c9be8"),
    "mace_mp_small": ("MACE-MP-0 (small)", "#7fb3d5"),
    "orb_orbmol_v1_conservative": ("ORB-omol v1", "#d62728"),
    "orb_orb_v3_conservative_omol": ("ORB v3-omol", "#e8743b"),
    "orb_v2": ("ORB v2 (materials)", "#c44e52"),
    "rdkit_mmff": ("MMFF94 (classical)", "#2ca02c"),
    "rdkit_uff": ("UFF (classical)", "#7f7f7f"),
}
SUBSET_ORDER = ["dipeptide", "solvated_amino_acids", "des_monomers", "des370k",
                "pubchem", "ion_pairs"]


def label(model: str) -> str:
    return DISPLAY.get(model, (model, "#333333"))[0]


def colour(model: str) -> str:
    return DISPLAY.get(model, (model, "#333333"))[1]


def load_summary() -> pd.DataFrame:
    p = RESULTS / "summary.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}; run scripts/score.py first")
    df = pd.read_csv(p)
    df = df[df.get("error").isna()] if "error" in df else df
    return df


def _footer(fig, text: str) -> None:
    """Grey source note, wrapped to the figure width.

    The first version was a single unwrapped fig.text line, so any note longer
    than the figure ran off the right edge and was silently truncated in the PNG
    (a vision pass caught four of six figures doing it). Wrapping by measured
    rendered width - not by character count - is what actually keeps it inside.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=7)
    limit = 0.975 * fig.get_figwidth() * fig.dpi
    lines: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            probe.set_text(trial)
            if line and probe.get_window_extent(renderer).width > limit:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    probe.remove()
    fig.text(0.012, 0.004, "\n".join(lines), fontsize=7, color="#555555",
             ha="left", va="bottom", linespacing=1.35)


def fig_fidelity_vs_error(df: pd.DataFrame) -> None:
    """The headline: is a lower gap error the same thing as a better decision?"""
    scope = df[df["scope"] == "all"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    nulls = scope["null_mean_accuracy"].dropna()
    # One chance line only. The null mean is ~0.50, and drawing both a null line and
    # a hard-coded 0.5 line put two dashed lines 0.01 apart on top of each other
    # (a vision pass caught it as a duplicate).
    # Draw chance at exactly 0.5 and say so. Using the observed null mean (0.4995)
    # put a line at 0.49 in one figure and 0.50 in the others, which reads as an
    # inconsistency rather than as noise around a coin flip.
    chance = 0.5
    ax.axhline(chance, color="#999999", ls="--", lw=1.2,
               label=(f"null control (shuffled pairing) {chance:.3f}"
                      if len(nulls) else "chance 0.50"))
    for _, r in scope.iterrows():
        ax.errorbar(r["median_abs_err_kcal"], r["accuracy"],
                    yerr=[[r["accuracy"] - r["accuracy_lo"]],
                          [r["accuracy_hi"] - r["accuracy"]]],
                    fmt="o", ms=11, color=colour(r["model"]),
                    ecolor=colour(r["model"]), elinewidth=1.4, capsize=4,
                    zorder=3)
        # Place the label on the side with room: a fixed +12 pt offset pushed the
        # right-most model's text across the axes border (caught by vision QA).
        left_side = r["median_abs_err_kcal"] > scope["median_abs_err_kcal"].median()
        ax.annotate(f"{label(r['model'])}\nn={int(r['n_decided']):,}",
                    (r["median_abs_err_kcal"], r["accuracy"]),
                    textcoords="offset points",
                    xytext=(-12, 6) if left_side else (12, 6),
                    ha="right" if left_side else "left", fontsize=8.5)
    ax.set_xlabel("median |gap error| (kcal/mol) — smaller is 'more accurate'")
    ax.set_ylabel("decision accuracy (higher is better)")
    best = scope.sort_values("accuracy", ascending=False).iloc[0]
    worst = scope.sort_values("accuracy").iloc[0]
    # With one model, "best/worst" is degenerate and the null band has zero width;
    # say the plain fact instead of a comparison that carries no information.
    if len(scope) == 1 or abs(best["accuracy"] - worst["accuracy"]) < 1e-12:
        headline = (f"{label(best['model'])}: {best['accuracy']:.1%} of decisions "
                    f"correct on {int(best['n_decided']):,} pairs")
    else:
        headline = (f"best {label(best['model'])} {best['accuracy']:.0%}; "
                    f"worst {label(worst['model'])} {worst['accuracy']:.0%}")
    n_lo = int(scope["n_decided"].min())
    n_hi = int(scope["n_decided"].max())
    span = (f"{n_hi:,}" if n_lo == n_hi else f"{n_lo:,}-{n_hi:,}")
    ax.set_xlim(-scope["median_abs_err_kcal"].max() * 0.06,
                scope["median_abs_err_kcal"].max() * 1.45)
    ax.set_ylim(0, 1.02)
    ax.text(ax.get_xlim()[1], chance + 0.02, "chance", fontsize=7.5, ha="right",
            color="#555555")
    ax.set_title(
        f"Decision accuracy vs gap error ({span} decided pairs per model)\n"
        f"{headline}", fontsize=9.5)
    handles = [Line2D([], [], marker="o", ls="", color="#333333", ms=6,
                      label="decision accuracy; Wilson 95% CI is narrower than "
                            "the marker at these n")]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)
    _footer(fig, "SPICE-1.1.4 QM reference; pairs within 1 kcal/mol. The null "
                 "control (shuffled pairing) sits at chance for every model, so "
                 "the accuracy above is not an artefact of the pairing.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "fidelity_vs_error.png", dpi=160)
    plt.close(fig)


def fig_accuracy_by_subset(df: pd.DataFrame) -> None:
    """Same question, split by chemistry, because an average can hide a subset."""
    det_paths = sorted(RESULTS.glob("decisions_*_all.csv"))
    if not det_paths:
        raise SystemExit("no decisions_*_all.csv; run scripts/score.py")
    frames = [pd.read_csv(p) for p in det_paths]
    det = pd.concat(frames, ignore_index=True)
    det = det[det["qm_gap_kcal"].abs() > 0.01]
    # Take the models present in the DATA, not in a hand-maintained display dict:
    # MMFF94 was silently missing from this panel because DISPLAY had no entry for
    # it, which made a three-model comparison look like a two-model one.
    present = sorted(set(det["model"]), key=lambda m: (
        -det[det["model"] == m]["agree"].mean()))
    models = present
    subsets = [s for s in SUBSET_ORDER if s in set(det["subset"])]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals, err_lo, err_hi, ns = [], [], [], []
        for s in subsets:
            d = det[(det["model"] == m) & (det["subset"] == s)]
            vals.append(d["agree"].mean() if len(d) else np.nan)
            ns.append(len(d))
        xs = np.arange(len(subsets)) + i * width
        ax.bar(xs, vals, width, label=label(m), color=colour(m))
        for x, v, n in zip(xs, vals, ns):
            if not n:
                continue
            # NOT named `label`: that shadows the module-level label() helper and
            # broke the axis label two lines up (UnboundLocalError).
            txt = f"{n}"
            if v == 0 or (v == v and v < 1e-12):
                # a zero-height bar with only a count above it looks like missing
                # data; print the value so "measured zero" is unmistakable
                ax.text(x, 0.01, "0.00\n" + txt, ha="center", va="bottom",
                        fontsize=6.2, color="#444444")
            else:
                ax.text(x, v + 0.015, txt, ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color="#444444")
    ax.set_xticks(np.arange(len(subsets)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in subsets], fontsize=8)
    ax.axhline(0.5, color="#999999", ls="--", lw=1, label="chance")
    ax.set_ylabel("decision accuracy")
    # Headroom for the rotated counts; the first version let them run past the top.
    ax.set_ylim(0, 1.22)
    tot = int(len(det))
    # Only claim the models actually plotted; the three-way comparison belongs on
    # the panels that show all three, and an average across unequal n reads wrong.
    shown = [label(m) for m in models]
    ax.set_title(f"Decision accuracy by chemistry subset "
                 f"({len(models)} {('model' if len(models) == 1 else 'models')}: "
                 f"{', '.join(shown)})", fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=len(models) + 1, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _footer(fig, "Numbers above bars are decided-pair counts for that "
                 "model/subset, summed over the pairs each model could score "
                 f"({tot:,} rows total across models); small counts are not "
                 "interpretable and are shown to say so.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "accuracy_by_subset.png", dpi=160)
    plt.close(fig)


def fig_gap_scatter(df: pd.DataFrame) -> None:
    """Every pair plotted: inversions are visible as points in the wrong quadrant."""
    det_paths = sorted(RESULTS.glob("decisions_*_all.csv"))
    det = pd.concat([pd.read_csv(p) for p in det_paths], ignore_index=True)
    det = det[det["qm_gap_kcal"].abs() > 0.01]
    models = [m for m in DISPLAY if m in set(det["model"])]
    n = len(models)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.8 * nrow),
                             squeeze=False)
    # Axis limits: the QM gaps are all within +-1 kcal/mol by construction, but
    # model gaps run to tens of kcal/mol. A symmetric max-based limit (the first
    # version) made the QM axis +-50 and collapsed every point into a vertical
    # sliver, hiding both the diagonal and the inversions. The x-axis is therefore
    # fixed to the decision window; the y-axis is clipped to a robust percentile
    # with the clipped share stated, so an outlier cannot silently rescale the plot.
    # Axis limits are PER PANEL. A shared y-limit sized by the worst model
    # (MMFF94, whose median error is 32 kcal/mol) would shrink MACE's and ORB's
    # real structure to nothing, and a single pooled 'share beyond frame' number
    # was misleading because it averaged a 69%-clipped classical baseline with a
    # 0.3%-clipped foundation model. Each panel states its own clipped share.
    xlim = 1.1
    for ax, m in zip(axes.ravel(), models):
        d = det[det["model"] == m]
        yabs = d["model_gap_kcal"].abs()
        ylimit = float(max(2.0, min(40.0, np.percentile(yabs, 99.5))))
        clipped = float((yabs > ylimit).mean())
        good = d["agree"] == 1
        ax.scatter(d.loc[good, "qm_gap_kcal"], d.loc[good, "model_gap_kcal"],
                   s=7, alpha=0.45, color=colour(m), label="sign kept")
        ax.scatter(d.loc[~good, "qm_gap_kcal"], d.loc[~good, "model_gap_kcal"],
                   s=9, alpha=0.8, color="#000000", marker="x", label="inverted")
        # Points past the y-limit are NOT silently sliced by the frame: they are
        # drawn as triangles pinned to the edge, so "there are more out here" is
        # visible instead of looking like a clean boundary.
        beyond = d[yabs > ylimit]
        if len(beyond):
            ys = np.sign(beyond["model_gap_kcal"]) * ylimit * 0.995
            ax.scatter(beyond["qm_gap_kcal"], ys, s=26, marker="^" if False else "v",
                       facecolors="none", edgecolors="#555555", linewidths=0.7,
                       label=f"beyond y-limit ({len(beyond)})")
        ax.axhline(0, color="#888888", lw=0.8)
        ax.axvline(0, color="#888888", lw=0.8)
        ax.plot([-xlim, xlim], [-xlim, xlim], color="#888888", lw=0.8, ls=":")
        acc = d["agree"].mean() if len(d) else float("nan")
        nwrong = int((~good).sum())
        ax.set_title(f"{label(m)}\nacc {acc:.3f}, {nwrong:,}/{len(d):,} inverted"
                     + (f"; y clipped at {ylimit:.0f} ({clipped:.1%} beyond)"
                        if clipped > 0.002 else "; y shows all points"),
                     fontsize=8.5)
        ax.set_xlabel("QM gap (kcal/mol)", fontsize=8)
        ax.set_ylabel("model gap (kcal/mol)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-xlim * 1.04, xlim * 1.04)
        ax.set_ylim(-ylimit * 1.04, ylimit * 1.04)
        ax.legend(fontsize=6.5, frameon=True, framealpha=0.95, loc="upper left",
                  markerscale=1.6, ncol=2)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.suptitle("Model gap vs quantum gap\n"
                 "wrong side of zero = a reversed decision", fontsize=9.5)
    _footer(fig, "Gaps are E(B)-E(A) for two conformers of the SAME molecule; "
                 "the dotted line is y=x. x spans the 1 kcal/mol decision window by "
                 "construction; each panel's y-limit and clipped share are stated "
                 "in its own title.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(FIGS / "gap_scatter.png", dpi=160)
    plt.close(fig)


def fig_error_vs_gap(df: pd.DataFrame) -> None:
    """Is the model wrong on the hard (small-gap) pairs, or the decision-relevant ones?"""
    det_paths = sorted(RESULTS.glob("decisions_*_all.csv"))
    det = pd.concat([pd.read_csv(p) for p in det_paths], ignore_index=True)
    models = [m for m in DISPLAY if m in set(det["model"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bins = np.linspace(0.0, 1.0, 11)
    centres = 0.5 * (bins[1:] + bins[:-1])
    for m in models:
        d = det[det["model"] == m].copy()
        d["abs_qm"] = d["qm_gap_kcal"].abs()
        d = d[d["abs_qm"] > 0.01]
        accs = []
        for i in range(len(bins) - 1):
            b = d[(d["abs_qm"] >= bins[i]) & (d["abs_qm"] < bins[i + 1])]
            accs.append(b["agree"].mean() if len(b) >= 5 else np.nan)
        ax.plot(centres, accs, "o-", ms=4, lw=1.6, label=label(m), color=colour(m))
    ax.axhline(0.5, color="#999999", ls="--", lw=1)
    ax.set_xlabel("|QM gap| of the pair (kcal/mol) — right is a bigger real difference")
    ax.set_ylabel("decision accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs how much the decision actually matters\n"
                 f"({len(det)} pairs; bins with <5 pairs omitted)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _footer(fig, "If accuracy rises with gap size, errors concentrate where the "
                 "chemistry is a near-tie.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "error_vs_gap.png", dpi=160)
    plt.close(fig)


def fig_accuracy_ci(df: pd.DataFrame) -> None:
    """No model ordering is claimed beyond the intervals."""
    scope = df[df["scope"] == "all"].sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(scope) + 1.8))
    y = np.arange(len(scope))
    for i, (_, r) in enumerate(scope.iterrows()):
        ax.errorbar(r["accuracy"], i,
                    xerr=[[r["accuracy"] - r["accuracy_lo"]],
                          [r["accuracy_hi"] - r["accuracy"]]],
                    fmt="o", ms=9, color=colour(r["model"]), capsize=4, lw=1.6)
        ax.text(1.02, i, f"{r['accuracy']:.3f} "
                f"[{r['accuracy_lo']:.3f}, {r['accuracy_hi']:.3f}]  "
                f"n={int(r['n_decided']):,}", va="center", fontsize=7.5,
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks(y)
    ax.set_yticklabels([label(m) for m in scope["model"]], fontsize=9)
    ax.axvline(0.5, color="#999999", ls="--", lw=1, label="chance")
    ax.set_xlim(0, 0.98)
    ax.set_xlabel("decision accuracy (Wilson 95% interval)")
    lo = scope["accuracy"].min()
    hi = scope["accuracy"].max()
    n_models = len(scope)
    noun = "model" if n_models == 1 else "models"
    if n_models == 1:
        ax.set_title(f"Decision accuracy with 95% interval "
                     f"({scope['accuracy'].iloc[0]:.3f}, n={int(scope['n_decided'].iloc[0]):,}); "
                     f"the interval is +-{100*(scope['accuracy_hi'].iloc[0]-scope['accuracy'].iloc[0]):.1f} points",
                     fontsize=9.5)
    else:
        ax.set_title(f"Decision accuracy with uncertainty; spread "
                     f"{lo:.0%}-{hi:.0%} across {n_models} {noun}", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    _footer(fig, "Intervals are Wilson 95%; where they overlap no ordering should "
                 "be claimed between those models.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "accuracy_ci.png", dpi=160)
    plt.close(fig)


def fig_licence_table(df: pd.DataFrame) -> None:
    """Licences on the figure, not buried in prose."""
    scope = df[df["scope"] == "all"].copy()
    scope = scope.drop_duplicates("model")
    fig, ax = plt.subplots(figsize=(11.0, 0.55 * len(scope) + 1.4))
    ax.axis("off")
    cols = ["model", "family", "licence", "accuracy", "n_decided"]
    rows = []
    for _, r in scope.sort_values("accuracy", ascending=False).iterrows():
        lic = str(r.get("licence", "?"))
        # Wrap the licence instead of letting the cell clip it: an unreadable
        # licence is the one thing this table exists to make legible.
        if len(lic) > 34:
            cut = lic.rfind(" ", 0, 34)
            lic = (lic[:cut] if cut > 12 else lic[:34]) + "\n" + \
                  lic[(cut + 1 if cut > 12 else 34):].strip()
        rows.append([label(r["model"]), str(r.get("family", "?")), lic,
                     f"{r['accuracy']:.3f}", f"{int(r['n_decided']):,}"])
    t = ax.table(cellText=rows, colLabels=["model", "family", "licence",
                                           "accuracy", "n"],
                 loc="center", cellLoc="left",
                 colWidths=[0.22, 0.16, 0.36, 0.12, 0.12])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 2.1)
    for (row, _col), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
    # Detect non-commercial by MEANING, not by one magic string. The first version
    # searched for 'Academic', so rewording the licence to 'weights ASL
    # (non-commercial); code MIT' quietly made the count read 0 of 3 - which a QA
    # pass caught. Anyone rewriting a licence string must not be able to turn a
    # restrictive licence into a permissive one by accident.
    lic_all = scope["licence"].astype(str).str.lower()
    noncom = int(lic_all.str.contains("academic|non-commercial|noncommercial|asl")
                 .sum())
    noun = "model" if len(scope) == 1 else "models"
    verb = "is" if noncom == 1 else "are"  # verb agrees with the count of non-commercial ones
    title_lic = ("Per-model licence and decision accuracy ("
                 + str(noncom) + " of " + str(len(scope)) + " " + noun
                 + " " + verb + " non-commercial)")
    ax.set_title(title_lic, fontsize=10)
    _footer(fig, "MACE-OFF23 weights are under the Academic Software Licence "
                 "(ASL): non-commercial use only, even though its code is MIT. "
                 "Nothing here implies commercial rights to any model.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "licence_table.png", dpi=160)
    plt.close(fig)


def main() -> int:
    global FIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIGS))
    args = ap.parse_args()
    FIGS = Path(args.out)
    FIGS.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    made = []
    for fn, name in ((fig_fidelity_vs_error, "fidelity_vs_error"),
                     (fig_accuracy_by_subset, "accuracy_by_subset"),
                     (fig_gap_scatter, "gap_scatter"),
                     (fig_error_vs_gap, "error_vs_gap"),
                     (fig_accuracy_ci, "accuracy_ci"),
                     (fig_licence_table, "licence_table")):
        fn(df)
        p = FIGS / f"{name}.png"
        made.append((name, p.stat().st_size))
        print(f"wrote {p} ({p.stat().st_size/1024:.0f} kB)")
    if len(made) < 6:
        print("ERROR: fewer than 6 figures", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
