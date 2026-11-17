# Do machine-learned potentials preserve the chemical decisions?

[![tests](https://github.com/Qwuantabee/mlip-fidelity/actions/workflows/tests.yml/badge.svg)](https://github.com/Qwuantabee/mlip-fidelity/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Qwuantabee/mlip-fidelity/blob/master/LICENSE)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://github.com/Qwuantabee/mlip-fidelity/blob/master/pyproject.toml)

**Three energy models were asked, 16,089 times, which of two near-degenerate conformers is lower in energy (quantum reference gap under 1 kcal/mol). ORB-omol got 90.0% right, MACE-OFF23 76.0% and the classical MMFF94 force field 60.1%; always guessing the same way scores 53.4%. Accuracy falls as a model's pair-gap error rises (median 0.15 / 0.45 / 1.61 kcal/mol), so an energy error only means something next to the gap it has to resolve, not as a leaderboard MAE.**

**The caveat that governs those numbers: ORB-omol and MACE-OFF23 were both trained on the dataset the pairs come from (SPICE), so two of the three figures are in-sample.** On a second dataset with a higher-level reference (Folmsbee, 22,379 pairs) the ordering is the same, 0.833 / 0.773 / 0.622, but that test does not isolate how much of the in-sample figures is memorisation; see [the held-out test](#the-held-out-test-does-any-of-this-survive-outside-the-training-data).

![Decision accuracy vs gap error](figures/fidelity_vs_error.png)

Universal machine-learned interatomic potentials (uMLIPs) are ranked, in every
2026 benchmark, by energy MAE in meV/atom. Nobody downstream of those benchmarks
consumes an energy MAE. A chemist consumes a **decision**: which of these two
conformers is lower, does this reaction go forward, which tautomer wins. Those are
questions about *differences*, at roughly a 1 kcal/mol bar.

That bar is **not** orders of magnitude coarser than a leaderboard number — it is
the same size, because the two are different quantities. 1 kcal/mol is 43 meV per
molecule, which at the median 36 atoms in this set is **1.2 meV/atom**: the same
order as the ~1 meV/atom MAEs these models advertise. The point is not that the
bar is coarser; it is that an average error and a resolution threshold are not
interchangeable, and only one of them tells you whether two candidates swap.

So this repo asks it directly. Take **16,089 conformer pairs** from the SPICE
quantum-chemistry dataset — two geometries of the *same molecule* whose reference
energies differ by **less than 1 kcal/mol** — and ask each model: *which one is
lower?* Score the sign. Compare that to the model's own energy error.

## The result

**Read this table as an in-sample measurement.** Two of the three models trained
on SPICE — MACE-OFF23 on ~95% of v1, ORB-omol via OMol25's documented SPICE2 — so
their rows are fidelity to their own training data, not generalisation. Only
MMFF94 is a clean row.

| model | licence | scored conformers | decided pairs | **decision accuracy** (95% CI) | median \|ΔE\| error | null control p |
|---|---|---|---|---|---|---|
| **ORB-omol v1 (conservative)** | Apache-2.0 | 15,232 | 15,991 | **0.900** (0.895–0.905) | 0.15 kcal/mol | < 0.005 |
| MACE-OFF23 (small) | weights ASL, **non-commercial**; code MIT | 14,913 | 15,717 | **0.760** (0.753–0.766) | 0.45 kcal/mol | < 0.005 |
| MMFF94 (classical force field) | BSD | 14,763 | 15,578 | 0.601 (0.594–0.609) | 1.61 kcal/mol | < 0.005 |

How to read the columns:

- **Ties are excluded, and counted.** A pair whose reference gap is within 0.01
  kcal/mol is a tie and is not scored: 98 pairs for ORB-omol and MACE-OFF23, 97
  for MMFF94. A pair containing a conformer the model could not evaluate is not
  scored either: 0 for ORB-omol, 274 for MACE-OFF23, 414 for MMFF94. "Decided
  pairs" is what is left of the 16,089. A model gap of exactly zero is *not* a
  tie: it is scored as wrong (121 ORB-omol pairs, 82 MACE-OFF23; see Limits).
- **The intervals** are Wilson intervals over pairs. Pairs from one molecule are
  not independent, so for ORB-omol and MACE-OFF23 they are about 1.4x too narrow
  (see Limits). The 1,588 molecules are the independent units.
- **The null-control p** is the floor 200 shuffles can report (1/201), not a
  measured value.
- **Run-to-run noise.** ORB-omol's GPU energies are not bit-reproducible. On the
  held-out set, where it was measured, re-running moved ORB-omol's accuracy by
  0.001 to 0.004 (details in the held-out section); it was not re-measured on
  SPICE. Every gap between models in this table is 0.14 or more, about forty
  times the larger of those figures. MACE-OFF23 flipped no decision across re-runs and MMFF94, on the
  CPU, was bit-identical.
- **Out of sample** the order is unchanged (ORB > MACE > MMFF94) and ORB's lead
  over MACE is 6 points instead of 14; the held-out section says what that does
  and does not show.

All three rows are recomputed from the committed `results/decisions_*.csv` by
`scripts/score.py`, and a model only appears here if
`scripts/verify_completeness.py` confirms its raw artefact covers every one of the
15,232 conformers exactly once.

Four things worth taking from that table.

1. **A small average error is not a resolution guarantee.** MACE-OFF23's median
   error is 0.45 kcal/mol, and the pairs it is being asked about have a median
   true gap of only 0.55 kcal/mol (5th-95th percentile: 0.09-0.95). An error the
   size of the gap means a coin flip on the small ones, which is why it reverses
   24%.

2. **The best model is still wrong one time in ten.** ORB-omol's median error is
   0.15 kcal/mol and 0.900 accuracy means 1,596 wrong orderings out of 15,991. If
   your design loop ranks hundreds of candidates per run, that is a lot of
   swapped pairs.

3. **Accuracy tracks error, and the ordering is unambiguous.** With the classical
   baseline's units corrected (see [the defect trail](docs/DEFECT_TRAIL.md)), the three models are
   monotone in *every* error measure: median gap error 0.15 / 0.45 / 1.61 kcal/mol
   gives accuracy 0.900 / 0.760 / 0.601. Pairwise differences are robust to
   clustering by molecule — on the 15,578 pairs all three models scored, with a
   bootstrap over their 1,530 molecules, ORB−MACE +0.144 [0.135, 0.152] and
   MACE−MMFF +0.161 [0.150, 0.171] — so this is not a tie. **I do not fit a trend line:** three
   points cannot distinguish a curve from a coincidence, and an earlier draft of
   this README claimed the two were independent, which the corrected units
   refuted.

4. **The biomolecular slice is consistently harder.** Every model drops on
   peptides and solvated amino acids: ORB 0.900 → **0.855**, MACE 0.760 → **0.647**,
   MMFF94 0.601 → 0.552. See
   [`figures/accuracy_by_subset.png`](figures/accuracy_by_subset.png). Part of that
   is composition, not chemistry — see Limits.

![Accuracy with confidence intervals](figures/accuracy_ci.png)

### The baseline that makes MMFF's number honest

Pair orientation is arbitrary (whichever conformer happened to be index A), and
the QM gaps are not symmetric: **always guessing "B is higher" scores 0.534** on
these pairs. Against that constant, scored on each model's own pairs (where it
reads 0.534 for all three), the excess over guessing is ORB **+0.366**,
MACE **+0.226**, and MMFF **+0.067**.

MMFF94's edge is small but not noise: it beats the constant predictor on 4,125
pairs and loses on 3,079 (McNemar-style sign test p ≈ 6e-35). So the classical
force field does carry real signal on sub-kcal orderings — it just carries an
order of magnitude less of it than either ML potential. The earlier version of
this paragraph said "barely above a coin flip" against its pre-fix 0.563; that was
both a stale number and the wrong conclusion, and I am correcting it rather than
quietly deleting it.

## The held-out test: does any of this survive outside the training data?

A reviewer established that two of the three models train on the data I scored
them on — MACE-OFF23 on ~95% of SPICE v1, ORB-omol's OMol25 documenting SPICE2.
So the table above measures **fidelity to their own training data**. Only MMFF94
was ever a clean row.

So I scored all three on two sets from outside SPICE. "Held out" here means a
different dataset: whether individual Folmsbee or MPCONF196 molecules also occur
in SPICE or in OMol25 was not checked. Both are compared on the slice where they
overlap with SPICE: reference gaps under 1 kcal/mol.

**What this test can and cannot say, up front.** It shows the ranking of the
three models survives a change of dataset and of reference method. It does *not*
separate training-set contamination from difficulty. The only uncontaminated
model, MMFF94, is used as the difficulty control, and it is a poor one: its
accuracy barely depends on the size of the gap, while both ML models' accuracy
depends on it steeply, and Folmsbee's pairs are much closer to ties (median
reference gap 0.24 kcal/mol against SPICE's 0.55). At equal gap size all three
models, ORB-omol included, score *higher* on Folmsbee than on SPICE. No
inflation from contamination is demonstrated for either ML model.

| benchmark | compounds | sub-kcal pairs | reference |
|---|---|---|---|
| **Folmsbee & Hutchison 2021** | 690 | **22,379** | DLPNO-CCSD(T) |
| MPCONF196 (Řezáč 2018) | 7 | 30 | PNO-LCCSD(T)-F12 |

### The result, and a correction

![In-sample vs held out](figures/heldout.png)

*Left: accuracy on each dataset. Right: each model's shift minus MMFF94's. The
right panel is an estimate under an assumption that does not hold (see "Accuracy
at equal gap size"); it is not a measured contamination effect.*

| model | SPICE (in-sample) | **Folmsbee held out** | shift | shift minus MMFF94's | Folmsbee, ties excluded (decided pairs) |
|---|---|---|---|---|---|
| ORB-omol v1 | 0.900 | **0.833** (0.828–0.837) | −0.068 | −0.089 | 0.840 (21,812) |
| MACE-OFF23 | 0.760 | **0.773** (0.768–0.779) | +0.013 | −0.008 | 0.780 (21,812) |
| MMFF94 | 0.601 | **0.622** (0.616–0.629) | +0.021 | — (control) | 0.624 (21,377) |

The bold column scores every sub-kcal Folmsbee pair, near-ties included, while
the SPICE column excludes ties, so the "shift" columns mix two definitions. The
last column is the like-for-like figure (567 ties excluded for the ML models, 553
for MMFF94); on that basis the shifts are −0.060 / +0.020 / +0.023. The "shift
minus MMFF94's" column is a difference of raw shifts, not a measured
contamination effect: see the gap-size table below before quoting it.

**Correction (review R2, 2026-11-16).** The first version of this table ran the
74 charged Folmsbee compounds (1,943 of the 22,379 pairs; +1: 60, −1: 10, +2: 4)
as neutral singlets: the builder hard-coded charge 0 although the dataset lists
each ion's charge in `geometries/CHG-charges.txt`. With the true charges ORB-omol,
which takes the charge as input, goes from 0.668 to 0.794 on those compounds'
1,887 decided pairs and from 0.823 to 0.833 overall, a step that includes the
run-to-run noise measured below; MMFF94 can now perceive
bonds for most of the ions and scores 21,930 pairs instead of 20,169 (0.625 →
0.622); MACE-OFF23 takes no charge input and is unchanged. No compound had to be
excluded for want of a charge.

**Run-to-run noise, measured** (`scripts/orb_repeatability.py` →
`results/orb_repeatability.json`). ML potential energies are not
bit-reproducible on the GPU. Re-running each model three times
on the same 607 conformers (63 seeded-random compounds, 2,217 pairs), in fresh
processes with a byte-identical input, moved all 607 of ORB-omol's energies
(largest |ΔE| 0.021 kcal/mol, median 0.0035), flipped up to 16 decisions between
two runs (0.72%) and spread its sample accuracy over 0.836–0.839 (0.0036).
MACE-OFF23 moved 41 energies by at most 1.7e-09 kcal/mol and flipped none;
MMFF94, the CPU control, was bit-identical, so the noise is the models', not the
harness's. On the full set the two committed ORB-omol runs saw identical neutral
inputs and disagree on 187 of those 20,436 decisions (0.92%; largest gap change
0.040 kcal/mol): a net 27 decisions, 0.0012 of the 22,379-pair accuracy, which is
why a predicted 0.834 came out 0.833. **So ORB-omol's 0.833 carries an error bar
of about ±0.0012** (one pair of full-set runs, so itself an estimate; the
three-run sample above spread 0.0036 on a tenth of the pairs). The 0.823 → 0.833 step (+0.0095, 7.9× that) survives it: it
is +0.0107 from the charged pairs, whose inputs changed (a net 240 decisions,
where noise could move at most 19), less 0.0012 of noise on the neutral ones.

The intervals in the table are Wilson intervals, which treat the ~32 pairs per
compound as independent. They are not: resampling whole compounds (cluster
bootstrap, 4,000 draws) widens them to ORB [0.823, 0.841], MACE [0.761, 0.784]
and MMFF94 [0.607, 0.637], a design effect of 3.4–5.2. The Folmsbee column also
counts every sub-kcal pair, including 567 near-ties (|gap| ≤ 0.01 kcal/mol; 553
for MMFF94) that the SPICE column excludes; with ties excluded the held-out
figures are 0.840 / 0.780 / 0.624.

#### Accuracy at equal gap size

Decision accuracy by size of the reference gap, SPICE (in-sample) against
Folmsbee, from `scripts/reconcile_heldout.py` (block H3; ties included in the
first bin):

| reference gap (kcal/mol) | ORB-omol SPICE → Folmsbee | MACE-OFF23 SPICE → Folmsbee | MMFF94 SPICE → Folmsbee |
|---|---|---|---|
| 0 – 0.1 | 0.595 → 0.623 | 0.533 → 0.595 | 0.513 → 0.537 |
| 0.1 – 0.25 | 0.764 → 0.821 | 0.623 → 0.769 | 0.515 → 0.606 |
| 0.25 – 0.5 | 0.894 → 0.924 | 0.733 → 0.843 | 0.595 → 0.661 |
| 0.5 – 0.75 | 0.938 → 0.963 | 0.800 → 0.888 | 0.613 → 0.699 |
| 0.75 – 1.0 | 0.980 → 0.983 | 0.845 → 0.902 | 0.648 → 0.677 |

Every model is more accurate on Folmsbee in every bin. ORB-omol's lower overall
figure (0.833 against 0.900) is therefore a composition effect: 5,488 of
Folmsbee's 22,379 pairs (24.5%) have a gap under 0.1 kcal/mol, against 922 of
SPICE's 16,089 (5.7%), and those are the pairs every model gets wrong most
often. Reweighting Folmsbee's five bins to SPICE's gap distribution (ties
excluded on both sides, as in the SPICE table) gives ORB-omol 0.924, MACE-OFF23
0.851 and MMFF94 0.663, which is +0.024 / +0.091 / +0.062 against their SPICE
figures; `tests/test_readme_secondary_claims.py` recomputes it. Why Folmsbee is easier at equal gap is not
established; it differs from SPICE in at least three ways at once (relaxed minima
instead of MD snapshots, a DLPNO-CCSD(T) reference instead of the DFT level the
models were fitted near, and no ion pairs or dipeptides).

#### The MMFF94-offset estimate, and why it is not a contamination measurement

MMFF94 has never seen either dataset, so one can take its +0.021 as an estimate
of how much easier the held-out slice is, and subtract it from each ML model's
raw shift. That treats slice difficulty as the same for every model, which the
gap-size table shows is false. Intervals resample SPICE molecules and Folmsbee
compounds independently (4,000 draws).

- **ORB-omol: −0.089** (95% CI −0.107 to −0.070) on that estimate. An earlier
  version of this README read it as "ORB-omol loses about 9 points beyond
  difficulty, consistent with in-sample inflation". That reading is withdrawn:
  the interval covers sampling error only, and the sign of the effect reverses
  once gap size is matched (+0.024 on its own, −0.038 relative to MMFF94's
  gap-matched +0.062). The data are consistent with some inflation and with none.
- **MACE-OFF23: −0.008** (95% CI −0.027 to +0.012); gap-matched it is +0.091.
  No inflation detected, which is absence of evidence, not evidence of absence.

The offset is the weak link for both models, for the same reasons: Folmsbee's
sub-kcal pairs sit much closer to ties than SPICE's, ML accuracy rises steeply
with the gap while MMFF94's is nearly flat, MMFF94 scores a different pair set,
and the reference level changes from DFT (ωB97M-D3(BJ) for SPICE and for
MACE-OFF23's training labels) to DLPNO-CCSD(T). Matching on gap size, an
equally defensible specification, moves both figures by more than their
intervals (ORB-omol −0.089 → −0.038, MACE-OFF23 −0.008 → +0.029, each relative
to MMFF94), so neither is identified. An
experiment that could identify it needs the same molecules scored by a model
before and after they enter its training set, which this repo does not have.

**The MPCONF196 result I published earlier is withdrawn.** I reported that
MACE's sub-kcal accuracy "collapses out of sample, 0.760 → 0.467, p = 0.0008."
That p-value was computed over 30 pairs as if they were 30 independent
observations. They are not: the 30 pairs come from **7 compounds, 26 of them from
3 compounds** (FGG114 10, GGF01 9, GFA01 7). A cluster bootstrap over compounds gives a 95% interval of
[0.14, 0.71] — consistent with almost anything. At n = 30 that benchmark cannot
distinguish a real drop from noise, and the well-powered set contradicts it. The
number is kept here, labelled, so it cannot be misquoted again.

Why the two differ is not fully resolved, but an earlier version of this
paragraph called those three compounds macrocycles, and they are not: their
formulas (C13H17N3O4, C13H17N3O4, C14H19N3O4) are those of linear tripeptides (a
cyclic one would have lost another H2O; I checked formulas, not geometries). That
points to composition rather than size. MACE-OFF23's own in-sample accuracy on
SPICE dipeptides is 0.648 (6,510 pairs), its weakest large subset, so these 30
pairs probe its known peptide weakness. MPCONF196 as a whole is larger (median 32
heavy atoms, 15 conformers larger than anything in Folmsbee, whose median is 22),
but a size effect is not needed to explain these 30 pairs.

### What this leaves standing

On 22,379 sub-kcal pairs from a dataset outside SPICE, **ORB-omol still orders near-degenerate conformers at 0.833 against MMFF94's
0.622** — an ML potential does beat a force field outside SPICE, by about 21
points on Folmsbee's own mix of gap sizes against 30 in-sample. Most of that
difference is the gap mix rather than the dataset: at SPICE's gap distribution
the Folmsbee margin is 26 points. That the ranking ORB > MACE > MMFF94 holds on
both datasets is the defensible claim. How much of either ML model's in-sample
accuracy is memorisation is not measured here.

## Why the null control matters

Every accuracy above is checked against the answer-leaking failure mode: shuffle
which model gap is paired with which quantum gap, 200 times, and rescore. If the
pipeline were leaking, shuffled pairs would score as well as real ones. Shuffled
accuracy collapses to 0.498–0.502 for every model, and no shuffle reached the
real accuracy in any of the six rows: p < 0.005, where 1/201 ≈ 0.005 is the
smallest p-value 200 shuffles can report — a floor, not a measurement. What the
shuffle shows is that the labels carry signal and the pipeline does not leak the
answer. It does not by itself make 0.900 right; the raw-energy recomputation in
`scripts/check_artefacts.py` and the held-out test carry more of that weight.

## Where the failures live

Accuracy rises with the size of the true gap: near-ties are close to coin flips
and clear preferences are mostly right. That shape is what a model resolving
*some* chemistry looks like, as opposed to one that is guessing:

![Error vs gap](figures/error_vs_gap.png)

Every pair, plotted. A point on the wrong side of zero is a reversed decision.
The triangles pinned to the frame edges are points whose model gap exceeded the
plotted range — counted rather than hidden:

![Model gap vs quantum gap](figures/gap_scatter.png)

![Licences](figures/licence_table.png)

## How the data is built

- **Reference:** [SPICE 1.1.4](https://zenodo.org/records/8222043)
  (ωB97M-D3(BJ)/def2-TZVPPD), 19,238 molecules. Downloaded and byte-verified against its
  Content-Length (16,058,156,944 bytes).
- **Pairs:** every conformer pair of a molecule whose QM energies differ by under
  1 kcal/mol, capped per molecule and selected deterministically from a seed:
  **16,089 pairs from 1,588 molecules**, 6,607 of them biomolecular (6,579
  dipeptides). Each model is scored on the pairs it can actually evaluate.
- **Charges:** 404 molecules are ions. Formal charge and spin multiplicity are
  parsed from each molecule's SMILES rather than assumed neutral — running an
  anion as a closed-shell neutral returns a finite number for the wrong
  electronic state.
- **Scoring:** every comparison is on the **difference** `E(B) − E(A)` within one
  molecule, never absolute energies. Models use different reference zeros (ORB and
  MACE both sit near −28,000 eV on these molecules), so only differences are
  transferable.

## Reproducing it

The two `--max-*-per-molecule 12` flags are **not optional**: they are the values
the committed manifest was built with, and a reviewer found that omitting them
yields 12,242 pairs instead of 16,089 while every later step silently succeeds.

```bash
python scripts/par_fetch.py <zenodo-url> SPICE-1.1.4.hdf5   # 16 GB, resumable
python scripts/spice_inventory.py --archive SPICE-1.1.4.hdf5
python scripts/build_pairs.py --archive SPICE-1.1.4.hdf5 --max-per-subset 300 \
    --max-pairs-per-molecule 12 --max-conformers-per-molecule 12
bash scripts/sweep_all.sh                                   # one launcher, one GPU
python scripts/verify_completeness.py --write               # gate: artefact coverage
python scripts/score.py --require-complete
python scripts/make_figures.py
pytest tests/
```

Each model runs in **its own virtualenv** (`envs/<model>/`), not out of taste:
installing `fairchem-core` upgrades `e3nn` to 0.6, which silently breaks MACE
checkpoint loading (`ValueError: too many values to unpack`) *at evaluation time*,
not at import. A shared environment produces wrong numbers, quietly.

## What went wrong on the way (kept on purpose)

Six defects changed or withdrew a published number or broke a guard on one.
Regression tests for them are in `tests/`. Full account: [`docs/DEFECT_TRAIL.md`](docs/DEFECT_TRAIL.md).

| defect | effect on the published number |
|---|---|
| SPICE coordinates are Bohr, were read as Ångström | MACE-OFF23 scored at chance, 0.489; 0.723 once converted |
| ORB-omol row scored from a 1,500-conformer artefact | 0.893 withdrawn; full re-run gives 0.900 on 15,991 pairs |
| Chunk cache keyed by index, not by slice | merged artefacts had 1,200 duplicated and 6,232 missing conformers |
| Failure counts summed from truncated lists | a run with 319 failed conformers reported 70 |
| MMFF94 energies stored as eV, are kcal/mol | MMFF94 error inflated 23x; an "error does not predict accuracy" headline was refuted and removed |
| Completeness gate was a hand-edited file | now recomputed from raw artefacts; scoring refuses partial sweeps |

## Limits, stated plainly

- **SPICE is ωB97M-D3(BJ), not CCSD(T).** At sub-kcal gaps the reference itself carries
  error; a "wrong" decision within ~0.2 kcal/mol should be read as
  indistinguishable from reference noise, not as a model failure. ORB-omol's
  training labels (OMol25) are ωB97M-V/def2-TZVPD, a neighbouring but different
  level of theory, so part of its residual is method difference rather than
  model error.
- **Two of the three rows are train/test contaminated, and I only found out after
  publishing.** This is the single biggest caveat in the table and a reviewer had
  to point it out:
  - **MACE-OFF23** is trained on ~95% of SPICE v1 (MACE-OFF paper, arXiv 2312.15211,
    Sec. II.2), so scoring it on SPICE 1.1.4 is mostly an **in-sample** evaluation.
    Its 0.760 is an in-sample number that is still only 0.760.
  - **ORB-omol** is trained on OMol25, whose community domain documents SPICE2 — a
    superset of SPICE 1.x. So it is also very likely in-sample at the molecule
    level; geometry-level overlap I cannot verify.
  - **MMFF94** is the only clean out-of-sample row.
  - Read the ML accuracies as *fidelity to their own training data*, not as
    generalisation. The held-out test above (Folmsbee, with MPCONF196 as a small
    check) shows the ranking survives outside SPICE; it does not measure how
    large the in-sample inflation is, because it has no valid difficulty
    control. TorsionNet206 has not been run.
- **Not every pair is a conformer pair, and the wording above overstates it.**
  3,149 of 16,089 pairs (19.6%) come from multi-component systems: the DES370K
  subset is hydrogen-bonded **dimers** (`C=C.[Na+]`), so those pairs are
  intermolecular geometry changes, and the 21 ion-pair molecules are diatomics
  scanned over 2.5-7.4 Å. The ion-pair row is a mechanical failure, not a chemistry
  failure — the two ML models return exact-zero gaps there (ORB on 121 of its 143
  decided ion-pair pairs, MACE-OFF23 on all 74 of its), consistent with a graph
  cutoff, and ORB scores 0.126 there. MMFF94 has no cutoff now that its
  inter-fragment terms are on, and scores 1.000 on its 46. Excluding those two subsets changes
  every headline number and is the first thing a follow-up should do.
- **These geometries are MD snapshots, not relaxed minima.** SPICE's conformers are
  minimised with only a few L-BFGS steps plus a short 100 K MD run, and the parent
  sets include 300/500 K MD frames. An electronic ΔE between two such snapshots is
  not the ΔG a chemist means by "the preferred conformer". Ranking snapshots is a
  stricter test of a potential than ranking minima; it is not the same test.
- **The "biomolecular" slice is 99.6% dipeptides** (6,579 of 6,607 pairs), with 28
  from solvated amino acids. "Bio is harder" is really "dipeptides are harder"; it
  carries no evidence about proteins, nucleic acids, or ligands.
- **Pairs cluster by molecule and the intervals do not account for it.** The design
  effect is about 2 (ICC 0.10-0.13), so the ORB and MACE intervals should be about
  1.4x wider than quoted. The held-out Wilson intervals are worse off (design
  effect 3.4–5.2); compound-clustered intervals are given beside that table. No conclusion changes — the model differences are far
  larger than that — but the CIs as printed are optimistic.
- **A conformer pair is a proxy for a chemical decision.** It is the cheapest
  honest proxy — same molecule, real quantum energies, thousands of examples — but
  it is not a reaction energy, a binding affinity, or a barrier. Do not quote these
  numbers as "reaction prediction accuracy".
- **Three models is not a model zoo.** No trend line is fitted, no ranking beyond
  these three is implied, and anything said here about how energy error relates to
  decision accuracy is a statement about these three, not a law.
- **Counts differ per model** because a conformer a model cannot evaluate cannot
  be scored in any pair containing it; each row states its own n. MACE-OFF23
  returned NaN on 319 of 15,232 conformers (2.1%). I originally wrote "almost all
  charged or radical species"; that was wrong. Checked: **every one of the 319
  contains an element MACE-OFF23 does not support** (Li 43, Na 135, K 154, Mg 3,
  Ca 20), and none is merely charged.
- **MACE-OFF23 has no charge input, yet 20.6% of the pairs it is scored on are
  charged species.** Measured: 3,241 of its 15,717 decided pairs involve a
  charged conformer, and it scores **0.623 there against 0.796 on neutral
  pairs**. It was trained on the neutral ten-element subset of SPICE and accepts
  no charge argument, so part of its gap to ORB is domain mismatch rather than
  model quality. Scored only on its own training domain (neutral, supported
  elements, no ion pairs) MACE reads 0.796 on 12,476 pairs — the neutral pairs
  above, since every conformer with an unsupported element already returned NaN.
- **MACE-OFF23 runs in float64; an earlier version of this paragraph described a
  float32 run.** Its absolute energies sit near −28,000 eV, where float32
  resolution is ~0.045 kcal/mol, and the float32 run had 441 exact-zero gaps. The
  committed run is float64 (`scripts/run_model.py`): 82 of MACE-OFF23's 15,717
  decided pairs (0.52%) have a model gap of *exactly* zero — the 74 ion pairs
  above and 8 DES370K dimers, a cutoff effect rather than precision — and are
  counted as wrong. Excluding them moves MACE 0.760 → 0.764.
- **The subset split is mine.** SPICE's `subset` labels are read from the archive,
  but the biomolecular / small-molecule grouping is a choice I made; the headline
  numbers do not depend on it.
- **MMFF94 is not a fair fight on energy.** It is the classical reference point,
  and its huge error is why the accuracy axis matters — not a claim that a force
  field is competitive.
- **No commercial claim.** MACE-OFF23's **weights are non-commercial only** (an
  Academic Software Licence) even though its code is MIT; ORB-omol's package
  metadata declares Apache-2.0; RDKit/MMFF94 is BSD. Per-model terms are in every
  result artefact and on [`figures/licence_table.png`](figures/licence_table.png).
  Nothing here implies commercial use of an ASL model.

## Layout

```
src/mlipfid/decisions.py       decision pairs, Wilson intervals, null control
src/mlipfid/spice.py           SPICE reader (Bohr -> Angstrom, charges, subsets)
src/mlipfid/select.py          energy-stratified conformer selection
scripts/run_model.py           one model, one subprocess, JSON out
scripts/run_sweep.py           chunked, resumable, per-model venv dispatch, locked
scripts/verify_completeness.py recomputes which models cover every conformer
scripts/score.py               sign agreement, error stats, null control
scripts/make_figures.py        seven figures, every number read from a CSV
scripts/reconcile_heldout.py   held-out checks: clustered intervals, accuracy by gap size
docs/DEFECT_TRAIL.md           the wrong numbers found on the way, and their fixes
tests/                         tests: units, sign convention, null, layout, cache
```

MIT licence for this repository's own code. The models and data keep their own
terms — see the licence column and the figure above.

## References

- **SPICE** — Eastman et al., *SPICE, A Dataset of Drug-like Molecules and Peptides
  for Training Machine Learning Potentials*, Sci. Data 10, 11 (2023).
  doi:10.1038/s41597-022-01882-6. Level of theory ωB97M-D3(BJ)/def2-TZVPPD.
- **MACE-OFF23** — Kovács et al., arXiv:2312.15211. Trained on ~95% of SPICE v1
  (neutral, ten elements, no ion pairs), which is why its row is in-sample here.
- **OMol25** — arXiv:2505.08762. ORB-omol's training set; its documented community
  domain includes SPICE2, a superset of SPICE 1.x.
- **RDKit MMFF94** — Halgren, *J. Comput. Chem.* 17, 490 (1996), via RDKit
  2026.09.2. Note `CalcEnergy()` returns kcal/mol.
- **Prior art on decision-vs-average-error framing** — Matbench Discovery
  (Riebesell et al.) makes the "a low MAE does not mean the model is useful"
  argument for materials; conformer-ranking benchmarks (Folmsbee, MPCONF196,
  TorsionNet206) exist for the molecular case. **This repo's contribution is not the
  framing; it is the measurement on SPICE conformer pairs, and the disclosure that
  two of its three models are trained on the test data.**
