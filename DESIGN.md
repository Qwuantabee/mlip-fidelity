# Design: does the potential change the chemistry decision?

**Repo:** `mlip-fidelity` — working title *Do machine-learned interatomic potentials
preserve chemical decisions?*

> **Status: the plan as written before any measurement.** It is kept so the plan
> can be compared with what was done; the README is the record of the result.
> What shipped differs from this document in four ways:
>
> 1. Three models were scored (ORB-omol v1, MACE-OFF23 small, MMFF94), not the
>    zoo listed below.
> 2. The headline plots accuracy against each model's own measured pair-gap
>    error, not against a published energy MAE; figures (b), (e) and (f) below
>    were not made as described.
> 3. Two framing claims below turned out wrong and are corrected in the README:
>    the 1 kcal/mol bar is *not* far above a ~1 meV/atom MAE (at 36 atoms it is
>    1.2 meV/atom), and conformer-ranking benchmarks already exist (Folmsbee,
>    MPCONF196, TorsionNet206), so "nobody has measured this" is not true.
> 4. This document does not mention train/test contamination. Two of the three
>    models were trained on SPICE; that was found after the first results and is
>    the main caveat on them.

## The question nobody in the 2026 uMLIP wave is answering

Every 2026 benchmark of universal machine-learned interatomic potentials (uMLIPs)
reports the same thing: energy MAE in meV/atom, force MAE in meV/Å, maybe a
stress error. Papers published this year (OMOL-1k-MD, MOFSimBench, GRACE
universal potentials, the 15-model pretrained-MLIP benchmark) all rank models
this way.

But nobody downstream of these benchmarks consumes an energy MAE. A chemist
consumes a **decision**: is this conformer lower than that one, does this
reaction go forward, is this tautomer preferred, does this ligand bind in
pose A or pose B, which of these two products is favoured. Those are questions
about *differences* and *rankings*, at a precision of roughly 1 kcal/mol
(43 meV) — a threshold far above the ~1 meV/atom MAEs these models advertise.

**The gap:** a model can win the MAE leaderboard and still invert the chemistry,
because MAE is dominated by the bulk of the distribution while decisions live in
the tail of small differences. Nobody has measured that gap systematically.

That is the whole project. It is one sharp question, it needs a GPU, it produces
a number no one has published, and it is exactly what a computational chemistry
or protein-engineering team would want to know before trusting a foundation
potential in a design loop.

## Why this question

It is **physics, not sequence**: it is about whether the simulation engine itself
is trustworthy, and it needs thousands of GPU single-point evaluations across
several models.

## The core measurement

For each of N systematically chosen **decision pairs** — two geometries that
differ only in a chemically meaningful way — compute, for every model M:

    ΔE_M = E_M(geometry B) − E_M(geometry A)
    ΔE_QM = the same difference from the reference (ωB97M-D3(BJ) / DFT, or CCSD(T)
            where the dataset provides it)

and ask: **does sign(ΔE_M) == sign(ΔE_QM), and by how much does |ΔE_M − ΔE_QM|
miss the 1 kcal/mol bar?**

Then aggregate: for each model, a **decision accuracy** (fraction of pairs with
the right sign, with a Wilson interval) and a **decision error distribution**
(median and 90th percentile |ΔE error| in kcal/mol), plotted *against* that
model's published energy MAE. That scatter — decision accuracy vs MAE — is the
headline figure and the thing that makes the repo quotable.

## Where the data comes from

Use SPICE (16 GB, Zenodo 8222043: ~1M conformers with
ωB97M-D3(BJ)/def2-TZVPPD energies and forces across drug-like molecules, peptides,
solvated amino acids, and protein fragments). It is the community standard for
training and evaluating MLIPs on biomolecular chemistry, and unlike most
benchmarks it contains **many near-degenerate conformers of the same molecule** —
which is precisely what a decision pair is.

Critically this gives a *biologically relevant* angle no materials benchmark has:
SPICE's peptide and solvated-amino-acid subsets let the decisions be things like
"which backbone dihedral does this dipeptide prefer" and "is this hydrogen-bond
network tighter" — protein-engineering questions, not just small-molecule ones.

Decision pairs are constructed deterministically (fixed seed, recorded in a
manifest) by grouping conformers of the same molecule and taking pairs whose
QM energy differs by less than a threshold, plus a matched control set of pairs
that differ a lot. Nothing is hand-picked; the grouping rule is the method.

## The model zoo (planned; see Status for what ran)

- MACE-OFF23 (small/medium)
- MACE-MP-0 (materials-trained, as a domain-transfer contrast)
- ORB-v2 / ORB-v3 — `orb-models`
- eSEN / UMA via `fairchem-core`
- M3GNet / CHGNet via `mattersim`/`matgl` where installable
- A classical baseline (UFF/MMFF94 via RDKit) as the "how bad is no ML at all" anchor

Each model runs in its own subprocess so a crash or a licence wall cannot take
down the sweep, and every model's identity (version, checkpoint hash, licence)
is recorded in the results.

## Non-negotiables (house standard, carried from the other four repos)

1. **Every number on the page comes from a committed artefact.** No typed-in
   values; the figure code reads the CSV and the title is computed.
2. **A test that fails before the fix.** For the pairing logic, the sign
   convention (B − A, never A − B), the kcal/mol conversion, and the null control.
3. **A null control of its own.** Random pairing (shuffle which B goes with which
   A) must destroy the signal. If shuffled pairs score as well as real pairs, the
   pipeline is measuring nothing and the run is void.
4. **Report the losers.** Models that do well and models that fail both appear;
   no model is dropped from the table for being embarrassing.
5. **Stated limits, not hidden ones.** SPICE is ωB97M-D3(BJ), not CCSD(T); a decision
   pair is a proxy for a chemical decision; conformer pairs are not reaction
   energies. All in the README's limits section.

## Figures (each from a committed artefact)

- **a. Decision accuracy vs published energy MAE** — the headline scatter, with
  the 1 kcal/mol band shaded and a Wilson interval per model.
- **b. Error distribution per model** — violin/strip of |ΔE error| in kcal/mol
  against the 1 kcal/mol bar, so the tail is visible, not averaged away.
- **c. Where the failures live** — decision error vs QM gap size (small-gap
  pairs should be hardest; if a model is flat here it is not resolving anything).
- **d. Peptide/amino-acid subset only** — the protein-engineering slice.
- **e. Rank correlation** — Spearman between ΔE_M and ΔE_QM across all pairs, the
  ranking-preservation view a design loop actually needs.
- **f. The null control** — shuffled-pair scores, to prove the signal is real.

## Deliverables

- README whose first screen is the headline scatter and a one-sentence finding
- `results/decisions_<model>.csv` per model, `results/summary.csv` aggregated
- reproducible one-command sweep (`scripts/run_sweep.py`)
- tests for every invariant, green in CI
- published to GitHub under `Qwuantabee/mlip-fidelity`

## Honest risk list

- **SPICE download is slow** (~0.35 GB/min observed). Mitigation: parallel-range
  fetch; if it keeps failing, fall back to the smaller SPICE subsets
  (DES370K-like) that are hosted separately, and state the reduced scope.
- **Not all packages will install cleanly.** Mitigation: each model is optional;
  the sweep records which ones loaded and the paper names them. A 4-model result
  is still a result; a 7-model one is better.
- **Some models are licence-restricted** (MACE-OFF is academic-only). Mitigation:
  record the licence per model in the results and README; never imply commercial
  usability of a model that forbids it.
- **The finding may be "everything is fine".** That is a publishable, honest
  result too, and the null control plus intervals make the claim defensible
  either way. Do not manufacture a gap.
