# The held-out test in detail

Moved out of the README to keep it readable; the text is unchanged except for the
headings and one code comment. The README keeps the held-out tables, the accuracy-by-gap-size comparison and the conclusion; this file holds the charge correction, the measured run-to-run noise, the compound-clustered intervals, the MMFF94-offset estimate, the withdrawn MPCONF196 result and the commands that rebuild the held-out artefacts. "The table" below means the Folmsbee table in the README's held-out section.

## The charge correction

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

## Run-to-run noise

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

## Compound-clustered intervals

The intervals in the table are Wilson intervals, which treat the ~32 pairs per
compound as independent. They are not: resampling whole compounds (cluster
bootstrap, 10,000 draws, `scripts/intervals.py`) widens them to ORB [0.823, 0.842], MACE [0.762, 0.784]
and MMFF94 [0.607, 0.637], a design effect of 3.5–5.4. The Folmsbee column also
counts every sub-kcal pair, including 567 near-ties (|gap| ≤ 0.01 kcal/mol; 553
for MMFF94) that the SPICE column excludes; with ties excluded the held-out
figures are 0.840 / 0.780 / 0.624.

## The MMFF94-offset estimate, and why it is not a contamination measurement

MMFF94 has never seen either dataset, so one can take its +0.021 as an estimate
of how much easier the held-out slice is, and subtract it from each ML model's
raw shift. That treats slice difficulty as the same for every model, which the
gap-size table shows is false. Intervals resample SPICE molecules and Folmsbee
compounds independently of each other, one draw shared by all three models
(10,000 draws, `scripts/intervals.py`).

- **ORB-omol: −0.089** (95% CI −0.107 to −0.070) on that estimate. An earlier
  version of this README read it as "ORB-omol loses about 9 points beyond
  difficulty, consistent with in-sample inflation". That reading is withdrawn:
  the interval covers sampling error only, and the sign of the effect reverses
  once gap size is matched (+0.024 on its own, −0.038 relative to MMFF94's
  gap-matched +0.062). The data are consistent with some inflation and with none.
- **MACE-OFF23: −0.008** (95% CI −0.028 to +0.012); gap-matched it is +0.091.
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

## MPCONF196: the withdrawn result

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

## Reproducing the held-out half

**The held-out half** needs two small public repositories (no download over a
few MB) and the same model venvs:

```bash
git clone https://github.com/ghutchis/conformer-benchmark _data/folmsbee     # Folmsbee & Hutchison 2021, MIT; built at commit 0109c8e
git clone https://github.com/grimme-lab/benchmark-MPCONF196 _data/mpconf196  # Řezáč 2018, references revised 2023; built at commit e1d030d
python scripts/build_heldout_folmsbee.py     # -> results/heldout_folmsbee.json (22,379 pairs)
python scripts/build_heldout.py              # -> results/heldout_mpconf196.json
python scripts/score_heldout.py --bench results/heldout_folmsbee.json --tag heldout_folmsbee   # GPU
python scripts/score_heldout.py              # MPCONF196, GPU
python scripts/orb_repeatability.py --publish   # GPU, a few minutes
python scripts/reconcile_heldout.py          # clustered intervals, accuracy by gap size
python scripts/intervals.py                  # every bootstrap interval in the README and this file
```

Without a GPU, `score_heldout.py ... --tag heldout_folmsbee --from-raw` followed by
`orb_repeatability.py --from-saved --publish` re-derives every Folmsbee artefact
from the committed energies; both builders and that pair of commands reproduce
the committed files byte for byte.
