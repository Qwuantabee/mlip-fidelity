# Limits, in full

The README's `Limits, stated plainly` section is the short version; every limit
it lists is set out here with its measurement. Moved from the README unchanged.

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
  effect is about 2 (ICC about 0.10), so the ORB and MACE intervals should be about
  1.4x wider than quoted (`scripts/intervals.py`). The held-out Wilson intervals are worse off (design
  effect 3.5–5.4); compound-clustered intervals are given beside that table. No conclusion changes — the model differences are far
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
