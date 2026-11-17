# Defect trail: what went wrong on the way

Moved out of the README to keep it readable. Items 1 to 5 are unchanged; item 6
was added here because the README referred to it without listing it. Each item
is a wrong number or a broken guard that was found, what it did to the
published results, and the fix. Numbers quoted here are historical: they are
the values at the time of each defect, not current results.

This section exists because a repo that only shows its successes teaches the
reader nothing about how far to trust it.

## Summary

The table that was in the README; each row is detailed below.

| defect | effect on the published number |
|---|---|
| SPICE coordinates are Bohr, were read as Ångström | MACE-OFF23 scored at chance, 0.489; 0.723 once converted |
| ORB-omol row scored from a 1,500-conformer artefact | 0.893 withdrawn; full re-run gives 0.900 on 15,991 pairs |
| Chunk cache keyed by index, not by slice | merged artefacts had 1,200 duplicated and 6,232 missing conformers |
| Failure counts summed from truncated lists | a run with 319 failed conformers reported 70 |
| MMFF94 energies stored as eV, are kcal/mol | MMFF94 error inflated 23x; an "error does not predict accuracy" headline was refuted and removed |
| Completeness gate was a hand-edited file | now recomputed from raw artefacts; scoring refuses partial sweeps |

## The six defects

**1. A unit bug that made the first result meaningless.** The first full sweep said
MACE ranked sub-kcal pairs at chance: accuracy 0.489, null p = 0.58. The pipeline
was wrong, not the model — **SPICE stores conformations in Bohr**, and they were
being handed to ASE as Ångström, stretching every molecule by 1.89×. With the
conversion applied and nothing else changed, accuracy went 0.489 → 0.723.

**2. A withdrawn number.** An earlier revision of this README reported ORB-omol v1
at 0.893. That row had been scored from an artefact holding 1,500 conformers
while claiming 9,409 decided pairs. It was withdrawn rather than patched, and has
now been re-measured in full: the complete sweep gives **0.900 on 15,991 decided
pairs**, from an artefact that covers all 15,232 conformers. The pre-fix number
and the post-fix number are close, which is luck, not vindication — the first one
was not reproducible and should never have been published.

**3. A chunk cache that silently corrupted every result.** Chunk files were named
by chunk *index*. Two sweeps with different `--chunk` sizes — one launched by a
stale wrapper script that bypassed the shell-level lock — wrote into the same
slots, so a 600-conformer chunk was resumed as if it were a 1,500-conformer slice.
Merged artefacts reported `ok` while holding **1,200 duplicated and 6,232 missing
conformers**, and every published accuracy was computed from that. Three fixes,
each with a test that failed first: chunk cache keys now encode the *slice*, a
cached chunk is refused unless its ids match the requested slice exactly, and the
lock moved from the wrapper into `run_sweep.py` itself so no launcher can bypass
it.

**4. A failure count that under-reported missing data by 4.5×.** Per-chunk failure
lists are truncated to 10 entries for payload size; the merge summed those
truncated lengths, so a run with 319 NaN conformers reported `n_failed=70`. Each
chunk now carries its own true count.

**5. A completeness gate that was a hand-edited claim.** `sweep_complete.json` was
written by hand, which let a model be marked complete while its artefact was
partial. It is now recomputed from the raw artefacts by
`scripts/verify_completeness.py`, and `score.py --require-complete` will emit no
rows at all rather than quietly partial ones.

**6. A second unit bug, in the classical baseline.** RDKit's force-field energies
are kcal/mol; the runner stored them as eV, which inflated every MMFF94 gap by
23.06x (fixed in commit `107508d`, pinned by `tests/test_units.py`). The MMFF94
row published before the fix read 0.563. An earlier headline, that energy error
and decision accuracy were unrelated, rested on that inflated error and was
removed when the corrected numbers refuted it.

Earlier finds are also recorded with tests: an HDF5 reader that guessed the layout
wrong (the inventory returning *zero* multi-conformer molecules is what exposed
it), a classical baseline that "succeeded" with no bonds at all in the molecule,
a silent neutral-charge fallback, and a licence table that called MACE's
non-commercial weights permissive after a rewording.
