# Layout

Moved out of the README to keep it short; rows for the new `docs/` files were added.

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
docs/HELDOUT.md                held-out detail: charges, noise, clustered intervals, MPCONF196
docs/DEFECT_TRAIL.md           the wrong numbers found on the way, and their fixes
docs/LIMITS.md                 every limit, with its measurement
docs/DATA.md                   how the SPICE pairs are built
docs/REFERENCES.md             full citations
tests/                         tests: units, sign convention, null, layout, cache
```
