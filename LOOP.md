# Long-horizon loop: mlip-fidelity

Model for every turn in this loop: `opencode-go/deepseek-v4.1-flash` (Go allowance).
Started 2026-11-15. This file documents the build loop the repo was
produced under; it is kept because the defect trail is part of the evidence.

## Objective
Ship `Qwuantabee/mlip-fidelity` on GitHub: a repo that answers one sharp, novel
question — *do universal machine-learned interatomic potentials preserve the
chemical decisions (conformer orderings) that a design loop depends on, even when
their energy MAE is tiny?* — with real quantum reference data, real GPU sweeps,
honest statistics, tests, and publication-quality figures.

## State machine (each turn does exactly one step, then re-arms)

| step | done when |
|---|---|
| 1. DOWNLOAD | `_data/SPICE-1.1.4.hdf5` exists, size == Content-Length |
| 2. INVENTORY | `results/spice_inventory.json` written; subsets chosen |
| 3. PAIRS | `results/pairs_manifest.json` + `conformers.jsonl` committed |
| 4. SWEEP | `results/raw_*.json` for every model that will run |
| 5. SCORE | `results/summary.csv` + per-model decision CSVs |
| 6. FIGURES | 6 figures from artefacts, each read from a CSV |
| 7. README | headline finding, limits, licences, one-command reproduction |
| 8. CI | tests green locally and on GitHub Actions |
| 9. PUSH | repo public, README renders, figures embedded |

Progress ledger: `LOOP_STATE.json` (step, notes, timestamps). Always read it first.

## Rails (non-negotiable, carried from the other four repos)
- Every number on the page comes from a committed artefact; titles computed.
- A test must fail before a fix for any logic change.
- The null control must be reported, not optional.
- Losers are reported too. No model is dropped for being embarrassing.
- Licences recorded per model; never imply commercial use of an ASL model.
- Never claim a step DONE without the artefact existing and a test or command
  output proving it. Evidence before assertions.
- Long jobs run under `systemd-run --user`, never `setsid nohup`.
- Do not type Python with control flow into `python -c`; write a file.
