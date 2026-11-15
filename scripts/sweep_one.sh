#!/bin/bash
# One unit per model, so a slow model cannot starve a fast one and so a killed
# model never loses another model's finished raw file. The raw JSON is written
# only when the runner completes, so the runtime budget must exceed the real cost;
# budgets below are 2x the measured rate from the 12-conformer smoke test.
set -u
cd /home/ubuntu/biotech-portfolio/mlip-fidelity
P=/home/ubuntu/.openclaw/workspace/venv/bin/python
MODEL="$1"
$P scripts/run_sweep.py --models "$MODEL"
rc=$?
echo "rc=$rc model=$MODEL finished=$(date -Is)"
