#!/bin/bash
# The ONE sweep launcher. Chunked so an OOM or a kill keeps finished chunks, and
# serialised so two models never fight over the single GPU (measured: a second
# runner dropped throughput and doubled wall time with no gain).
#
# Models: the two whose venvs are proven, plus the classical force-field baseline.
# mace_off_small is kept first because it is the fastest and therefore delivers a
# trustworthy number earliest.
set -u
cd /home/ubuntu/biotech-portfolio/mlip-fidelity
P=/home/ubuntu/.openclaw/workspace/venv/bin/python
LOCK=/tmp/mlip_sweep.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another sweep holds $LOCK; refusing to start a duplicate"
  exit 75
fi
echo "start $(date -Is)"
$P scripts/run_sweep.py \
   --models mace_off_small orb_orbmol_v1_conservative rdkit_uff \
   --chunk 1500
rc=$?
echo "rc=$rc finished=$(date -Is)"
exit $rc
