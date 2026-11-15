#!/bin/bash
# The ONE sweep launcher. Chunked so an OOM or a kill keeps finished chunks, and
# serialised so two models never fight over the single GPU (measured: a second
# runner dropped throughput and doubled wall time with no gain).
#
# A second launcher is also refused by the flock for a second reason learned the
# hard way: a duplicate running with a different --chunk size reused the other's
# chunk cache slots and silently corrupted every merged raw file. Chunk cache
# names are now keyed by the slice, and every cache read re-checks the ids, but
# one sweep at a time is still the only sane state.
#
# Models: the two whose venvs are proven, plus two classical baselines.
# mace_off_small is first because it is the fastest trustworthy number.
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
   --models mace_off_small orb_orbmol_v1_conservative rdkit_mmff \
   --chunk 1500
rc=$?
echo "rc=$rc finished=$(date -Is)"
exit $rc
