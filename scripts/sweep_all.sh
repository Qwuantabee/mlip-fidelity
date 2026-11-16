#!/bin/bash
# The ONE sweep launcher. Run it with bash (the README used to say
# `python scripts/sweep_all.sh`, which is a SyntaxError):
#
#     bash scripts/sweep_all.sh                 # full sweep, all three models
#     bash scripts/sweep_all.sh --limit 4       # smoke run; extra args go to run_sweep.py
#
# Chunked so an OOM or a kill keeps finished chunks, and serialised so two models
# never fight over the single GPU (measured: a second runner dropped throughput and
# doubled wall time with no gain).
#
# A second launcher is also refused by the flock for a second reason learned the
# hard way: a duplicate running with a different --chunk size reused the other's
# chunk cache slots and silently corrupted every merged raw file. Chunk cache
# names are now keyed by the slice, and every cache read re-checks the ids, but
# one sweep at a time is still the only sane state.
#
# Models: the two whose venvs are proven, plus the classical baseline.
# mace_off_small is first because it is the fastest trustworthy number.
#
# Paths are relative to this file, not to the author's machine. The python that
# runs the dispatcher is $PYTHON if set, else the `python` on PATH (the same one
# every other README step uses); each ML model still runs in its own envs/<name>.
set -u
cd "$(dirname "$0")/.." || exit 1
P="${PYTHON:-python}"
LOCK=/tmp/mlip_sweep.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another sweep holds $LOCK; refusing to start a duplicate"
  exit 75
fi
echo "start $(date -Is)"
"$P" scripts/run_sweep.py \
   --models mace_off_small orb_orbmol_v1_conservative rdkit_mmff \
   --chunk 1500 "$@"
rc=$?
echo "rc=$rc finished=$(date -Is)"
exit $rc
