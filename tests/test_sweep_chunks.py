"""Tests for the sweep's chunk cache.

Written against a real, observed corruption rather than a hypothetical one: two
sweeps with different `--chunk` sizes ran against the same `results/` directory,
and because chunk files were named by chunk INDEX only

    chunk_orb_orbmol_v1_conservative_005.json

a 600-conformer chunk was written into the slot a 1500-conformer chunk had
occupied. The resume path then read that file as though it were a 1500-conformer
slice and concatenated it. The merged raw file ended up with 10200 energies but
only 9000 distinct conformer ids: 1200 conformers scored twice and 6232 never
scored at all. Nothing crashed. The sweep printed "ok" and a plausible accuracy.

So the cache key must identify the SLICE, not the ordinal, and a cache entry
whose contents do not match the slice currently requested must be rejected.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _sweep_module():
    spec = importlib.util.spec_from_file_location(
        "run_sweep", ROOT / "scripts" / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sweep():
    return _sweep_module()


def test_chunk_bounds_cover_every_conformer_exactly_once(sweep):
    n = 15232
    for chunk in (600, 1500, 2000, 15232, 99999):
        bounds = sweep.chunk_bounds(n, chunk)
        covered = []
        for lo, hi in bounds:
            assert 0 <= lo < hi <= n
            covered.extend(range(lo, hi))
        assert covered == list(range(n)), f"--chunk {chunk} does not tile 0..n"


def test_chunk_path_encodes_the_slice_not_the_ordinal(sweep, tmp_path):
    """The exact bug: index-only names collide across different chunk sizes."""
    a = sweep.chunk_path(tmp_path, "m", 7500, 9000)
    b = sweep.chunk_path(tmp_path, "m", 7500, 8100)
    assert a != b, "different slices must not share a cache filename"
    # Sanity: the same slice is stable, so caching still works.
    assert a == sweep.chunk_path(tmp_path, "m", 7500, 9000)


def test_stale_chunk_from_a_different_chunk_size_is_not_reused(sweep, tmp_path):
    """A cached chunk whose ids are not the requested slice must be refused."""
    ids_1500 = [f"c{i}|1" for i in range(1500)]
    ids_600 = [f"c{i}|1" for i in range(600)]
    path = sweep.chunk_path(tmp_path, "m", 7500, 9000)
    path.write_text(json.dumps({
        "ok": True, "model": "m", "ids": ids_600,
        "energies_ev": [0.0] * 600, "failures": [], "meta": {},
        "wall_seconds": 1.0,
    }))
    assert sweep.load_cached_chunk(path, ids_1500) is None, (
        "a 600-conformer chunk sat in a 1500-conformer slot and was reused")


def test_matching_chunk_is_reused(sweep, tmp_path):
    """The cache must still work, or every resume re-runs 15000 conformers."""
    ids = [f"c{i}|1" for i in range(10)]
    path = sweep.chunk_path(tmp_path, "m", 0, 10)
    path.write_text(json.dumps({
        "ok": True, "model": "m", "ids": ids,
        "energies_ev": [float(i) for i in range(10)], "failures": [],
        "meta": {"model": "m"}, "wall_seconds": 2.0,
    }))
    got = sweep.load_cached_chunk(path, ids)
    assert got is not None and got["ids"] == ids and got["wall_seconds"] == 2.0


def test_merged_ids_must_be_unique(sweep):
    """Guard the artefact itself: a merged raw file cannot score a conformer twice."""
    with pytest.raises(ValueError):
        sweep.check_merged_ids(["a", "b", "a"])
    sweep.check_merged_ids(["a", "b", "c"])   # must not raise


def test_failed_count_is_the_true_nan_count_not_the_truncated_list(sweep):
    """Second real bug: per-chunk failure lists are capped at 10 for payload size.

    The merge summed the LENGTH OF THE TRUNCATED LISTS, so a run with 319 NaN
    conformers reported n_failed=70 (10 per chunk x 7 chunks) and every downstream
    table under-reported missing data by 4.5x. Each chunk file carries its own
    true `n_failed`, so the merge must use that, and must still fall back to the
    list length for older chunks that predate the field.
    """
    chunk_new = {"ok": True, "ids": ["a"], "energies_ev": [1.0],
                 "n_failed": 88, "failures": [{"id": "x", "error": "boom"}] * 10,
                 "meta": {}, "wall_seconds": 1.0}
    chunk_old = {"ok": True, "ids": ["b"], "energies_ev": [2.0],
                 "failures": [{"id": "y", "error": "boom"}] * 3,
                 "meta": {}, "wall_seconds": 1.0}
    assert sweep.chunk_failed_count(chunk_new) == 88
    assert sweep.chunk_failed_count(chunk_old) == 3


def test_runner_takes_its_own_lock_so_wrappers_cannot_bypass_it(sweep):
    """Third real bug: the mutual exclusion lived in scripts/sweep_all.sh only.

    A stale /tmp/orb2.sh launcher called run_sweep.py DIRECTLY with --chunk 400
    while a --chunk 1500 sweep was running, and both wrote into results/. Because
    a chunk file named for slice (0,400) and one named (0,1500) are different
    files, the slice-keyed cache did not stop them colliding in the MERGED file:
    the second process truncated raw_orb_*.json to its own 400-row partial while
    the other was still going. The lock has to travel with the script that
    writes, not with one of the wrappers that calls it.
    """
    assert hasattr(sweep, "acquire_run_lock"), (
        "run_sweep must take the lock itself, not rely on a shell wrapper")


def test_second_concurrent_run_is_refused(sweep, tmp_path):
    lock = tmp_path / "sweep.lock"
    first = sweep.acquire_run_lock(lock, blocking=False)
    assert first is not None
    try:
        second = sweep.acquire_run_lock(lock, blocking=False)
        assert second is None, "a second concurrent sweep was allowed to start"
    finally:
        first.close()
    third = sweep.acquire_run_lock(lock, blocking=False)
    assert third is not None, "lock was not released"
    third.close()
