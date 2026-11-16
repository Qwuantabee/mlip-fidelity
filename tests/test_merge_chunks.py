"""Tests for stale-input paths in the sweep tooling (scripts/merge_chunks.py and
the chunk cache in scripts/run_sweep.py).

R2 code review (reviews/r2_mlip_code_20261116.md, finding 5, prior F4/F5):

* sim C: run_sweep's chunk cache was keyed on ids only. Scaling one conformer's
  coordinates (same id) still printed "(cached)" and kept the stale energy
  (1.887 eV versus a fresh 14.858 eV).
* sim E: merge_chunks kept the FIRST occurrence of a duplicated id with no
  disagreement check, so a stale overlapping chunk (999.0) beat the fresh one
  and the merge still wrote ok: true and exited 0.
* sim E2: merge_chunks never compared with conformers.jsonl, so 1 of 2 chunks
  merged to an ok: true raw file.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MERGE = ROOT / "scripts" / "merge_chunks.py"
MODEL = "rdkit_mmff"   # any name in run_sweep.MODELS; the runner is faked below


def _sweep_module():
    spec = importlib.util.spec_from_file_location(
        "run_sweep_t", ROOT / "scripts" / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def sweep():
    return _sweep_module()


def _recs(n=6):
    return [{"id": f"c{i}|0", "molecule": f"c{i}", "atomic_numbers": [6, 1],
             "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0 + 0.1 * i]],
             "charge": 0, "spin": 1} for i in range(n)]


def _write_conformers(d: Path, recs) -> Path:
    p = d / "conformers.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def _fake_energy(rec) -> float:
    # Deterministic function of the geometry, so a stale energy is detectable.
    return float(sum(x for xyz in rec["positions"] for x in xyz))


def _age(path: Path, seconds: float):
    st = path.stat()
    os.utime(path, (st.st_atime - seconds, st.st_mtime - seconds))


def _run_merge(d: Path, *extra):
    # No --conformers: merge_chunks must find <dir>/conformers.jsonl itself.
    return subprocess.run(
        [sys.executable, str(MERGE), "--model", MODEL, "--dir", str(d), *extra],
        capture_output=True, text=True)


def _chunk(d: Path, lo: int, hi: int, recs, energies, sweep=None):
    body = {"ok": True, "model": MODEL, "meta": {"model": MODEL},
            "ids": [r["id"] for r in recs[lo:hi]], "energies_ev": energies,
            "failures": [], "n_failed": sum(1 for e in energies if e != e),
            "wall_seconds": 1.0}
    if sweep is not None:
        body["input_sha256"] = sweep.chunk_input_hash(MODEL, recs[lo:hi])
    p = d / f"chunk_{MODEL}_{lo:06d}_{hi:06d}.json"
    p.write_text(json.dumps(body))
    return p


# ---------------------------------------------------------------- merge_chunks

def test_stale_disagreeing_duplicate_fails_the_merge(tmp_path):
    recs = _recs(6)
    _write_conformers(tmp_path, recs)
    _age(tmp_path / "conformers.jsonl", 100)
    fresh = [_fake_energy(r) for r in recs]
    # A stale, overlapping range-named chunk sorts first (000000_000003 <
    # 000000_000006) and disagrees with the fresh one on 3 conformers.
    _chunk(tmp_path, 0, 3, recs, [999.0] * 3)
    _chunk(tmp_path, 0, 6, recs, fresh)
    p = _run_merge(tmp_path)
    assert p.returncode != 0, f"disagreeing duplicates merged silently:\n{p.stdout}"
    assert not (tmp_path / f"raw_{MODEL}.json").exists(), (
        "an ok: true raw file was written from disagreeing chunks")


def test_partial_coverage_fails_the_merge(tmp_path):
    recs = _recs(6)
    _write_conformers(tmp_path, recs)
    _age(tmp_path / "conformers.jsonl", 100)
    _chunk(tmp_path, 0, 3, recs, [_fake_energy(r) for r in recs[:3]])
    p = _run_merge(tmp_path)
    assert p.returncode != 0, f"3 of 6 conformers merged as complete:\n{p.stdout}"
    assert not (tmp_path / f"raw_{MODEL}.json").exists()


def test_merge_rejects_chunk_older_than_conformers(tmp_path):
    recs = _recs(6)
    c = _chunk(tmp_path, 0, 6, recs, [_fake_energy(r) for r in recs])
    _age(c, 100)                         # chunk predates the geometry it covers
    _write_conformers(tmp_path, recs)
    p = _run_merge(tmp_path)
    assert p.returncode != 0, f"chunk older than conformers.jsonl merged:\n{p.stdout}"
    assert not (tmp_path / f"raw_{MODEL}.json").exists()


def test_happy_path_merges(tmp_path, sweep):
    recs = _recs(6)
    _write_conformers(tmp_path, recs)
    _age(tmp_path / "conformers.jsonl", 100)
    en = [_fake_energy(r) for r in recs]
    en[4] = float("nan")
    _chunk(tmp_path, 0, 3, recs, en[:3], sweep)
    _chunk(tmp_path, 3, 6, recs, en[3:], sweep)
    # an agreeing overlap (re-run with a different --chunk) is fine
    _chunk(tmp_path, 0, 6, recs, list(en), sweep)
    p = _run_merge(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    raw = json.loads((tmp_path / f"raw_{MODEL}.json").read_text())
    assert raw["ok"] is True
    assert raw["ids"] == [r["id"] for r in recs]
    assert raw["energies_ev"][:4] == en[:4] and math.isnan(raw["energies_ev"][4])
    assert raw["n_ok"] == 5 and raw["n_failed"] == 1


# ------------------------------------------------------- run_sweep chunk cache

def _fake_runner(calls):
    def run_one(model, job, out_dir, device, limit):
        confs = json.loads(Path(job).read_text())["conformers"]
        calls.append([c["id"] for c in confs])
        return {"ok": True, "model": model, "meta": {"model": model},
                "ids": [c["id"] for c in confs],
                "energies_ev": [_fake_energy(c) for c in confs],
                "failures": [], "n_failed": 0, "partial": True}
    return run_one


def _sweep_main(sweep, d: Path, monkeypatch, calls):
    monkeypatch.setattr(sweep, "run_one", _fake_runner(calls))
    monkeypatch.setattr(sys, "argv", [
        "run_sweep.py", "--models", MODEL, "--conformers", str(d / "conformers.jsonl"),
        "--out", str(d), "--chunk", "3", "--device", "cpu"])
    assert sweep.main() == 0
    return json.loads((d / f"raw_{MODEL}.json").read_text())


def test_stale_cache_same_ids_changed_positions_is_recomputed(tmp_path, monkeypatch, sweep):
    recs = _recs(6)
    conf = _write_conformers(tmp_path, recs)
    _age(conf, 100)
    calls = []
    _sweep_main(sweep, tmp_path, monkeypatch, calls)
    assert len(calls) == 2
    # Same ids, one conformer's geometry scaled by 1.15 (review sim C). Age the
    # conformer file so ONLY the content hash can tell the chunk is stale.
    recs[1]["positions"] = [[1.15 * x for x in xyz] for xyz in recs[1]["positions"]]
    _write_conformers(tmp_path, recs)
    _age(conf, 100)
    calls.clear()
    raw = _sweep_main(sweep, tmp_path, monkeypatch, calls)
    assert calls == [[r["id"] for r in recs[0:3]]], (
        f"stale chunk was reused from cache (recomputed: {calls})")
    assert raw["energies_ev"][1] == _fake_energy(recs[1])


def test_cache_chunk_older_than_conformers_is_recomputed(tmp_path, monkeypatch, sweep):
    recs = _recs(6)
    conf = _write_conformers(tmp_path, recs)
    _age(conf, 100)
    calls = []
    _sweep_main(sweep, tmp_path, monkeypatch, calls)
    # unchanged inputs and fresh chunks: the cache must be used
    calls.clear()
    _sweep_main(sweep, tmp_path, monkeypatch, calls)
    assert calls == [], "a valid cached chunk was not reused"
    # conformers.jsonl rewritten after the chunks: every chunk is now stale
    os.utime(conf, None)
    for c in tmp_path.glob(f"chunk_{MODEL}_*.json"):
        _age(c, 50)
    calls.clear()
    _sweep_main(sweep, tmp_path, monkeypatch, calls)
    assert len(calls) == 2, f"chunks older than conformers.jsonl were reused: {calls}"
