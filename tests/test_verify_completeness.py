"""Tests for scripts/verify_completeness.py, the sweep completeness gate.

The R2 code review (reviews/r2_mlip_code_20261116.md, finding 4) showed the gate
checked conformer ids only and ended with an unconditional "return 0":

* a raw file whose energies were all 0.0, or all NaN, was reported complete;
* a merged file with 3 ids and 1 energy was reported complete;
* a model missing 8,232 conformers was listed as partial, but the script still
  exited 0, so a shell or CI step gating on its exit code gated nothing.

Each test builds a tiny results/ directory and runs the real script as a
subprocess, because the exit code IS the gate.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_completeness.py"
N = 100
IDS = [f"m{i}|{i % 3}" for i in range(N)]


def _results(tmp_path, raw: dict, ids=IDS) -> Path:
    res = tmp_path / "results"
    res.mkdir()
    (res / "conformers.jsonl").write_text(
        "".join(json.dumps({"id": i}) + "\n" for i in ids))
    raw = {"ok": True, "model": "m", **raw}
    (res / "raw_m.json").write_text(json.dumps(raw))
    return res


def _run(res: Path, *extra):
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(res),
         "--conformers", str(res / "conformers.jsonl"), *extra],
        capture_output=True, text=True)
    return p


def _good_energies(n_nan=2):
    e = [-1000.0 - 0.01 * i for i in range(N)]
    for i in range(n_nan):
        e[i * 7] = float("nan")
    return e


def _assert_rejected(p, why):
    assert p.returncode != 0, f"{why}: gate exited 0\n{p.stdout}"
    out = json.loads(p.stdout[:p.stdout.rindex("}") + 1])
    assert "m" not in out["complete"], f"{why}: model still listed complete"


def test_missing_ids_exit_nonzero(tmp_path):
    ids = IDS[:-5]
    res = _results(tmp_path, {"ids": ids, "energies_ev": [-1.0] * len(ids)})
    _assert_rejected(_run(res), "5 conformers missing")


def test_all_nan_energy_block_is_not_complete(tmp_path):
    res = _results(tmp_path, {"ids": IDS, "energies_ev": [float("nan")] * N})
    _assert_rejected(_run(res), "all-NaN energies")


def test_all_zero_energy_block_is_not_complete(tmp_path):
    res = _results(tmp_path, {"ids": IDS, "energies_ev": [0.0] * N})
    _assert_rejected(_run(res), "all-zero energies")


def test_short_coverage_is_not_complete(tmp_path):
    # Every id is present, but 40% of the energies are NaN: below the default
    # --min-coverage (0.95) the model must not pass as complete.
    e = _good_energies(n_nan=0)
    for i in range(40):
        e[i] = float("nan")
    res = _results(tmp_path, {"ids": IDS, "energies_ev": e,
                              "n_ok": 60, "n_failed": 40})
    _assert_rejected(_run(res), "60% coverage")


def test_energies_length_mismatch_is_not_complete(tmp_path):
    # Sim D in the review: 6 ids, 4 energies, ok: true.
    res = _results(tmp_path, {"ids": IDS, "energies_ev": _good_energies()[:N - 2]})
    _assert_rejected(_run(res), "len(energies_ev) != len(ids)")


def test_passing_case_exits_zero_and_committed_results_pass(tmp_path):
    # A realistic complete file: every id once, 2 NaN (as MACE has NaN for
    # unsupported elements), counters consistent, explicit coverage threshold.
    res = _results(tmp_path, {"ids": IDS, "energies_ev": _good_energies(2),
                              "n_ok": N - 2, "n_failed": 2})
    p = _run(res, "--min-coverage", "0.95", "--write")
    assert p.returncode == 0, p.stdout + p.stderr
    gate = json.loads((res / "sweep_complete.json").read_text())
    assert gate["complete"] == ["m"] and gate["partial"] == {}
    # The committed artefacts must still pass the stricter gate (default args).
    p = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert sorted(json.loads(p.stdout)["complete"]) == [
        "mace_off_small", "orb_orbmol_v1_conservative", "rdkit_mmff"]
