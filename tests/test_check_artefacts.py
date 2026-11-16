"""Tests for scripts/check_artefacts.py: it must check CORRECTNESS, not only
self-consistency.

The code review (r2, finding 3) showed the previous version passed with every
MMFF gap re-inflated 23.06x (the F1 kcal/mol-as-eV bug), because a positive
rescale leaves every sign, every `agree` flag and every accuracy unchanged. These
tests run the checker on a scratch copy of results/ (via --results) and require:

  * the committed artefacts pass, and the recompute checks actually ran;
  * the F1 bug re-introduced in the decision CSVs AND the summary medians fails;
  * a single perturbed raw energy fails.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCRIPT = ROOT / "scripts" / "check_artefacts.py"
PY = sys.executable
KCAL_PER_EV = 627.5094740631 / 27.211386245988          # 23.0605...

# Everything check_artefacts reads. Held-out artefacts are out of its scope.
NEEDED = ["pairs_manifest.json", "conformers.jsonl", "summary.csv", "summary.json"]


def _scratch(tmp_path: Path, mutate=()) -> Path:
    """Copy results/ into tmp_path; files named in `mutate` are real copies,
    the rest are symlinks (the conformer file is 18 MB and never mutated)."""
    d = tmp_path / "results"
    d.mkdir()
    names = NEEDED + [p.name for p in RESULTS.glob("raw_*.json")] \
        + [p.name for p in RESULTS.glob("decisions_*.csv")]
    for n in names:
        src = RESULTS / n
        if n in mutate:
            shutil.copy(src, d / n)
        else:
            os.symlink(src, d / n)
    return d


def _run(results_dir: Path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([PY, str(SCRIPT), "--results", str(results_dir)],
                          capture_output=True, text=True, env=env, timeout=120)


def test_committed_artefacts_pass_and_recompute_checks_ran(tmp_path):
    d = _scratch(tmp_path)
    r = _run(d)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    # not a vacuous pass: the correctness checks must report they ran on every
    # model and the geometry sample must cover every subset in the manifest
    for model in ("mace_off_small", "orb_orbmol_v1_conservative", "rdkit_mmff"):
        assert f"raw-to-CSV recompute ok: {model}" in out, out
    subsets = json.loads((RESULTS / "pairs_manifest.json").read_text())["per_subset"]
    assert "geometry sample subsets: " + ",".join(sorted(subsets)) in out, out


def test_f1_unit_bug_in_csvs_and_summary_fails(tmp_path):
    """The reviewer's sim (b): MMFF gaps/errors x23.06 in both CSVs and in the
    summary medians. Signs, agree flags and accuracies are all unchanged."""
    mm = ["decisions_rdkit_mmff_all.csv", "decisions_rdkit_mmff_bio.csv"]
    d = _scratch(tmp_path, mutate=mm + ["summary.csv"])
    for n in mm:
        df = pd.read_csv(d / n)
        df["model_gap_kcal"] *= KCAL_PER_EV
        df["error_kcal"] *= KCAL_PER_EV
        df.to_csv(d / n, index=False)
    s = pd.read_csv(d / "summary.csv")
    m = s["model"] == "rdkit_mmff"
    for c in ("median_abs_err_kcal", "median_err_in_bars", "p90_abs_err_kcal",
              "bias_kcal"):
        s.loc[m, c] *= KCAL_PER_EV
    s.to_csv(d / "summary.csv", index=False)
    r = _run(d)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "rdkit_mmff" in r.stdout and "model_gap" in r.stdout, r.stdout


def test_one_perturbed_raw_energy_fails(tmp_path):
    d = _scratch(tmp_path, mutate=["raw_orb_orbmol_v1_conservative.json"])
    p = d / "raw_orb_orbmol_v1_conservative.json"
    raw = json.loads(p.read_text())
    man = json.loads((RESULTS / "pairs_manifest.json").read_text())
    # a conformer that is used by a pair, so a decision row depends on it
    cid = man["pairs"][0]["id_a"]
    i = raw["ids"].index(cid)
    raw["energies_ev"][i] += 1e-3          # 0.023 kcal/mol: no sign need flip
    p.write_text(json.dumps(raw))
    r = _run(d)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "orb_orbmol_v1_conservative" in r.stdout, r.stdout


def test_bohr_geometry_outside_pubchem_fails(tmp_path):
    """The old sample was sorted(ids)[:300] = 300 pubchem conformers. A dipeptide
    conformer left in bohr must now be seen: the sample covers every subset and
    always includes each subset's first sorted id."""
    d = _scratch(tmp_path, mutate=["conformers.jsonl"])
    lines = (d / "conformers.jsonl").read_text().splitlines()
    recs = [json.loads(x) for x in lines if x.strip()]
    target = sorted(r["id"] for r in recs if r["subset"] == "dipeptide")[0]
    with open(d / "conformers.jsonl", "w") as fh:
        for r in recs:
            if r["id"] == target:
                r["positions"] = [[c / 0.52917721067 for c in xyz]
                                  for xyz in r["positions"]]
            fh.write(json.dumps(r) + "\n")
    r = _run(d)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "geometry" in r.stdout, r.stdout


def test_raw_bookkeeping_mismatch_fails(tmp_path):
    """n_failed must equal the NaN count and n_ok + n_failed == len(ids)."""
    d = _scratch(tmp_path, mutate=["raw_rdkit_mmff.json"])
    p = d / "raw_rdkit_mmff.json"
    raw = json.loads(p.read_text())
    raw["n_failed"] = 0                      # the F6 root-cause symptom
    p.write_text(json.dumps(raw))
    r = _run(d)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "n_failed" in r.stdout, r.stdout
