"""Guard tests for scripts/score.py (review r2, prior F7 scorer half / science F-J).

score.py used to score a manifest built from a different pair set than the raw
files and exit 0 with a believable accuracy (sim: 12,242-pair manifest against the
committed raw files printed MACE 0.698). Its `n_pairs_absent_from_raw` counter
also lumped NaN conformers (a legitimate, scored-and-failed outcome: 274 MACE /
0 ORB / 414 MMFF pairs in the committed run) together with ids that are simply
not in the raw file, and it was never printed or enforced. Required now:

  * a pair whose id is absent from a raw file makes score.py exit non-zero,
    naming the model and the absent count;
  * NaN-energy pairs are counted separately from absent pairs and both counts
    are printed per model and written to summary.csv;
  * the committed manifest + raw files still score with exit 0 and reproduce the
    committed decision CSVs and summary byte for byte.
"""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCRIPT = ROOT / "scripts" / "score.py"
PY = sys.executable

# NaN-energy pairs in the committed run, scope "all" (review F-J, measured).
NAN_PAIRS_ALL = {"mace_off_small": 274, "orb_orbmol_v1_conservative": 0,
                 "rdkit_mmff": 414}


def _score(out: Path, manifest: Path | None = None):
    cmd = [PY, str(SCRIPT), "--out", str(out)]
    if manifest is not None:
        cmd += ["--manifest", str(manifest)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)


@pytest.fixture(scope="module")
def committed_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("score_committed")
    return out, _score(out)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_pair_absent_from_raw_exits_nonzero(tmp_path):
    man = json.loads((RESULTS / "pairs_manifest.json").read_text())
    # an ORB pair (ORB has no NaN conformers), with one member renamed to an id
    # that no raw file contains: the manifest and the raw files disagree
    man["pairs"][0] = dict(man["pairs"][0], id_a="ghost-molecule|0")
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man))
    out = tmp_path / "out"
    out.mkdir()
    r = _score(out, mpath)
    assert r.returncode != 0, r.stdout + r.stderr
    msg = r.stdout + r.stderr
    assert "orb_orbmol_v1_conservative" in msg and "absent" in msg, msg
    assert re.search(r"\b1 \S*\s*pair", msg) or "absent=1" in msg, msg
    # nothing believable may be left behind for a figure to pick up
    assert not (out / "summary.csv").exists()


def test_nan_pairs_counted_separately_and_printed(committed_run):
    out, r = committed_run
    assert r.returncode == 0, r.stdout + r.stderr
    for model, n_nan in NAN_PAIRS_ALL.items():
        line = next(x for x in r.stdout.splitlines()
                    if x.startswith(model) and " all " in x)
        assert "absent=0" in line, line
        assert f"nan_pairs={n_nan}" in line, line
    rows = {(x["model"], x["scope"]): x
            for x in csv.DictReader(open(out / "summary.csv"))}
    for model, n_nan in NAN_PAIRS_ALL.items():
        assert int(rows[(model, "all")]["n_pairs_absent_from_raw"]) == 0
        assert int(rows[(model, "all")]["n_pairs_nan_energy"]) == n_nan


def test_committed_inputs_score_and_reproduce_artefacts(committed_run):
    out, r = committed_run
    assert r.returncode == 0, r.stdout + r.stderr
    names = sorted(p.name for p in RESULTS.glob("decisions_*.csv"))
    assert len(names) == 6
    for n in names + ["summary.csv", "summary.json"]:
        assert _sha(out / n) == _sha(RESULTS / n), f"{n} differs from committed"
