"""End-to-end pipeline test on a miniature archive with known ground truth.

Ported from the code reviewer's scratch harness (was `/tmp/mlipreview/e2e/`). It
exists because the unit tests were not enough: the reviewer ran 55 single-edit
mutants against this repo and the suite killed only 21, while this pipeline test
killed 13 more. Nine scripts here had no test that actually RAN them, and that is
exactly where the real defects were (the kcal/mol-as-eV bug, the inter-fragment
flag, the float32 precision loss).

What it does: builds a tiny HDF5 in SPICE's real layout with energies we choose,
runs the repo's own `build_pairs.py` / `score.py` / `run_model.py` as subprocesses,
and asserts the invariants that a sign, unit, or id-mapping bug would break:

  * qm_gap equals E(B) - E(A) for the conformer indices the ids name;
  * an exported conformer is the one its id names, and is in Angstrom;
  * a perfect model scores exactly 1.0 and its gaps equal the QM gaps;
  * a negated model scores exactly 0.0 (catches a reversed sign anywhere);
  * a pair touching a missing conformer is EXCLUDED, not counted as agreement;
  * the classical baseline reaches the scorer in the right unit.

The last one is the regression guard for the bug that shipped: RDKit returns
kcal/mol, the runner stored it as eV, and score.py converted again, inflating every
MMFF gap by 23.06x. Sign accuracy was invariant to that, which is why 44 green
tests could not see it.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
BOHR = 0.52917721067
H2K = 627.5094740631
HA2EV = 27.211386245988

# Five heavy/light atoms; positions are perturbed per conformer so no two are equal.
BASE = np.array([[0, 0, 0], [1.2, 1.2, 1.2], [-1.2, -1.2, 1.2],
                 [-1.2, 1.2, -1.2], [1.2, -1.2, -1.2]], dtype=float)
# Offsets in mHa: sub-kcal spacings, so decision pairs actually exist.
SPEC = {
    "molA": ("SPICE Dipeptides Single Points Dataset v1.2",
             [0.0, 0.5, 1.0, 0.2, 1.4, 0.9, 0.7, 0.35]),
    "molB": ("SPICE PubChem Set 1 Single Points Dataset v1.2",
             [0.0, 0.3, 0.1, 1.2, 0.6]),
}


def _run(cmd):
    # Inherit the real environment and only add the flag. A hand-built
    # {"PATH": "/usr/bin:/bin"} dict dropped VIRTUAL_ENV and PYTHONPATH, which
    # made the subprocess use a different interpreter than the test process.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    arch = d / "mini.hdf5"
    rng = np.random.default_rng(1)
    truth = {}
    with h5py.File(arch, "w") as f:
        for key, (subset, offs) in SPEC.items():
            n = len(offs)
            g = f.create_group(key)
            g.create_dataset("atomic_numbers",
                             data=np.array([6, 1, 1, 1, 1], dtype="int16"))
            pos = np.stack([BASE + rng.normal(scale=0.05, size=BASE.shape)
                            for _ in range(n)]).astype("float32")
            energy = -40.0 + np.array(offs) * 1e-3
            g.create_dataset("conformations", data=pos)
            g.create_dataset("dft_total_energy", data=energy)
            g.create_dataset("subset", data=np.array([subset], dtype=object))
            g.create_dataset("smiles", data=np.array(["C"], dtype=object))
            truth[key] = (pos.astype(float), energy)
    out = d / "out"
    r = _run([PY, str(ROOT / "scripts" / "build_pairs.py"), "--archive", str(arch),
              "--out", str(out), "--max-per-subset", "0",
              "--max-pairs-per-molecule", "0",
              "--max-conformers-per-molecule", "5"])
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-800:]
    man = json.loads((out / "pairs_manifest.json").read_text())
    conf = {}
    for line in (out / "conformers.jsonl").read_text().splitlines():
        c = json.loads(line)
        conf[c["id"]] = c
    return dict(dir=d, out=out, man=man, conf=conf, truth=truth)


def _idx(cid):
    key, i = cid.rsplit("|", 1)
    return key, int(i)


def _score(world, tag, energies_by_id, name):
    out = world["out"]
    ids = sorted(world["conf"])
    raw = {"ok": True, "model": name,
           "meta": {"family": "fake", "licence": "test"},
           "ids": ids, "energies_ev": [energies_by_id(i) for i in ids],
           "n_ok": 0, "n_failed": 0}
    sd = world["dir"] / ("score_" + tag)
    sd.mkdir(exist_ok=True)
    (sd / f"raw_{name}.json").write_text(json.dumps(raw))
    r = _run([PY, str(ROOT / "scripts" / "score.py"),
              "--manifest", str(out / "pairs_manifest.json"),
              "--raw-glob", str(sd / "raw_*.json"), "--out", str(sd)])
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-800:]
    rows = list(csv.DictReader(open(sd / "summary.csv")))
    dec = list(csv.DictReader(open(sd / f"decisions_{name}_all.csv")))
    return rows, dec


def test_manifest_gap_is_E_B_minus_E_A_for_the_ids_it_names(world):
    """Sign AND the id -> conformer-index mapping, against the archive's own numbers."""
    assert world["man"]["n_pairs"] > 10
    for p in world["man"]["pairs"]:
        key, ia = _idx(p["id_a"])
        key2, ib = _idx(p["id_b"])
        assert key == key2 == p["molecule"]
        E = world["truth"][key][1]
        assert p["qm_gap_kcal"] == pytest.approx((E[ib] - E[ia]) * H2K, abs=2e-6), \
            (p["id_a"], p["id_b"])


def test_exported_conformer_is_the_one_its_id_names_and_in_angstrom(world):
    for cid, c in world["conf"].items():
        key, i = _idx(cid)
        pos, E = world["truth"][key]
        assert c["energy_hartree"] == pytest.approx(E[i], abs=1e-12)
        np.testing.assert_allclose(np.array(c["positions"]), pos[i] * BOHR, atol=1.1e-5)


def test_perfect_model_scores_one_and_gap_equals_qm_gap(world):
    en = {i: c["energy_hartree"] * HA2EV for i, c in world["conf"].items()}
    rows, dec = _score(world, "perfect", en.__getitem__, "perfect")
    r = [x for x in rows if x["scope"] == "all"][0]
    expected = sum(1 for p in world["man"]["pairs"] if abs(p["qm_gap_kcal"]) > 0.01)
    assert int(r["n_decided"]) == expected
    assert float(r["accuracy"]) == 1.0
    for d in dec:
        assert float(d["model_gap_kcal"]) == pytest.approx(
            float(d["qm_gap_kcal"]), abs=1e-4)


def test_negated_model_scores_zero(world):
    """A reversed sign anywhere in the chain shows up as exactly 0.0 accuracy."""
    en = {i: -c["energy_hartree"] * HA2EV for i, c in world["conf"].items()}
    rows, _ = _score(world, "neg", en.__getitem__, "negated")
    assert float([x for x in rows if x["scope"] == "all"][0]["accuracy"]) == 0.0


def test_pairs_touching_a_missing_conformer_are_excluded(world):
    """Absent must mean excluded. Counting it as agreement would inflate accuracy."""
    victim = sorted(world["conf"])[0]
    deg = sum(1 for p in world["man"]["pairs"] if victim in (p["id_a"], p["id_b"]))
    assert deg > 0

    def en(i):
        return float("nan") if i == victim else world["conf"][i]["energy_hartree"] * HA2EV

    rows, dec = _score(world, "nan", en, "nanone")
    r = [x for x in rows if x["scope"] == "all"][0]
    assert int(r["n_pairs"]) == len(world["man"]["pairs"]) - deg
    assert all(victim not in (d["id_a"], d["id_b"]) for d in dec)


def test_classical_baseline_reaches_the_scorer_in_the_right_unit(world):
    """The 23x-inflation regression: RDKit's gap must survive to the decisions CSV.

    This is the test that would have caught the shipped bug. It runs the real
    `run_model.py rdkit_mmff` and compares the scored model gap to RDKit's own
    kcal/mol difference, so a stored-kcal-labelled-eV defect fails here.
    """
    pytest.importorskip("rdkit", reason="the classical-baseline unit check needs RDKit")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdDetermineBonds
    RDLogger.DisableLog("rdApp.*")

    ids = sorted(world["conf"])
    job = world["dir"] / "job.json"
    job.write_text(json.dumps({"conformers": [world["conf"][i] for i in ids]}))
    r = _run([PY, str(ROOT / "scripts" / "run_model.py"),
              "--model", "rdkit_mmff", "--input", str(job)])
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert payload["ok"], payload
    energies = dict(zip(payload["ids"], payload["energies_ev"]))
    assert all(e == e for e in energies.values()), "bond perception failed on the toy molecule"

    def kcal_direct(cid):
        c = world["conf"][cid]
        m = Chem.RWMol()
        for z in c["atomic_numbers"]:
            m.AddAtom(Chem.Atom(int(z)))
        cf = Chem.Conformer(m.GetNumAtoms())
        for k, xyz in enumerate(c["positions"]):
            cf.SetAtomPosition(k, tuple(xyz))
        m.AddConformer(cf)
        mol = m.GetMol()
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        return AllChem.MMFFGetMoleculeForceField(
            mol, props, ignoreInterfragInteractions=False).CalcEnergy()   # kcal/mol

    rows, dec = _score(world, "mmff", energies.__getitem__, "mmffreal")
    checked = 0
    for d in dec[:12]:
        want = kcal_direct(d["id_b"]) - kcal_direct(d["id_a"])
        got = float(d["model_gap_kcal"])
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6), (
            f"MMFF gap in the decisions CSV is {got:.4f} kcal/mol, RDKit says "
            f"{want:.4f} (ratio {got / want if want else float('nan'):.3f}) - this is "
            f"the kcal/mol-as-eV inflation (23.06x) or a reverting of it")
        checked += 1
    assert checked > 0
