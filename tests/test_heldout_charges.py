"""Held-out Folmsbee builder: charged compounds must reach the models charged.

Review R2 (2026-11-16) found that build_heldout_folmsbee.py wrote charge 0, spin 1
for every conformer, although 74 of the 690 scored compounds are ions (1,943 of
22,379 pairs). 563 of their 602 conformers then had an odd electron count but were
run as closed-shell singlets, and ORB scored 0.668 on those pairs against 0.795
with the true charges. Nothing failed: the energies were finite and plausible.

These tests pin the fix at three levels: the charge-file parser, the builder run
end to end on a miniature Folmsbee tree, and the committed artefacts.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
BENCH = ROOT / "results" / "heldout_folmsbee.json"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_heldout_folmsbee", ROOT / "scripts" / "build_heldout_folmsbee.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_charge_file_parses_and_rejects_garbage(tmp_path):
    b = _load_builder()
    f = tmp_path / "CHG-charges.txt"
    f.write_text("astex_1gm8 CHARGE=-1 \nastex_1hnn CHARGE=1 \nastex_1mmv CHARGE=2 \n"
                 "omega_x CHARGE=0 \n")
    assert b.read_charges(f) == {"astex_1gm8": -1, "astex_1hnn": 1,
                                 "astex_1mmv": 2, "omega_x": 0}
    # A line that does not parse must raise: skipping it would turn that ion
    # into a neutral without a trace.
    f.write_text("astex_1gm8 CHARGE=-1\nastex_1hnn charge plus one\n")
    with pytest.raises(ValueError):
        b.read_charges(f)


def test_unlisted_charged_compound_is_not_scored_as_neutral():
    b = _load_builder()
    charges = {"ion": 1}
    assert b.compound_charge("Neutral_jobs", "anything", charges) == 0
    assert b.compound_charge("CHG_jobs", "ion", charges) == 1
    # The regression: a CHG_jobs compound with no listed charge used to be 0.
    assert b.compound_charge("CHG_jobs", "unlisted", charges) is None


def test_spin_follows_electron_parity():
    b = _load_builder()
    methylammonium = [6, 7] + [1] * 6            # 19 protons
    assert b.spin_multiplicity(methylammonium, 1) == 1    # 18 electrons
    assert b.spin_multiplicity(methylammonium, 0) == 2    # 19: what charge 0 implied


def _write_mol(path: Path, smiles: str, seed: int) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(m, str(path))


def test_builder_assigns_true_charges_and_excludes_unknowns(tmp_path):
    """Run the real builder on a three-compound Folmsbee-shaped tree."""
    root = tmp_path / "folmsbee"
    geo = root / "geometries"
    rows = ["name,geom,natoms,dlpno"]
    spec = {"neutral_meoh": ("Neutral_jobs", "CO"),
            "cation_mea": ("CHG_jobs", "C[NH3+]"),
            "cation_unlisted": ("CHG_jobs", "CC[NH3+]")}
    for name, (sub, smi) in spec.items():
        for k, e in enumerate((100.0, 100.4)):
            g = f"rmsd{k}-opt.mol"
            _write_mol(geo / sub / name / g, smi, seed=k + 1)
            rows.append(f"{name},{g},0,{e}")
    (root / "data-final.csv").write_text("\n".join(rows) + "\n")
    (geo / "CHG-charges.txt").write_text("cation_mea CHARGE=1 \n")

    out = tmp_path / "bench.json"
    r = subprocess.run([PY, str(ROOT / "scripts" / "build_heldout_folmsbee.py"),
                        "--root", str(root), "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    bench = json.loads(out.read_text())
    by_compound = {}
    for c in bench["conformers"].values():
        by_compound.setdefault(c["compound"], set()).add((c["charge"], c["spin"]))
    assert by_compound["neutral_meoh"] == {(0, 1)}
    assert by_compound["cation_mea"] == {(1, 1)}, by_compound
    # No listed charge -> excluded and counted, never run as a neutral.
    assert "cation_unlisted" not in by_compound
    assert bench["n_compounds_excluded_no_charge"] == 1
    assert bench["compounds_excluded_no_charge"] == ["cation_unlisted"]
    assert {p["compound"] for p in bench["pairs"]} == {"neutral_meoh", "cation_mea"}


@pytest.fixture(scope="module")
def bench():
    return json.loads(BENCH.read_text())


def test_committed_bench_has_no_odd_electron_singlets(bench):
    """The shipped artefact had 563 conformers with an odd electron count run as
    singlets (the signature of a cation passed as charge 0)."""
    bad = [c["id"] for c in bench["conformers"].values()
           if (sum(c["atomic_numbers"]) - c["charge"]) % 2 == 1 and c["spin"] == 1]
    assert not bad, f"{len(bad)} odd-electron conformers run as singlets, e.g. {bad[:3]}"


def test_committed_bench_carries_the_charged_compounds(bench):
    charged = {c["compound"] for c in bench["conformers"].values() if c["charge"] != 0}
    assert len(charged) == bench["n_charged_compounds"] > 0
    assert all("charge_source" in c for c in bench["conformers"].values())
    assert bench["n_compounds_excluded_no_charge"] == len(
        bench["compounds_excluded_no_charge"])


@pytest.mark.parametrize("model", ["orb_orbmol_v1_conservative", "mace_off_small",
                                   "rdkit_mmff"])
def test_committed_heldout_energies_were_computed_with_the_bench_charges(bench, model):
    raw = json.loads((ROOT / "results" / f"heldout_folmsbee_raw_{model}.json").read_text())
    assert raw["ids"] == sorted(bench["conformers"])
    assert len(raw["energies_ev"]) == len(raw["ids"])
    want = [bench["conformers"][i]["charge"] for i in raw["ids"]]
    assert raw["charge"] == want
    want_spin = [bench["conformers"][i]["spin"] for i in raw["ids"]]
    assert raw["spin"] == want_spin
