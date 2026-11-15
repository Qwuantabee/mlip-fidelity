"""Run one universal MLIP over a fixed conformer set and print JSON to stdout.

Why a subprocess and a separate venv per model: the packages genuinely conflict.
Installing `fairchem-core` upgrades e3nn to 0.6, which silently breaks MACE
checkpoint loading (`ValueError: too many values to unpack` inside e3nn codegen)
— a failure that only appears when a model is actually evaluated, not at import.
So each model gets its own venv (`envs/<model>`) and this script is the only
entry point, invoked as:

    envs/<venv>/bin/python scripts/run_model.py --model mace_off --input job.json

The job file lists conformers (atomic numbers, positions, an id) and the script
answers with one energy per conformer, in eV, plus the wall time and the model's
own metadata. Nothing about the chemistry is decided here; this only produces the
numbers the decision scorer consumes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback


def _atoms(rec: dict):
    from ase import Atoms
    import numpy as np
    at = Atoms(numbers=np.asarray(rec["atomic_numbers"], dtype=int),
               positions=np.asarray(rec["positions"], dtype=float))
    # Some calculators (ORB's molecular checkpoints) require explicit total charge
    # and spin multiplicity in info; supplying them unconditionally is harmless
    # for the others and keeps one conformer path for every model. SPICE molecules
    # are neutral singlets unless a future subset says otherwise.
    at.info["charge"] = int(rec.get("charge", 0))
    at.info["spin"] = int(rec.get("spin", 1))
    return at


def build_calculator(model: str, device: str = "cuda"):
    """Return (calculator, metadata). Raises with a clear message if unavailable."""
    if model == "mace_off_small" or model == "mace_off_medium":
        from mace.calculators import mace_off
        size = model.rsplit("_", 1)[1]
        calc = mace_off(model=size, device=device, default_dtype="float32")
        return calc, {"model": model, "family": "MACE-OFF23",
                      "licence": "Academic Software Licence (ASL), non-commercial",
                      "device": device}
    if model == "mace_mp_small":
        from mace.calculators import mace_mp
        calc = mace_mp(model="small", device=device, default_dtype="float32")
        return calc, {"model": model, "family": "MACE-MP-0",
                      "licence": "MIT", "device": device}
    if model.startswith("orb"):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        # Prefer the MOLECULE-trained checkpoints (orbmol_*, *_omol) for SPICE
        # chemistry; the materials-trained ones are kept as a domain-transfer
        # contrast and are named explicitly so the results never conflate them.
        variant = model[len("orb_"):] if model != "orb_v2" else "v2"
        fn_name = {
            "orbmol_v1_conservative": "orbmol_v1_conservative",
            "orb_v3_conservative_omol": "orb_v3_conservative_omol",
            "v2": "orb_v2",
            "v3_conservative_20_mpa": "orb_v3_conservative_20_mpa",
        }.get(variant)
        if fn_name is None or not hasattr(pretrained, fn_name):
            raise RuntimeError(f"orb variant {variant!r} not exposed by orb_models")
        model_obj, adapter = getattr(pretrained, fn_name)(device=device)
        trained_on = ("molecules" if ("mol" in fn_name) else "materials")
        return ORBCalculator(model_obj, adapter, device=device), {
            "model": model, "family": "ORB", "trained_on": trained_on,
            "checkpoint": fn_name,
            "licence": "ASL 3.0", "device": device}
    if model == "rdkit_uff":
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return "UFF", {"model": "rdkit_uff", "family": "classical force field",
                       "licence": "BSD", "device": "cpu"}
    raise RuntimeError(f"unknown model {model}")


def energies_uff(atoms, records, datums):
    """Classical baseline: build an RDKit molecule from the geometry and relax-free
    single-point UFF energy. Needs bonds, which come from RDKit's perception."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    out = []
    for rec in records:
        mol = Chem.RWMol()
        for z in rec["atomic_numbers"]:
            mol.AddAtom(Chem.Atom(int(z)))
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(rec["positions"]):
            conf.SetAtomPosition(i, tuple(float(v) for v in xyz))
        mol.AddConformer(conf)
        m = mol.GetMol()
        Chem.SanitizeMol(m)
        if not _uff_ok(m):
            from rdkit.Chem import rdDetermineBonds
            try:
                rdDetermineBonds.DetermineConnectivity(m, charge=0)
            except Exception:  # noqa: BLE001
                pass
        try:
            ff = AllChem.UFF_get_forcefield(m) if hasattr(AllChem, "UFF_get_forcefield") \
                else None
        except Exception:  # noqa: BLE001
            ff = None
        if ff is None:
            AllChem.MMFFOptimizeMolecule  # noqa: B018 - keep import warm
            props = AllChem.MMFFGetMoleculeProperties(m)
            ff = (AllChem.MMFFGetMoleculeForceField(m, props) if props else None)
        e = ff.CalcEnergy() if ff is not None else float("nan")
        out.append(e)
    return out


def _uff_ok(m) -> bool:
    return m is not None and m.GetNumBonds() > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="job json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    job = json.loads(open(args.input).read())
    records = job["conformers"]
    ids = [r["id"] for r in records]

    t0 = time.time()
    try:
        calc, meta = build_calculator(args.model, args.device)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "model": args.model,
                          "error": f"build failed: {exc}",
                          "traceback": traceback.format_exc()[-1500:]}))
        return 2

    energies_ev: list[float] = []
    failures: list[dict] = []
    if calc == "UFF":
        try:
            vals = energies_uff(None, records, None)
            energies_ev = list(vals)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "model": args.model,
                              "error": f"UFF failed: {exc}"}))
            return 2
    else:
        for rec in records:
            try:
                at = _atoms(rec)
                at.calc = calc
                energies_ev.append(float(at.get_potential_energy()))
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": rec["id"], "error": str(exc)[:300]})
                energies_ev.append(float("nan"))

    dt = time.time() - t0
    print(json.dumps({
        "ok": True,
        "model": args.model,
        "meta": meta,
        "ids": ids,
        "energies_ev": energies_ev,
        "n_ok": int(sum(1 for e in energies_ev if e == e)),
        "n_failed": len(failures),
        "failures": failures[:10],
        "seconds": dt,
        "version": _versions(),
    }))
    return 0


def _versions() -> dict:
    import importlib
    out = {}
    for name in ("torch", "mace", "orb_models", "fairchem", "ase", "e3nn"):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            continue
    return out


if __name__ == "__main__":
    sys.exit(main())
