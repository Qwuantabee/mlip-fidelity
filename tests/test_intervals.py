"""Every clustered interval and design effect in the README has generating code.

Until this file existed, nine interval statements in the README (the pairwise
bootstrap, the Folmsbee compound-clustered intervals, the design effects, the ICC
and the MMFF94-offset intervals) had no script in the repo: a mutation of any of
them passed the suite. `scripts/intervals.py` now computes them from the committed
decision CSVs with a fixed seed. This test re-runs it, requires the committed
`results/intervals.json` to equal the recompute, and requires each README
sentence to print the JSON's value at the README's precision.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MINUS = "−"


def _signed(x: float, nd: int = 3) -> str:
    return ("+" if x >= 0 else MINUS) + f"{abs(x):.{nd}f}"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("intervals", ROOT / "scripts" / "intervals.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def iv(mod) -> dict:
    path = ROOT / "results" / "intervals.json"
    assert path.exists(), "results/intervals.json is missing; run scripts/intervals.py"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def readme() -> str:
    return " ".join((ROOT / "README.md").read_text().split())


def test_committed_json_equals_a_recompute(mod, iv):
    assert mod._fmt(mod.compute()) == mod._fmt(iv), (
        "results/intervals.json is stale: re-run scripts/intervals.py")


def test_pairwise_bootstrap(readme, iv):
    pw = iv["spice_pairwise"]
    assert f"{pw['n_pairs']:,} pairs all three models scored" in readme
    assert f"bootstrap over their {pw['n_molecules']:,} molecules" in readme
    for label, key in (("ORB−MACE", "orb_minus_mace"), ("MACE−MMFF", "mace_minus_mmff")):
        r = pw[key]
        want = f"{label} {_signed(r['diff'])} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        assert want in readme, f"README does not print {want!r}"


@pytest.fixture(scope="module")
def heldout_doc() -> str:
    """The held-out detail moved out of the README (docs/HELDOUT.md)."""
    return " ".join((ROOT / "docs" / "HELDOUT.md").read_text().split())


def test_folmsbee_compound_clustered_intervals(readme, heldout_doc, iv):
    fc = iv["folmsbee_cluster"]
    parts = [f"{name} [{fc[k]['cluster_ci'][0]:.3f}, {fc[k]['cluster_ci'][1]:.3f}]"
             for name, k in (("ORB", "orb"), ("MACE", "mace"), ("MMFF94", "mmff"))]
    want = f"{parts[0]}, {parts[1]} and {parts[2]}"
    assert want in heldout_doc, f"docs/HELDOUT.md does not print {want!r}"
    d = sorted(r["design_effect"] for r in fc.values())
    rng = f"design effect of {d[0]:.1f}–{d[-1]:.1f}"
    assert rng in heldout_doc, f"docs/HELDOUT.md does not print {rng!r}"
    assert f"effect {d[0]:.1f}–{d[-1]:.1f})" in readme, "Limits bullet design effect drifted"


def test_spice_design_effect_icc_and_width(readme, iv):
    sc = iv["spice_clustering"]
    for k in ("orb", "mace"):
        assert round(sc[k]["design_effect"]) == 2, f"{k}: design effect is no longer 'about 2'"
        assert f"{sc[k]['width_ratio']:.1f}x" == "1.4x", f"{k}: width ratio is no longer 1.4x"
        assert f"{sc[k]['icc']:.2f}" == "0.10", f"{k}: ICC is no longer about 0.10"
    assert "effect is about 2 (ICC about 0.10)" in readme
    assert "about 1.4x too narrow" in readme
    assert "1.4x wider than quoted" in readme


def test_mmff_offset_intervals(heldout_doc, iv):
    mo = iv["mmff_offset"]
    for name, k in (("ORB-omol", "orb"), ("MACE-OFF23", "mace")):
        r = mo[k]
        want = (f"**{name}: {_signed(r['estimate'])}** (95% CI {_signed(r['ci'][0])} "
                f"to {_signed(r['ci'][1])})")
        assert want in heldout_doc, f"docs/HELDOUT.md does not print {want!r}"
