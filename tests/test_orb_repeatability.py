"""The held-out numbers' run-to-run error bar: measured, published and quoted.

Context: the round-2 charge fix (review R2, 2026-11-16) re-ran ORB-omol on the
Folmsbee set and it landed at 0.833, not the predicted 0.834, because re-running
unchanged neutral compounds flipped 187 of their 20,436 pair decisions. That error
bar lived only in a build report while the README read "0.823 -> 0.833" as settled.
scripts/orb_repeatability.py now measures it - fresh repeats on a seeded sample,
plus the two committed full-set runs - into results/orb_repeatability.json and
publishes a block into results/heldout_folmsbee_summary.json.

These tests recompute the statistics without a GPU, keep artefact, summary and
README in step, and hold the MMFF94 control to bit-identity: if the CPU control
moves, the harness is broken and nothing it says about the GPU models holds.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "orb_repeatability.json"
SUMMARY = ROOT / "results" / "heldout_folmsbee_summary.json"
BENCH = ROOT / "results" / "heldout_folmsbee.json"
README = ROOT / "README.md"
ORB, MACE, MMFF = "orb_orbmol_v1_conservative", "mace_off_small", "rdkit_mmff"
KCAL_PER_EV = 627.5094740631 / 27.211386245988


@pytest.fixture(scope="module")
def rep():
    spec = importlib.util.spec_from_file_location(
        "orb_repeatability", ROOT / "scripts" / "orb_repeatability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def art():
    assert ART.exists(), ("results/orb_repeatability.json is missing; "
                          "run scripts/orb_repeatability.py")
    return json.loads(ART.read_text())


@pytest.fixture(scope="module")
def bench():
    return json.loads(BENCH.read_text())


# --------------------------------------------- the statistics, on known answers
def _toy():
    ids = ["m|a", "m|b", "m|c"]
    pairs = [{"id_a": "m|a", "id_b": "m|b", "compound": "m", "ref_kcal": 0.5},
             {"id_a": "m|a", "id_b": "m|c", "compound": "m", "ref_kcal": -0.2},
             {"id_a": "m|b", "id_b": "m|c", "compound": "m", "ref_kcal": -0.7}]
    return ids, pairs


def test_identical_runs_are_bit_identical_with_nothing_flipped(rep):
    ids, pairs = _toy()
    e = [-100.0, -99.99, -100.001]
    s = rep.compare_runs(ids, {"r1": e, "r2": list(e)}, pairs)
    assert s["bit_identical"] and s["conformers_changed"] == 0
    assert s["pairs_flipped_max"] == 0 and s["accuracy_spread"] == 0.0
    assert s["max_abs_dE_kcal"] == 0.0


def test_a_nudged_energy_flips_exactly_the_pair_it_crosses(rep):
    ids, pairs = _toy()
    e1 = [-100.0, -99.99, -100.0001]   # c - a = -0.0001 eV: agrees with ref -0.2
    e2 = [-100.0, -99.99, -99.9999]    # c - a = +0.0001 eV: flipped; c - b still < 0
    s = rep.compare_runs(ids, {"r1": e1, "r2": e2}, pairs)
    assert s["conformers_changed"] == 1
    assert s["pairs_flipped_max"] == 1
    assert s["max_abs_dE_kcal"] == pytest.approx(0.0002 * KCAL_PER_EV, rel=1e-6)
    assert s["accuracy_per_run"]["r1"] == pytest.approx(1.0)
    assert s["accuracy_per_run"]["r2"] == pytest.approx(2 / 3)
    assert s["accuracy_spread"] == pytest.approx(1 / 3)
    assert s["comparisons"][0]["net_agree_change"] == -1


def test_a_failed_conformer_is_unchanged_only_if_it_fails_every_time(rep):
    ids, pairs = _toy()
    nan = float("nan")
    s = rep.compare_runs(ids, {"r1": [-100.0, nan, -100.001],
                               "r2": [-100.0, nan, -100.001]}, pairs)
    assert s["bit_identical"]
    s = rep.compare_runs(ids, {"r1": [-100.0, nan, -100.001],
                               "r2": [-100.0, -99.99, -100.001]}, pairs)
    assert not s["bit_identical"] and s["conformers_changed"] == 1
    # pairs that touch the failed conformer were not decided in r1, so cannot flip
    assert s["comparisons"][0]["pairs_decided_both"] == 1


def test_one_run_is_not_a_repeatability_measurement(rep):
    ids, pairs = _toy()
    with pytest.raises(ValueError):
        rep.compare_runs(ids, {"r1": [-100.0, -99.99, -100.001]}, pairs)


# ------------------------------------------------------------ the measurement
def test_the_sample_is_reproducible_bounded_and_covers_both_charge_states(rep, art, bench):
    s = art["sample"]
    compounds, ids, pairs = rep.sample_from_bench(bench, s["target_conformers"], s["seed"])
    assert compounds == s["compounds"] and ids == s["conformer_ids"]
    assert len(pairs) == s["n_pairs"]
    # the job every repeat ran is the job this bench produces today
    job = json.dumps({"conformers": [bench["conformers"][i] for i in ids]}).encode()
    assert hashlib.sha256(job).hexdigest() == s["job_sha256"]
    charged = {bench["conformers"][i]["compound"] for i in ids
               if bench["conformers"][i]["charge"] != 0}
    assert charged, "the sample must exercise ORB-omol's charge input"
    assert len(charged) == s["n_charged_compounds"]
    assert 200 <= s["n_conformers"] <= 1500 < len(bench["conformers"])


def test_the_mmff_control_is_bit_identical(art):
    """MMFF94 runs on the CPU through the same harness: it must not move at all."""
    assert art["control"]["model"] == MMFF
    m = art["models"][MMFF]["fresh_repeats"]
    assert art["control"]["bit_identical"] and m["bit_identical"]
    assert m["conformers_changed"] == 0 and m["pairs_flipped_max"] == 0
    assert m["accuracy_spread"] == 0.0
    # identical to the published MMFF energies too: the harness fed the same inputs
    assert art["control"]["bit_identical_to_published"]


@pytest.mark.parametrize("model", [ORB, MACE, MMFF])
def test_every_model_was_repeated_on_the_whole_sample(art, model):
    rec = art["energies"][model]
    n = art["sample"]["n_conformers"]
    assert len(rec["repeats"]) >= 2
    assert art["models"][model]["repeats"] == len(rec["repeats"])
    assert all(len(r) == n for r in rec["repeats"]) and len(rec["published"]) == n
    # the published run used the same library versions, so it is a fair repeat
    assert rec["version"] == rec["published_version"]


@pytest.mark.parametrize("model", [ORB, MACE, MMFF])
def test_the_statistics_follow_from_the_saved_energies(art, bench, model):
    """Recomputed with plain numpy, independently of compare_runs."""
    ids = art["sample"]["conformer_ids"]
    keep = set(art["sample"]["compounds"])
    pairs = [p for p in bench["pairs"] if p["compound"] in keep]
    pos = {c: k for k, c in enumerate(ids)}
    ia = np.array([pos[p["id_a"]] for p in pairs])
    ib = np.array([pos[p["id_b"]] for p in pairs])
    ref = np.sign([p["ref_kcal"] for p in pairs])
    E = np.array(art["energies"][model]["repeats"], dtype=float)
    stored = art["models"][model]["fresh_repeats"]

    same = (E == E[0]) | (np.isnan(E) & np.isnan(E[0]))
    assert int((~np.all(same, axis=0)).sum()) == stored["conformers_changed"]

    gaps = E[:, ib] - E[:, ia]
    accs = []
    for g in gaps:
        ok = np.isfinite(g)
        accs.append(float(((np.sign(g) == ref) & ok).sum() / ok.sum()))
    assert list(stored["accuracy_per_run"].values()) == pytest.approx(accs, abs=1e-15)
    assert stored["accuracy_spread"] == pytest.approx(max(accs) - min(accs), abs=1e-15)

    flips = []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            both = np.isfinite(gaps[i]) & np.isfinite(gaps[j])
            flips.append(int((both & (np.sign(gaps[i]) != np.sign(gaps[j]))).sum()))
    assert max(flips) == stored["pairs_flipped_max"]

    d = np.abs(E[:, None, :] - E[None, :, :]) * KCAL_PER_EV
    assert float(np.nanmax(d)) == pytest.approx(stored["max_abs_dE_kcal"], rel=1e-9, abs=1e-15)


def test_the_full_set_comparison_follows_from_the_committed_runs(rep, art):
    try:
        h = rep.history_comparison()
    except SystemExit as exc:          # a shallow clone has no c8a8693
        pytest.skip(f"git history unavailable: {exc}")
    assert h == art["history_full_set"]


def test_the_history_ends_at_the_published_figure(art):
    """If ORB-omol is re-scored, this artefact describes a run that is gone."""
    rows = {r["model"]: r for r in json.loads(SUMMARY.read_text())}
    assert art["history_full_set"]["accuracy_after_all"] == pytest.approx(
        rows[ORB]["acc_under_1kcal"], abs=1e-12)


def test_the_verdict_follows_from_the_numbers(rep, art):
    assert rep.verdict(art["models"], art["history_full_set"]) == art["orb_delta_verdict"]


# ------------------------------------------------------------- the publication
def test_the_summary_carries_the_measured_block(art):
    rows = {r["model"]: r for r in json.loads(SUMMARY.read_text())}
    for model in (ORB, MACE, MMFF):
        assert "repeatability" in rows[model], (
            f"{model}: no repeatability block in heldout_folmsbee_summary.json. "
            "score_heldout.py rewrites that file; re-run "
            "scripts/orb_repeatability.py --from-saved --publish")
        b, m = rows[model]["repeatability"], art["models"][model]["fresh_repeats"]
        assert b["source"] == "results/orb_repeatability.json"
        assert b["repeats"] == art["models"][model]["repeats"]
        for key in ("conformers_changed", "max_abs_dE_kcal", "median_abs_dE_kcal",
                    "pairs_flipped_max", "flipped_share_max", "accuracy_spread"):
            assert b[key] == m[key], (model, key)
        assert b["accuracy_per_repeat"] == list(m["accuracy_per_run"].values())
    orb = rows[ORB]["repeatability"]
    hist = art["history_full_set"]
    assert orb["full_set_two_committed_runs"]["pairs_flipped"] == \
        hist["neutral_identical_inputs"]["pairs_flipped"]
    assert orb["delta_survives_spread"] == art["orb_delta_verdict"]["survives"]


def _heldout_section() -> str:
    text = README.read_text()
    sec = text.split("## The held-out test", 1)[1].split("\n## ", 1)[0]
    return " ".join(sec.split())          # line wraps are presentation


def _quoted(art) -> list[str]:
    """Every figure the held-out section takes from orb_repeatability.json, as printed."""
    s = art["sample"]
    orb = art["models"][ORB]["fresh_repeats"]
    mace = art["models"][MACE]["fresh_repeats"]
    h = art["history_full_set"]
    neu, chg = h["neutral_identical_inputs"], h["charged_inputs_changed"]
    v = art["orb_delta_verdict"]
    changed = (f"all {orb['conformers_changed']}" if orb["conformers_changed"] == s["n_conformers"]
               else f"{orb['conformers_changed']} of {s['n_conformers']}")
    return [
        f"same {s['n_conformers']} conformers ({s['n_compounds']} seeded-random compounds, "
        f"{s['n_pairs']:,} pairs)",
        f"moved {changed} of ORB-omol's energies",
        f"largest |ΔE| {orb['max_abs_dE_kcal']:.3f} kcal/mol, median {orb['median_abs_dE_kcal']:.4f}",
        f"flipped up to {orb['pairs_flipped_max']} decisions between two runs "
        f"({orb['flipped_share_max']:.2%})",
        f"over {orb['accuracy_min']:.3f}–{orb['accuracy_max']:.3f} ({orb['accuracy_spread']:.4f})",
        f"moved {mace['conformers_changed']} energies by at most "
        f"{mace['max_abs_dE_kcal']:.1e} kcal/mol and flipped none",
        f"{neu['pairs_flipped']} of those {neu['n_pairs']:,} decisions ({neu['flipped_share']:.2%}; "
        f"largest gap change {neu['max_abs_dgap_kcal']:.3f} kcal/mol)",
        f"a net {abs(h['neutral_net_agree_change'])} decisions, "
        f"{abs(h['neutral_shift_on_all_pairs']):.4f} of the {h['n_pairs_all']:,}-pair accuracy",
        f"a predicted {h['accuracy_predicted_charged_only_rerun']:.3f} came out "
        f"{h['accuracy_after_all']:.3f}",
        f"error bar of about ±{v['full_set_run_to_run_shift']:.4f}",
        f"The {h['accuracy_before_all']:.3f} → {h['accuracy_after_all']:.3f} step "
        f"(+{v['delta_all']:.4f}, {v['delta_over_full_set_shift']:.1f}× that)",
        f"+{v['charged_gain_on_all_pairs']:.4f} from the charged pairs",
        f"a net {v['charged_net_agree_gain']} decisions, where noise could move at most "
        f"{v['charged_worst_case_noise_decisions']:.0f}",
        f"less {abs(v['neutral_noise_on_all_pairs']):.4f} of noise on the neutral ones",
        # the correction paragraph's charged-pair figures, now backed by the artefact
        f"from {chg['accuracy_decided_before']:.3f} to {chg['accuracy_decided_after']:.3f} on "
        f"those compounds' {chg['n_decided']:,} decided pairs",
        f"{chg['n_pairs']:,} of the {h['n_pairs_all']:,} pairs",
    ]


def test_the_readme_quotes_the_measured_error_bar(art):
    held = _heldout_section()
    assert "not bit-reproducible on the GPU" in held
    assert "results/orb_repeatability.json" in held
    for needle in _quoted(art):
        assert " ".join(needle.split()) in held, (
            f"the README's held-out section does not say {needle!r}, which is what "
            "results/orb_repeatability.json measures")


def test_the_readme_claim_matches_the_verdict(art):
    """If the step stops surviving its own noise, the README must say so plainly."""
    held = _heldout_section()
    if art["orb_delta_verdict"]["survives"]:
        assert "not separable from" not in held
    else:
        assert "not separable from" in held, (
            "the 0.823 -> 0.833 step no longer survives the measured spread; the "
            "README must say it is not separable from run-to-run noise")
