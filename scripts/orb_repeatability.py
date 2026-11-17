#!/usr/bin/env python3
"""Run-to-run repeatability of the held-out Folmsbee energies, measured.

Why this exists: the round-2 charge fix (review R2, 2026-11-16) re-ran all three
models on the Folmsbee set. ORB-omol was predicted to land at 0.834 and landed at
0.833, because re-running compounds whose inputs had NOT changed (the neutral
ones) flipped 187 of their 20,436 pair decisions. A model whose energies move
between identical runs puts an error bar on every accuracy it is quoted at, and
that error bar lived only in a build report while the README quoted
"0.823 -> 0.833" as if the whole step were the fix.

What it does:
  1. Draws a bounded, seeded sample of held-out compounds, stratified by charge
     (neutral / charged) in proportion to their share of conformers, and keeps
     every conformer and every pair of the drawn compounds.
  2. Runs each model N times on that sample through exactly the published path:
     scripts/run_model.py in the model's own venv (score_heldout.py's own table),
     one fresh process per repeat, the same job file every time (its sha256 is
     recorded, so "identical inputs" is checkable, not asserted).
  3. Reports per model: conformers whose energy changed at all, max and median
     |dE| between repeats, pair decisions flipped and their share, and the
     accuracy computed from each repeat with the published table's definition
     (sign agreement over every scored sub-kcal pair). The same comparison is
     repeated against the PUBLISHED energies (results/heldout_folmsbee_raw_*.json)
     so the published run counts as one more observation.
  4. Re-derives the full-set figure the build report quoted (187 of 20,436) from
     the two committed runs in git history, whose neutral compounds had
     identical inputs (verified here, conformer by conformer).

MMFF94 is the control. It runs on the CPU through the same harness and must come
out bit-identical; if it does not, the harness is non-deterministic, and that is
the finding: the script exits non-zero and publishes nothing.

    python scripts/orb_repeatability.py --publish                # GPU, a few minutes
    python scripts/orb_repeatability.py --from-saved --publish   # no GPU

--publish writes a "repeatability" block into each model's row of
results/heldout_folmsbee_summary.json. score_heldout.py rewrites that file
without it, so re-run "--from-saved --publish" after any re-scoring
(tests/test_orb_repeatability.py fails until you do).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BENCH = RESULTS / "heldout_folmsbee.json"
SUMMARY_JSON = RESULTS / "heldout_folmsbee_summary.json"
OUT = RESULTS / "orb_repeatability.json"
ORB = "orb_orbmol_v1_conservative"
CONTROL = "rdkit_mmff"
SEED = 20261116
TARGET_CONFORMERS = 600
# The committed run before the charge fix and the one that fixed it. Their
# neutral compounds had identical inputs (checked in history_comparison), so the
# pair is a free, full-set repeat of ORB-omol.
HISTORY_BEFORE, HISTORY_AFTER = "c8a8693", "1368db0"
TIE_KCAL = 0.01          # decisions.decision_scores' reference near-tie cut-off


def _load_scorer():
    """score_heldout.py's model table, venv lookup and unit constants.

    Imported rather than copied so this harness cannot drift from the path that
    produced the published numbers.
    """
    spec = importlib.util.spec_from_file_location(
        "score_heldout", ROOT / "scripts" / "score_heldout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SH = _load_scorer()


def to_kcal(ev):
    """eV -> kcal/mol, the same expression score_heldout.py uses for gaps."""
    return np.asarray(ev, dtype=float) / SH.HA2EV * SH.KCAL_PER_HA


# --------------------------------------------------------------------- sample
def select_compounds(conformers: dict, target: int, seed: int) -> list[str]:
    """Seeded compound sample, stratified neutral / charged by conformer share.

    Whole compounds are drawn so every pair of a drawn compound is scored; each
    stratum takes compounds in a seeded random order until it holds its share of
    the target conformer count.
    """
    n_conf: dict[str, int] = defaultdict(int)
    charge: dict[str, int] = {}
    for c in conformers.values():
        n_conf[c["compound"]] += 1
        charge[c["compound"]] = int(c.get("charge", 0))
    total = sum(n_conf.values())
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for neutral in (True, False):
        stratum = sorted(k for k in n_conf if (charge[k] == 0) == neutral)
        want = round(target * sum(n_conf[k] for k in stratum) / total)
        got = 0
        for i in rng.permutation(len(stratum)):
            if got >= want:
                break
            chosen.append(stratum[int(i)])
            got += n_conf[stratum[int(i)]]
    return sorted(chosen)


def sample_from_bench(bench: dict, target: int, seed: int):
    compounds = set(select_compounds(bench["conformers"], target, seed))
    ids = sorted(k for k, c in bench["conformers"].items() if c["compound"] in compounds)
    pairs = [p for p in bench["pairs"] if p["compound"] in compounds]
    return sorted(compounds), ids, pairs


# ---------------------------------------------------------------- statistics
def _same(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bitwise equality per conformer; two failed (NaN) energies count as equal."""
    return (x == y) | (np.isnan(x) & np.isnan(y))


def compare_runs(ids: list[str], runs: dict[str, list[float]], pairs: list[dict]) -> dict:
    """Repeatability of one model across runs on identical inputs.

    runs maps a label to energies in eV aligned with ids. Accuracy uses the
    published held-out definition (score_heldout.py's acc_under_1kcal): a pair
    agrees when sign(model gap) == sign(reference gap), over every pair whose two
    energies are finite. A flip is a pair scored in both runs whose model sign
    differs between them.
    """
    labels = list(runs)
    if len(labels) < 2:
        raise ValueError("need at least two runs to measure repeatability")
    E = {k: np.asarray(v, dtype=float) for k, v in runs.items()}
    for k in labels:
        if E[k].shape != (len(ids),):
            raise ValueError(f"run {k} has {E[k].shape} energies for {len(ids)} ids")
    pos = {cid: n for n, cid in enumerate(ids)}
    ia = np.array([pos[p["id_a"]] for p in pairs], dtype=int)
    ib = np.array([pos[p["id_b"]] for p in pairs], dtype=int)
    ref = np.array([p["ref_kcal"] for p in pairs], dtype=float)

    gaps = {k: to_kcal(E[k][ib] - E[k][ia]) for k in labels}
    per_run = {}
    for k in labels:
        ok = np.isfinite(gaps[k])
        agree = ok & (np.sign(gaps[k]) == np.sign(ref))
        per_run[k] = {"n_pairs_scored": int(ok.sum()), "n_agree": int(agree.sum()),
                      "accuracy": float(agree.sum() / ok.sum()) if ok.any() else float("nan")}

    changed_any = np.zeros(len(ids), dtype=bool)
    d_pool, comparisons = [], []
    for a, b in itertools.combinations(labels, 2):
        same = _same(E[a], E[b])
        changed_any |= ~same
        d = np.abs(to_kcal(E[a] - E[b]))
        d = d[np.isfinite(d)]
        d_pool.append(d)
        both = np.isfinite(gaps[a]) & np.isfinite(gaps[b])
        flipped = both & (np.sign(gaps[a]) != np.sign(gaps[b]))
        dg = np.abs(gaps[a] - gaps[b])[both]
        comparisons.append({
            "runs": [a, b],
            "conformers_changed": int((~same).sum()),
            "max_abs_dE_kcal": float(d.max()) if d.size else 0.0,
            "median_abs_dE_kcal": float(np.median(d)) if d.size else 0.0,
            "max_abs_dgap_kcal": float(dg.max()) if dg.size else 0.0,
            "pairs_decided_both": int(both.sum()),
            "pairs_flipped": int(flipped.sum()),
            "flipped_share": float(flipped.sum() / both.sum()) if both.any() else 0.0,
            "net_agree_change": per_run[b]["n_agree"] - per_run[a]["n_agree"],
        })
    pool = np.concatenate(d_pool) if d_pool else np.zeros(0)
    nonzero = pool[pool > 0]
    signs = np.vstack([np.sign(gaps[k]) for k in labels])
    scored_all = np.all(np.isfinite(signs), axis=0)
    unstable = scored_all & np.any(signs != signs[0], axis=0)
    accs = [per_run[k]["accuracy"] for k in labels]
    agrees = [per_run[k]["n_agree"] for k in labels]
    return {
        "runs": labels,
        "n_conformers": len(ids),
        "n_pairs": len(pairs),
        "conformers_changed": int(changed_any.sum()),
        "bit_identical": bool(not changed_any.any()),
        "max_abs_dE_kcal": float(pool.max()) if pool.size else 0.0,
        "median_abs_dE_kcal": float(np.median(pool)) if pool.size else 0.0,
        "median_abs_dE_kcal_changed_only": float(np.median(nonzero)) if nonzero.size else 0.0,
        "max_abs_dgap_kcal": max(c["max_abs_dgap_kcal"] for c in comparisons),
        "pairs_flipped_max": max(c["pairs_flipped"] for c in comparisons),
        "flipped_share_max": max(c["flipped_share"] for c in comparisons),
        "flipped_share_mean": float(np.mean([c["flipped_share"] for c in comparisons])),
        "pairs_unstable": int(unstable.sum()),
        "accuracy_per_run": dict(zip(labels, accs)),
        "accuracy_min": float(min(accs)),
        "accuracy_max": float(max(accs)),
        "accuracy_spread": float(max(accs) - min(accs)),
        "n_agree_spread": int(max(agrees) - min(agrees)),
        "comparisons": comparisons,
        "per_run": per_run,
    }


# ------------------------------------------------------------ full-set history
def _git_show(sha: str, path: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{path}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git show {sha}:{path} failed: {r.stderr.strip()}")
    return r.stdout


def history_comparison(before: str = HISTORY_BEFORE, after: str = HISTORY_AFTER) -> dict:
    """The two committed ORB-omol runs, compared on the pairs whose inputs match.

    The charge fix changed the inputs of the charged compounds only. Every
    neutral conformer is checked to carry the same atomic numbers, positions,
    charge (0) and spin in both committed benches; their pair decisions are
    then two runs of the same model on the same conformers.
    """
    import pandas as pd

    b_old = json.loads(_git_show(before, "results/heldout_folmsbee.json"))["conformers"]
    b_new = json.loads(_git_show(after, "results/heldout_folmsbee.json"))["conformers"]
    neutral_ids = [k for k, c in b_new.items() if int(c.get("charge", 0)) == 0]
    for k in neutral_ids:
        o, n = b_old.get(k), b_new[k]
        if (o is None or o["atomic_numbers"] != n["atomic_numbers"]
                or o["positions"] != n["positions"]
                or int(o.get("charge", 0)) != 0
                or int(o.get("spin", 1)) != int(n.get("spin", 1))):
            raise SystemExit(f"{k}: inputs differ between {before} and {after}; "
                             "the two runs are not repeats")
    neutral = {b_new[k]["compound"] for k in neutral_ids}
    charged = {c["compound"] for c in b_new.values() if int(c.get("charge", 0)) != 0}
    if neutral & charged:
        raise SystemExit("a compound has both neutral and charged conformers")

    path = f"results/heldout_folmsbee_decisions_{ORB}.csv"
    old = pd.read_csv(io.StringIO(_git_show(before, path)))
    new = pd.read_csv(io.StringIO(_git_show(after, path)))
    m = old.merge(new, on=["id_a", "id_b"], suffixes=("_before", "_after"),
                  validate="one_to_one")
    if not (len(m) == len(old) == len(new)):
        raise SystemExit("the two committed runs scored different pair sets")

    def block(df) -> dict:
        dec = df.ref_kcal_after.abs() > TIE_KCAL
        flipped = np.sign(df.model_kcal_before) != np.sign(df.model_kcal_after)
        return {
            "n_pairs": int(len(df)),
            "n_agree_before": int(df.agree_before.sum()),
            "n_agree_after": int(df.agree_after.sum()),
            "accuracy_before": float(df.agree_before.mean()),
            "accuracy_after": float(df.agree_after.mean()),
            "n_decided": int(dec.sum()),
            "accuracy_decided_before": float(df.agree_before[dec].mean()),
            "accuracy_decided_after": float(df.agree_after[dec].mean()),
            "pairs_flipped": int(flipped.sum()),
            "flipped_share": float(flipped.mean()),
            "gaps_changed": int((df.model_kcal_before != df.model_kcal_after).sum()),
            "max_abs_dgap_kcal": float((df.model_kcal_before - df.model_kcal_after).abs().max()),
        }

    neu = block(m[m.compound_after.isin(neutral)])
    chg = block(m[m.compound_after.isin(charged)])
    n_all = int(len(m))
    acc_before = float(m.agree_before.mean())
    acc_after = float(m.agree_after.mean())
    noise_net = neu["n_agree_after"] - neu["n_agree_before"]
    return {
        "before_commit": before, "after_commit": after,
        "note": "Committed ORB-omol decisions before and after the charge fix. "
                "Neutral compounds had identical inputs in both (checked "
                "conformer by conformer), so their pairs are a full-set repeat; "
                "library versions of the first run were not recorded.",
        "n_pairs_all": n_all,
        "accuracy_before_all": acc_before,
        "accuracy_after_all": acc_after,
        "delta_all": acc_after - acc_before,
        # what re-running only the charged compounds would have published: the
        # build report's prediction, short by exactly the neutral pairs' noise
        "accuracy_predicted_charged_only_rerun":
            (neu["n_agree_before"] + chg["n_agree_after"]) / n_all,
        "neutral_identical_inputs": neu,
        "charged_inputs_changed": chg,
        "neutral_net_agree_change": int(noise_net),
        "neutral_shift_on_all_pairs": float(noise_net / n_all),
    }


# ------------------------------------------------------------------- running
def run_model_once(model: str, job: Path, device: str) -> tuple[dict, float]:
    """One fresh process of scripts/run_model.py, exactly as score_heldout runs it."""
    py = SH.venv_python(SH.MODELS[model])
    cmd = [py, str(ROOT / "scripts" / "run_model.py"),
           "--model", model, "--input", str(job), "--device", device]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
    payload = None
    for line in reversed(r.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None or not payload.get("ok"):
        err = (payload or {}).get("error", r.stderr[-800:])
        raise SystemExit(f"{model}: run_model failed: {err}")
    return payload, time.time() - t0


def gpu_name() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def published_energies(model: str, ids: list[str], confs: dict) -> tuple[list[float], dict]:
    raw = json.loads((RESULTS / f"heldout_folmsbee_raw_{model}.json").read_text())
    e = dict(zip(raw["ids"], raw["energies_ev"]))
    q = dict(zip(raw["ids"], raw["charge"]))
    s = dict(zip(raw["ids"], raw["spin"]))
    for i in ids:
        if q[i] != int(confs[i].get("charge", 0)) or s[i] != int(confs[i].get("spin", 1)):
            raise SystemExit(f"{model}: published run used another charge/spin for {i}")
    return [e[i] for i in ids], raw.get("version")


# ----------------------------------------------------------------- reporting
def analyse(saved: dict, bench: dict) -> dict:
    """All statistics from the saved energies; no GPU."""
    ids = saved["sample"]["conformer_ids"]
    keep = set(saved["sample"]["compounds"])
    pairs = [p for p in bench["pairs"] if p["compound"] in keep]
    if len(pairs) != saved["sample"]["n_pairs"]:
        raise SystemExit("the bench's pairs no longer match the saved sample")
    models = {}
    for model, rec in saved["energies"].items():
        runs = {f"repeat_{n + 1}": e for n, e in enumerate(rec["repeats"])}
        fresh = compare_runs(ids, runs, pairs)
        with_pub = compare_runs(ids, {**runs, "published": rec["published"]}, pairs)
        models[model] = {"repeats": len(rec["repeats"]), "fresh_repeats": fresh,
                         "fresh_and_published": with_pub}
    return models


def verdict(models: dict, hist: dict) -> dict:
    """Does ORB-omol's published 0.823 -> 0.833 survive its measured run-to-run spread?

    Two tests, both on measured quantities:
      1. Scale. The full-set run-to-run shift is measured directly: the two
         committed runs on identical neutral inputs. Both 0.823 and 0.833 carry
         that error bar, so the step must exceed TWICE it.
      2. Source. The step is a sum of the charged pairs (inputs changed) and the
         neutral pairs (inputs identical, so pure noise). Even if every charged
         pair that could flip from noise - the largest flip share measured
         anywhere times their count - had flipped the same way, it must not
         account for the charged pairs' net gain.
    The sample's own accuracy spread is reported, not used here: it is a spread
    over the sample's pairs, not over the 22,379 the published figure averages.
    """
    orb = models[ORB]
    n_all = hist["n_pairs_all"]
    neu, chg = hist["neutral_identical_inputs"], hist["charged_inputs_changed"]
    delta = hist["delta_all"]
    shift = abs(hist["neutral_shift_on_all_pairs"])
    flip_share = max(orb["fresh_and_published"]["flipped_share_max"], neu["flipped_share"])
    gain = chg["n_agree_after"] - chg["n_agree_before"]
    worst_noise = flip_share * chg["n_pairs"]
    return {
        "delta_all": delta,
        "full_set_run_to_run_shift": shift,
        "delta_over_full_set_shift": delta / shift if shift > 0 else None,
        "sample_accuracy_spread": orb["fresh_repeats"]["accuracy_spread"],
        "sample_accuracy_spread_incl_published": orb["fresh_and_published"]["accuracy_spread"],
        "flipped_share_max": flip_share,
        "charged_net_agree_gain": int(gain),
        "charged_gain_on_all_pairs": gain / n_all,
        "neutral_noise_on_all_pairs": hist["neutral_shift_on_all_pairs"],
        "charged_worst_case_noise_decisions": worst_noise,
        "survives": bool(delta > 2 * shift and gain > worst_noise),
        "note": "Accuracy is recomputed from each run, so every spread is measured. "
                "The step 0.823 -> 0.833 survives if it exceeds twice the full-set "
                "run-to-run shift (both endpoints carry it) and the charged pairs' "
                "net gain exceeds the most their decisions could move by noise.",
    }


def publish_block(model: str, m: dict, hist: dict, ver: dict) -> dict:
    fr, fp = m["fresh_repeats"], m["fresh_and_published"]
    block = {
        "source": "results/orb_repeatability.json",
        "repeats": m["repeats"],
        "sample_conformers": fr["n_conformers"],
        "sample_pairs": fr["n_pairs"],
        "bit_identical": fr["bit_identical"],
        "conformers_changed": fr["conformers_changed"],
        "max_abs_dE_kcal": fr["max_abs_dE_kcal"],
        "median_abs_dE_kcal": fr["median_abs_dE_kcal"],
        "pairs_flipped_max": fr["pairs_flipped_max"],
        "flipped_share_max": fr["flipped_share_max"],
        "accuracy_per_repeat": list(fr["accuracy_per_run"].values()),
        "accuracy_spread": fr["accuracy_spread"],
        "accuracy_spread_incl_published": fp["accuracy_spread"],
    }
    if model == ORB:
        block["full_set_two_committed_runs"] = {
            "neutral_pairs": hist["neutral_identical_inputs"]["n_pairs"],
            "pairs_flipped": hist["neutral_identical_inputs"]["pairs_flipped"],
            "net_agree_change": hist["neutral_net_agree_change"],
            "shift_on_all_pairs": hist["neutral_shift_on_all_pairs"],
        }
        block["delta_all"] = ver["delta_all"]
        block["delta_over_full_set_shift"] = ver["delta_over_full_set_shift"]
        block["delta_survives_spread"] = ver["survives"]
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description="Run-to-run repeatability of held-out energies")
    ap.add_argument("--models", nargs="+", default=[ORB, "mace_off_small", CONTROL])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--target-conformers", type=int, default=TARGET_CONFORMERS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--from-saved", action="store_true",
                    help="recompute every statistic from the energies already in --out")
    ap.add_argument("--publish", action="store_true",
                    help="write the repeatability block into heldout_folmsbee_summary.json")
    args = ap.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be >= 2")
    if CONTROL not in args.models:
        raise SystemExit(f"{CONTROL} is the determinism control and must be run")

    bench = json.loads(BENCH.read_text())
    out_path = Path(args.out)
    if args.from_saved:
        saved = json.loads(out_path.read_text())
    else:
        compounds, ids, pairs = sample_from_bench(bench, args.target_conformers, args.seed)
        confs = bench["conformers"]
        job_bytes = json.dumps({"conformers": [confs[i] for i in ids]}).encode()
        charged = {confs[i]["compound"] for i in ids if int(confs[i].get("charge", 0)) != 0}
        saved = {
            "sample": {
                "seed": args.seed, "target_conformers": args.target_conformers,
                "stratified_by": "charge (neutral / charged), proportional to conformer share",
                "compounds": compounds, "n_compounds": len(compounds),
                "n_charged_compounds": len(charged),
                "conformer_ids": ids, "n_conformers": len(ids), "n_pairs": len(pairs),
                "job_sha256": hashlib.sha256(job_bytes).hexdigest(),
            },
            "environment": {"gpu": gpu_name(), "device": args.device},
            "energies": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / "job.json"
            job.write_bytes(job_bytes)
            for model in args.models:
                reps, secs, versions, meta = [], [], [], None
                for r in range(args.repeats):
                    if hashlib.sha256(job.read_bytes()).hexdigest() != saved["sample"]["job_sha256"]:
                        raise SystemExit("job file changed between repeats")
                    payload, dt = run_model_once(model, job, args.device)
                    if payload["ids"] != ids:
                        raise SystemExit(f"{model}: answered for a different conformer list")
                    reps.append(payload["energies_ev"])
                    secs.append(round(dt, 1))
                    versions.append(payload.get("version"))
                    meta = payload.get("meta")
                    print(f"-- {model} repeat {r + 1}/{args.repeats}: {len(ids)} "
                          f"conformers in {dt:.0f}s", flush=True)
                if any(v != versions[0] for v in versions):
                    raise SystemExit(f"{model}: library versions changed between repeats")
                pub, pub_version = published_energies(model, ids, confs)
                saved["energies"][model] = {
                    "repeats": reps, "seconds": secs, "version": versions[0],
                    "meta": meta, "published": pub, "published_version": pub_version,
                }

    models = analyse(saved, bench)
    ctl = models[CONTROL]
    control_ok = ctl["fresh_repeats"]["bit_identical"]
    hist = history_comparison()
    ver = verdict(models, hist)
    summary_rows = json.loads(SUMMARY_JSON.read_text())
    published_acc = {r["model"]: r.get("acc_under_1kcal") for r in summary_rows
                     if r.get("model") in models}
    out = {
        "what": "Run-to-run repeatability of held-out Folmsbee energies: the same model "
                "on the same conformers, re-run in fresh processes with identical inputs.",
        "reproduce": "python scripts/orb_repeatability.py --publish (GPU, a few minutes); "
                     "--from-saved --publish recomputes everything from the energies "
                     "stored here, without a GPU",
        "units": "energies in eV under 'energies'; every |dE| and gap in kcal/mol",
        "accuracy_definition": "sign(model gap) == sign(reference gap) over every scored "
                               "pair, as acc_under_1kcal in heldout_folmsbee_summary.json",
        "control": {"model": CONTROL, "bit_identical": control_ok,
                    "bit_identical_to_published":
                        ctl["fresh_and_published"]["bit_identical"]},
        "published_full_set_accuracy": published_acc,
        "models": models,
        "history_full_set": hist,
        "orb_delta_verdict": ver,
        **{k: saved[k] for k in ("sample", "environment", "energies")},
    }
    out_path.write_text(json.dumps(out, indent=1) + "\n")

    for model, m in models.items():
        fr, fp = m["fresh_repeats"], m["fresh_and_published"]
        print(f"{model:28s} repeats {m['repeats']} | changed {fr['conformers_changed']}/"
              f"{fr['n_conformers']} | max|dE| {fr['max_abs_dE_kcal']:.3g} median "
              f"{fr['median_abs_dE_kcal']:.3g} kcal | flipped max {fr['pairs_flipped_max']}/"
              f"{fr['n_pairs']} ({fr['flipped_share_max']:.2%}) | acc "
              f"{[round(a, 4) for a in fr['accuracy_per_run'].values()]} spread "
              f"{fr['accuracy_spread']:.4f} (with published {fp['accuracy_spread']:.4f})")
    neu = hist["neutral_identical_inputs"]
    print(f"full set, two committed runs: {neu['pairs_flipped']} of {neu['n_pairs']} "
          f"neutral decisions flipped, net {hist['neutral_net_agree_change']:+d}; "
          f"delta {hist['delta_all']:+.4f} = charged {ver['charged_gain_on_all_pairs']:+.4f}"
          f" + neutral noise {hist['neutral_shift_on_all_pairs']:+.4f}; survives: "
          f"{ver['survives']} ({ver['delta_over_full_set_shift']:.1f}x the full-set shift)")
    print(f"-> {out_path}")

    if not control_ok:
        print(f"CONTROL FAILED: {CONTROL} was not bit-identical across repeats, so the "
              "harness is non-deterministic; nothing published.", file=sys.stderr)
        return 3
    if args.publish:
        for row in summary_rows:
            if row.get("model") in models:
                row["repeatability"] = publish_block(row["model"], models[row["model"]],
                                                     hist, ver)
        SUMMARY_JSON.write_text(json.dumps(summary_rows, indent=2) + "\n")
        print(f"-> {SUMMARY_JSON} (repeatability block)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
