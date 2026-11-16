#!/usr/bin/env python3
"""Score every model on the HELD-OUT MPCONF196 benchmark.

This is the clean out-of-sample counterpart to the SPICE numbers: neither
MACE-OFF23 nor ORB-omol has trained on MPCONF196, whereas both have seen (or
very likely seen) SPICE. MMFF94 was already out-of-sample.

The pair set is also deliberately different in kind. SPICE pairs were filtered to
|gap| < 1 kcal/mol, so they cover only the hardest bottom of the curve; MPCONF196
gaps run 0.07 to 39.9 kcal/mol (median 4.30). So this benchmark answers a
different question - does the model preserve ordering at *any* gap size - and the
two are reported side by side rather than blended.

Outputs:
  results/<tag>_decisions_<model>.csv   one row per pair
  results/<tag>_summary.csv             one row per model
  results/<tag>_raw_<model>.json        per-conformer energies (eV) and the charge
                                        and spin each conformer was run with, so
                                        the decisions can be re-derived without a
                                        GPU and the inputs audited
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlipfid.decisions import decision_scores, wilson_interval  # noqa: E402

HA2EV = 27.211386245988
KCAL_PER_HA = 627.5094740631
MODELS = {
    "mace_off_small": "mace",
    "orb_orbmol_v1_conservative": "orb",
    "rdkit_mmff": None,          # shared venv
}


def venv_python(name: str | None) -> str:
    if name is None:
        return sys.executable
    p = ROOT / "envs" / name / "bin" / "python"
    if not p.exists():
        raise SystemExit(f"missing venv {p}")
    return str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "results" / "heldout_mpconf196.json"))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--tag", default="heldout",
                    help="filename prefix, so two benchmarks do not overwrite "
                         "each other's summaries")
    ap.add_argument("--from-raw", action="store_true",
                    help="re-score from the committed <tag>_raw_<model>.json "
                         "energies instead of running the models (no GPU); the "
                         "raw file must carry the same ids, charges and spins as "
                         "the bench")
    args = ap.parse_args()

    bench = json.loads(Path(args.bench).read_text())
    confs = bench["conformers"]
    ids = sorted(confs)
    job = Path(args.out) / f"_{args.tag}_job.json"
    job_confs = [confs[i] for i in ids]
    job.write_text(json.dumps({"conformers": job_confs}))
    out_dir = Path(args.out)

    summary = []
    for model in args.models:
        if model not in MODELS:
            print(f"skip unknown model {model}")
            continue
        raw_path = out_dir / f"{args.tag}_raw_{model}.json"
        if args.from_raw:
            saved = json.loads(raw_path.read_text())
            if (saved["ids"] != [c["id"] for c in job_confs]
                    or saved["charge"] != [int(c.get("charge", 0)) for c in job_confs]
                    or saved["spin"] != [int(c.get("spin", 1)) for c in job_confs]):
                raise SystemExit(f"{raw_path.name} was computed for different "
                                 "conformers, charges or spins than the bench")
            payload = {"ok": True, "ids": saved["ids"],
                       "energies_ev": saved["energies_ev"],
                       "meta": saved.get("meta"), "version": saved.get("version")}
            print(f"-- {model}: re-scoring {raw_path.name} (no model run)", flush=True)
        else:
            py = venv_python(MODELS[model])
            cmd = [py, str(ROOT / "scripts" / "run_model.py"),
                   "--model", model, "--input", str(job), "--device", "cuda"]
            print(f"-- {model} on {len(ids)} held-out conformers ...", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
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
                err = (payload or {}).get("error", r.stderr[-400:])
                print(f"   FAILED: {err}")
                summary.append({"model": model, "error": str(err)[:200]})
                continue

        if len(payload["energies_ev"]) != len(payload["ids"]) or \
                payload["ids"] != [c["id"] for c in job_confs]:
            raise SystemExit(f"{model}: run_model answered for a different "
                             "conformer list than it was given")
        # Failures are the NaN count. run_model's own n_failed counts only ML
        # exceptions, so the classical baseline reported 0 failed conformers
        # while 668 (now 115) of its energies were NaN.
        n_nan = int(sum(1 for e in payload["energies_ev"] if e != e))
        n_finite = len(payload["energies_ev"]) - n_nan
        if not args.from_raw:
            raw_out = {
                "model": model, "bench": Path(args.bench).name,
                "meta": payload.get("meta"), "version": payload.get("version"),
                "ids": payload["ids"], "energies_ev": payload["energies_ev"],
                "charge": [int(c.get("charge", 0)) for c in job_confs],
                "spin": [int(c.get("spin", 1)) for c in job_confs],
                "n_ok": n_finite, "n_failed": n_nan,
            }
            raw_path.write_text(json.dumps(raw_out) + "\n")
        emap = {i: e for i, e in zip(payload["ids"], payload["energies_ev"]) if e == e}
        rows, qm, ml = [], [], []
        absent = 0
        for p in bench["pairs"]:
            a, b = emap.get(p["id_a"]), emap.get(p["id_b"])
            if a is None or b is None:
                absent += 1
                continue
            gap_kcal = (b - a) / HA2EV * KCAL_PER_HA
            rows.append({
                "model": model, "compound": p["compound"],
                "id_a": p["id_a"], "id_b": p["id_b"],
                "ref_kcal": p["ref_kcal"], "model_kcal": gap_kcal,
                "error_kcal": gap_kcal - p["ref_kcal"],
                "agree": int(np.sign(gap_kcal) == np.sign(p["ref_kcal"])),
            })
            qm.append(p["ref_kcal"])
            ml.append(gap_kcal)

        if not rows:
            summary.append({"model": model, "error": "no scorable pairs"})
            continue

        st = decision_scores(np.array(qm), np.array(ml))
        st.update({"model": model, "n_absent_pairs": absent,
                   "scored_conformers": n_finite,
                   "failed_conformers": n_nan,
                   "licence": payload.get("meta", {}).get("licence", "?")})
        # The held-out set spans every gap size, so also report the hard slice.
        small = [r for r in rows if abs(r["ref_kcal"]) <= 1.0]
        if small:
            k = sum(r["agree"] for r in small)
            lo, hi = wilson_interval(k, len(small))
            st.update({"n_under_1kcal": len(small),
                       "acc_under_1kcal": k / len(small),
                       "acc_under_1kcal_lo": lo, "acc_under_1kcal_hi": hi})
        summary.append(st)

        with open(out_dir / f"{args.tag}_decisions_{model}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"   acc {st['accuracy']:.3f} on {st['n_decided']} pairs | "
              f"median err {st['median_abs_err_kcal']:.2f} kcal | "
              f"under-1kcal acc {st.get('acc_under_1kcal', float('nan')):.3f}")

    if summary:
        keys = sorted({k for s in summary for k in s})
        with open(out_dir / f"{args.tag}_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)
    (out_dir / f"{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\n-> {out_dir/(args.tag + '_summary.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
