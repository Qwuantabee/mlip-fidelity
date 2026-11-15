#!/usr/bin/env python3
"""Score every installed model over the committed conformer set.

Design constraints that came from real failures:

* One subprocess per model, in that model's own venv (`envs/<name>`). Installing
  fairchem-core upgrades e3nn and silently breaks MACE checkpoint loading, so a
  shared environment is not an option.
* The conformer set is read from `results/conformers.jsonl`, which is committed.
  Every model therefore sees byte-identical geometry, and a model added later is
  comparable to one scored months earlier.
* A model that fails to build or crashes mid-way does not abort the sweep: its
  failure is recorded with the error text, because "model X would not run" is
  itself a result the README must be able to state.
* Nothing here interprets the chemistry. It writes raw energies; the decision
  scorer does the rest, so a scoring bug cannot hide inside the runner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# model name -> venv directory under envs/ (each may pin conflicting deps)
MODELS = {
    "mace_off_small": "mace",
    "mace_off_medium": "mace",
    "mace_mp_small": "mace",
    "orb_orbmol_v1_conservative": "orb",
    "orb_orb_v3_conservative_omol": "orb",
    "orb_v2": "orb",
    "rdkit_mmff": None,         # shared venv: rdkit is not torch-dependent
    "rdkit_uff": None,
}


def venv_python(venv: str | None) -> str:
    if venv is None:
        return "/home/ubuntu/.openclaw/workspace/venv/bin/python"
    p = ROOT / "envs" / venv / "bin" / "python"
    if not p.exists():
        raise SystemExit(f"missing venv {p}; build it first")
    return str(p)


def run_one(model: str, job: Path, out_dir: Path, device: str,
            limit: int | None) -> dict:
    venv = MODELS.get(model)
    if model not in MODELS:
        return {"ok": False, "model": model, "error": "unknown model"}
    py = venv_python(venv)
    cmd = [py, str(ROOT / "scripts" / "run_model.py"),
           "--model", model, "--input", str(job), "--device", device]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
    dt = time.time() - t0
    raw = p.stdout.strip().splitlines()
    payload = None
    for line in reversed(raw):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        return {"ok": False, "model": model,
                "error": f"no JSON from runner (rc={p.returncode})",
                "stderr_tail": p.stderr[-1200:], "seconds": dt}
    payload["wall_seconds"] = dt
    payload["runner_rc"] = p.returncode
    (out_dir / f"raw_{model}.json").write_text(json.dumps(payload) + "\n")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--conformers", default=str(ROOT / "results" / "conformers.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N conformers (0 = all)")
    ap.add_argument("--chunk", type=int, default=2000,
                    help="conformers per subprocess; bounds peak memory (a single "
                         "15k-conformer job plus torch OOM-killed this box)")
    args = ap.parse_args()

    models = args.models or list(MODELS)
    conf_path = Path(args.conformers)
    if not conf_path.exists():
        raise SystemExit(f"missing {conf_path}; run scripts/build_pairs.py first")
    recs = [json.loads(line) for line in conf_path.read_text().splitlines() if line.strip()]
    if args.limit:
        recs = recs[:args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(recs)} conformers, models: {models}, "
          f"chunks of {args.chunk}", flush=True)

    # Chunked execution. A single job file with 15k conformers plus torch inside the
    # runner exceeded this 19 GB machine and was OOM-killed (rc=137) after an
    # earlier identical run had succeeded, so the memory ceiling is not
    # reproducible at full size. Each chunk is a separate subprocess, merged after.
    chunk = args.chunk if args.chunk and args.chunk > 0 else len(recs)
    bounds = [(i, min(i + chunk, len(recs))) for i in range(0, len(recs), chunk)]

    summary = {}
    for model in models:
        ids_all: list[str] = []
        energies_all: list[float] = []
        failures: list[dict] = []
        model_meta: dict = {}
        wall = 0.0
        ok = True
        err = ""
        for ci, (lo, hi) in enumerate(bounds):
            # Persist each chunk the moment it finishes. An earlier full run was
            # OOM-killed at chunk 2 and lost ALL of chunk 1's work because results
            # were only merged at the end; a crash must cost one chunk, not the run.
            part_raw = out_dir / f"chunk_{model}_{ci:03d}.json"
            if part_raw.exists():
                saved = json.loads(part_raw.read_text())
                ids_all += saved["ids"]
                energies_all += saved["energies_ev"]
                failures += saved.get("failures", [])
                model_meta = saved.get("meta", {})
                wall += saved.get("wall_seconds", 0.0)
                print(f"-- {model} chunk {ci+1}/{len(bounds)} (cached)", flush=True)
                continue
            part = out_dir / f"_job_{model}_{ci}.json"
            part.write_text(json.dumps({"conformers": recs[lo:hi]}))
            print(f"-- {model} chunk {ci+1}/{len(bounds)} "
                  f"({hi-lo} conformers) ...", flush=True)
            res = run_one(model, part, out_dir, args.device, None)
            part.unlink(missing_ok=True)
            if not res.get("ok"):
                ok = False
                err = res.get("error", "failed")
                print(f"   FAILED: {err[:200]}", flush=True)
                break
            part_raw.write_text(json.dumps(res))
            ids_all += res["ids"]
            energies_all += res["energies_ev"]
            failures += res.get("failures", [])
            model_meta = res.get("meta", {})
            wall += res.get("wall_seconds", 0.0)
        if not ok:
            summary[model] = {"ok": False, "model": model, "error": err}
            continue
        merged = {"ok": True, "model": model, "meta": model_meta,
                  "ids": ids_all, "energies_ev": energies_all,
                  "n_ok": int(sum(1 for e in energies_all if e == e)),
                  "n_failed": len(failures), "failures": failures[:10],
                  "wall_seconds": wall}
        (out_dir / f"raw_{model}.json").write_text(json.dumps(merged) + "\n")
        print(f"   ok  n={merged['n_ok']} failed={merged['n_failed']} "
              f"{wall:.1f}s total", flush=True)
        summary[model] = {k: v for k, v in merged.items()
                          if k not in ("ids", "energies_ev", "failures")}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ok = [m for m, s in summary.items() if s.get("ok")]
    print(f"\nran {len(ok)}/{len(models)}: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
