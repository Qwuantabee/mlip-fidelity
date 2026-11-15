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
    "rdkit_uff": None,          # shared venv: rdkit is not torch-dependent
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
    job = out_dir / "_job.json"
    job.write_text(json.dumps({"conformers": recs}))
    print(f"{len(recs)} conformers, models: {models}", flush=True)

    summary = {}
    for model in models:
        print(f"-- {model} ...", flush=True)
        res = run_one(model, job, out_dir, args.device, args.limit)
        if res.get("ok"):
            print(f"   ok  n={res['n_ok']} failed={res['n_failed']} "
                  f"{res['wall_seconds']:.1f}s", flush=True)
        else:
            print(f"   FAILED: {res.get('error', '')[:200]}", flush=True)
        summary[model] = {k: v for k, v in res.items()
                          if k not in ("ids", "energies_ev", "failures")}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ok = [m for m, s in summary.items() if s.get("ok")]
    print(f"\nran {len(ok)}/{len(models)}: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
