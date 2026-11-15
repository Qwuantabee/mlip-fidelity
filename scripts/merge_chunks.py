#!/usr/bin/env python3
"""Merge persisted per-chunk raw results into one model file, and say what is missing."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dir", default=str(ROOT/"results"))
a = ap.parse_args()
d = Path(a.dir)
import re
# Match ONLY the range-named chunks (chunk_<model>_<6d>_<6d>.json). A legacy
# ordinal name like chunk_<model>_000.json sorts BEFORE every range file
# ('.' < '0'), and with first-occurrence-wins it silently supplied the stale
# energy for those conformers while the fresh chunks were discarded.
pat = re.compile(rf"^chunk_{re.escape(a.model)}_\d{{6}}_\d{{6}}\.json$")
parts = sorted(p for p in d.glob(f"chunk_{a.model}_*.json") if pat.match(p.name))
if not parts:
    raise SystemExit(f"no chunk files for {a.model}")
# Deduplicate by conformer id. Chunk boundaries are not fixed - a re-run with a
# different --chunk writes ranges that OVERLAP earlier files (0-400 and 0-1500 both
# start at 0), and naive concatenation would count those conformers twice and
# silently inflate coverage. First occurrence wins, which is safe because every
# chunk is computed from the same committed geometry.
ids, ens, fails, meta, wall = [], [], [], {}, 0.0
seen: dict[str, int] = {}
dupes = 0
for p in parts:
    r = json.loads(p.read_text())
    for cid, e in zip(r["ids"], r["energies_ev"]):
        if cid in seen:
            dupes += 1
            continue
        seen[cid] = len(ids)
        ids.append(cid)
        ens.append(e)
    fails += r.get("failures", [])
    meta = r.get("meta", meta); wall += r.get("wall_seconds", 0.0)
import math
nan_count = int(sum(1 for e in ens if e != e))
# n_failed is the NaN count, not len(failures): run_model keeps only the first 10
# failure dicts, so len(failures) reported 10 for a model with 319 NaN conformers
# and the published summary inherited that wrong number.
merged = {"ok": True, "model": a.model, "meta": meta, "ids": ids,
          "energies_ev": ens, "n_ok": len(ids) - nan_count,
          "n_failed": nan_count, "failures": fails[:10], "wall_seconds": wall,
          "n_chunks_merged": len(parts)}
if len(ids) != len(ens):
    raise SystemExit(f"{a.model}: ids/energies length mismatch "
                     f"({len(ids)} vs {len(ens)}) - refusing to write")
(d/f"raw_{a.model}.json").write_text(json.dumps(merged)+"\n")
print(f"{a.model}: merged {len(parts)} chunks ({dupes} duplicate conformers "
      f"dropped), n_ok={merged['n_ok']} failed={merged['n_failed']} wall={wall:.0f}s")
