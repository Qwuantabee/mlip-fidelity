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
parts = sorted(d.glob(f"chunk_{a.model}_*.json"))
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
merged = {"ok": True, "model": a.model, "meta": meta, "ids": ids,
          "energies_ev": ens, "n_ok": int(sum(1 for e in ens if e == e)),
          "n_failed": len(fails), "failures": fails[:10], "wall_seconds": wall,
          "n_chunks_merged": len(parts)}
(d/f"raw_{a.model}.json").write_text(json.dumps(merged)+"\n")
print(f"{a.model}: merged {len(parts)} chunks ({dupes} duplicate conformers "
      f"dropped), n_ok={merged['n_ok']} failed={merged['n_failed']} wall={wall:.0f}s")
