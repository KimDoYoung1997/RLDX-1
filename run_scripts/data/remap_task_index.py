#!/usr/bin/env python3
"""Make a language-probe variant of a subtask dataset.

Copies the dataset and rewrites the per-frame parquet task_index column with a
fixed mapping — either a permutation (--map "1:2,2:3,3:1", the wrong-language
probe) or a constant (--set-all 0, the whole-sentence probe). Arrow-level
column swap, schema and all other columns untouched. tasks.jsonl already
contains every index used, so nothing else changes.
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--map", dest="mapping")
g.add_argument("--set-all", dest="set_all", type=int)
args = ap.parse_args()

mapping = None
if args.mapping:
    mapping = {int(k): int(v) for k, v in (p.split(":") for p in args.mapping.split(","))}

assert not args.out.exists(), f"{args.out} exists"
shutil.copytree(args.src, args.out)

info = json.loads((args.out / "meta" / "info.json").read_text())
episodes = [json.loads(l) for l in (args.out / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
counts: dict[int, int] = {}
for ep in episodes:
    idx = ep["episode_index"]
    p = args.out / info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx)
    t = pq.read_table(p)
    col = t.schema.get_field_index("task_index")
    field = t.schema.field(col)
    vals = t.column("task_index").to_pylist()
    new = [args.set_all] * len(vals) if mapping is None else [mapping.get(v, v) for v in vals]
    for v in new:
        counts[v] = counts.get(v, 0) + 1
    t2 = t.set_column(col, field, pa.array(new, type=field.type))
    assert t2.schema.metadata == t.schema.metadata
    pq.write_table(t2, p)
print(f"episodes: {len(episodes)}, task_index counts: {dict(sorted(counts.items()))}")
print("OK")
