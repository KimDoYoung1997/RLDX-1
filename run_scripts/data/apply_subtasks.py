#!/usr/bin/env python3
"""Stamp per-frame subtask labels into a LeRobot v2.1 split dataset.

Takes human-written segment labels (episodes.jsonl with a `sub_tasks` list of
{start, end, text} per *merge* episode) and produces a copy of a train/holdout
split where the parquet `task_index` column varies per frame. That is the
entire intervention: the RLDX-1 loader already converts `task_index` to a task
string per frame (lerobot_episode_loader.py:350), so training code, modality
config and modality.json are untouched — subtask conditioning is purely a data
change.

Label indices are on the merge dataset; the split's episodes are re-numbered,
so each split episode is matched back to its merge index through
(source_repo, source_episode_index) in the two source_episodes.jsonl files,
with episode lengths asserted as a second check.

The parquet rewrite is Arrow-level on purpose: a pandas round-trip demotes
fixed_size_list<float>[57] to list<float> and drops the huggingface schema
metadata (measured during the 0826 merge). Only the task_index column is
replaced; every other column and the schema metadata are verified unchanged.

Usage:
    python run_scripts/data/apply_subtasks.py \
        --labels dataset/rlwrld_demo_0826_merge/meta/subtask_labels.jsonl \
        --dataset-dir dataset/rlwrld_demo_0826_train \
        --merge-meta dataset/rlwrld_demo_0826_merge/meta \
        --out-dir dataset/rlwrld_demo_0826_train_subtask
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--merge-meta", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    labels = {e["episode_index"]: e for e in read_jsonl(args.labels)}
    merge_src = read_jsonl(args.merge_meta / "source_episodes.jsonl")
    split_src = read_jsonl(args.dataset_dir / "meta" / "source_episodes.jsonl")
    by_origin = {(e["source_repo"], e["source_episode_index"]): e["episode_index"] for e in merge_src}

    # split episode index -> merge episode index, doubly checked
    split_to_merge: dict[int, int] = {}
    for e in split_src:
        merge_idx = by_origin[(e["source_repo"], e["source_episode_index"])]
        lab = labels[merge_idx]
        assert lab["length"] == e["length"], (
            f"length mismatch: split ep {e['episode_index']} ({e['length']}f) vs "
            f"label for merge ep {merge_idx} ({lab['length']}f)"
        )
        split_to_merge[e["episode_index"]] = merge_idx

    # subtask text -> new task index (0 stays the original whole-task sentence)
    old_tasks = read_jsonl(args.dataset_dir / "meta" / "tasks.jsonl")
    assert len(old_tasks) == 1, "expected the single whole-task sentence"
    text_to_idx: dict[str, int] = {}
    for lab in labels.values():
        for seg in lab["sub_tasks"]:
            if seg["text"] not in text_to_idx:
                text_to_idx[seg["text"]] = 1 + len(text_to_idx)

    assert not args.out_dir.exists(), f"{args.out_dir} already exists — refusing to overwrite"
    print(f"copying {args.dataset_dir} -> {args.out_dir}")
    shutil.copytree(args.dataset_dir, args.out_dir)

    meta = args.out_dir / "meta"
    write_jsonl(
        meta / "tasks.jsonl",
        [{"task_index": 0, "task": old_tasks[0]["task"]}]
        + [{"task_index": i, "task": t} for t, i in sorted(text_to_idx.items(), key=lambda kv: kv[1])],
    )
    info = json.loads((meta / "info.json").read_text())
    info["total_tasks"] = 1 + len(text_to_idx)
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    shutil.copyfile(args.labels, meta / "subtask_labels.jsonl")

    episodes = read_jsonl(meta / "episodes.jsonl")
    chunks_size = info["chunks_size"]

    for ep in episodes:
        idx = ep["episode_index"]
        lab = labels[split_to_merge[idx]]
        pq_path = args.out_dir / info["data_path"].format(
            episode_chunk=idx // chunks_size, episode_index=idx
        )
        table = pq.read_table(pq_path)
        n = table.num_rows
        assert n == lab["length"], f"ep {idx}: parquet {n}f vs label {lab['length']}f"

        frame_task = [-1] * n
        for seg in lab["sub_tasks"]:
            for f in range(seg["start"], seg["end"]):
                frame_task[f] = text_to_idx[seg["text"]]
        assert -1 not in frame_task, f"ep {idx}: uncovered frames"

        col = table.schema.get_field_index("task_index")
        old_field = table.schema.field(col)
        new_table = table.set_column(
            col, old_field, pa.array(frame_task, type=old_field.type)
        )

        # everything except task_index must be untouched
        assert new_table.schema.metadata == table.schema.metadata
        for name in table.schema.names:
            if name != "task_index":
                assert new_table.column(name).equals(table.column(name)), f"ep {idx}: {name} changed"

        pq.write_table(new_table, pq_path)

        # episodes.jsonl: list the subtask sentences in order of appearance
        seen: list[str] = []
        for seg in lab["sub_tasks"]:
            if seg["text"] not in seen:
                seen.append(seg["text"])
        ep["tasks"] = seen

    write_jsonl(meta / "episodes.jsonl", episodes)

    # re-read one parquet to confirm the write round-trips with schema intact
    sample_idx = episodes[0]["episode_index"]
    sample = pq.read_table(
        args.out_dir / info["data_path"].format(episode_chunk=sample_idx // chunks_size, episode_index=sample_idx)
    )
    orig = pq.read_table(
        args.dataset_dir / info["data_path"].format(episode_chunk=sample_idx // chunks_size, episode_index=sample_idx)
    )
    assert sample.schema.remove_metadata().equals(orig.schema.remove_metadata()), "schema drift after rewrite"
    assert sample.schema.metadata == orig.schema.metadata, "schema metadata lost after rewrite"

    counts: dict[int, int] = {}
    for ep in episodes:
        p = pq.read_table(
            args.out_dir / info["data_path"].format(episode_chunk=ep["episode_index"] // chunks_size, episode_index=ep["episode_index"]),
            columns=["task_index"],
        )
        for v in p.column("task_index").to_pylist():
            counts[v] = counts.get(v, 0) + 1
    print(f"episodes: {len(episodes)}, frame counts per task_index: {dict(sorted(counts.items()))}")
    print(f"tasks.jsonl: {1 + len(text_to_idx)} entries")
    print("OK")


if __name__ == "__main__":
    main()
