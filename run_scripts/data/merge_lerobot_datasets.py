#!/usr/bin/env python3
"""Concatenate several LeRobot v2.1 datasets into one, re-indexed end to end.

Merging is not a file copy: ``episode_index``, the global frame ``index`` and
``task_index`` are all dataset-scoped, so every parquet has to be rewritten and
every meta file regenerated. Video files, by contrast, are copied byte-for-byte
— re-encoding them would be a second generation of lossy compression for no
gain, since the decoder identifies the codec from the container and ignores the
``video.codec`` string in ``info.json``.

Sources must agree on robot_type, fps, and the state/action schema; the script
refuses to merge if they do not. Anything it has to reconcile rather than copy
(a codec that differs between sources, a meta file present in some sources only)
is printed as a warning, because those are exactly the details that go unnoticed
until training.

Usage:
    python run_scripts/data/merge_lerobot_datasets.py \
        --out dataset/merged dataset/a dataset/b dataset/c
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Bookkeeping columns whose values are dataset-scoped and therefore rewritten.
REINDEXED = ("episode_index", "index", "task_index")


def _reindex_parquet(src: Path, dst: Path, episode_index: int, first_frame: int, task_index: int) -> int:
    """Rewrite the three dataset-scoped columns, leaving everything else alone.

    Deliberately Arrow-level rather than pandas: a pandas round-trip downgrades
    ``fixed_size_list<float>[57]`` to a variable-length ``list<float>`` and drops
    the ``huggingface`` schema metadata that declares the feature as a
    ``Sequence`` of fixed length. Both survive a naive read/write unnoticed and
    then misreport the dataset to the Hub viewer and to LeRobot's feature
    decoding. ``set_column`` with the original field object keeps the declared
    type, and the table's schema metadata is carried over verbatim.
    """
    table = pq.read_table(src)
    n = table.num_rows
    new_values = {
        "episode_index": np.full(n, episode_index, dtype=np.int64),
        "index": np.arange(first_frame, first_frame + n, dtype=np.int64),
        "task_index": np.full(n, task_index, dtype=np.int64),
    }
    for name, values in new_values.items():
        i = table.schema.get_field_index(name)
        field = table.schema.field(i)
        table = table.set_column(i, field, pa.array(values, type=field.type))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression="snappy")
    return n


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _stats_of(arr: np.ndarray) -> dict:
    """Match LeRobot's episodes_stats layout for a 1-D bookkeeping column."""
    a = arr.astype(np.float64)
    return {
        "min": [float(a.min())], "max": [float(a.max())],
        "mean": [float(a.mean())], "std": [float(a.std())],
        "count": [int(a.size)],
    }


def merge(sources: list[Path], out: Path, copy_videos: bool = True) -> dict:
    infos = [json.loads((s / "meta" / "info.json").read_text()) for s in sources]
    warnings: list[str] = []

    # ---- compatibility gate ----------------------------------------------
    def _must_agree(getter, label):
        vals = {json.dumps(getter(i), sort_keys=True) for i in infos}
        if len(vals) > 1:
            detail = "\n".join(f"    {s.name}: {json.dumps(getter(i), sort_keys=True)}"
                               for s, i in zip(sources, infos))
            raise SystemExit(f"ERROR: sources disagree on {label}; refusing to merge:\n{detail}")

    _must_agree(lambda i: i["codebase_version"], "codebase_version")
    _must_agree(lambda i: i["robot_type"], "robot_type")
    _must_agree(lambda i: i["fps"], "fps")
    _must_agree(lambda i: sorted(i["features"].keys()), "the feature set")
    for feat in ("observation.state", "action"):
        _must_agree(lambda i, f=feat: i["features"][f]["shape"], f"the shape of {feat}")
        _must_agree(lambda i, f=feat: i["features"][f].get("names"), f"the joint names of {feat}")

    base = infos[0]
    video_keys = [k for k, v in base["features"].items() if v["dtype"] == "video"]
    for vk in video_keys:
        for field in ("video.height", "video.width", "video.fps", "video.pix_fmt"):
            vals = {i["features"][vk]["info"][field] for i in infos}
            if len(vals) > 1:
                raise SystemExit(f"ERROR: sources disagree on {vk} {field}: {vals}")
        codecs = collections.Counter(i["features"][vk]["info"]["video.codec"] for i in infos)
        if len(codecs) > 1:
            per_src = ", ".join(f"{s.name}={i['features'][vk]['info']['video.codec']}"
                                for s, i in zip(sources, infos))
            # The declared codec is metadata only — torchcodec reads the real one
            # from the container — so the merge keeps the files untouched and
            # declares the majority. Worth knowing about, not worth re-encoding.
            warnings.append(
                f"{vk}: sources use different codecs ({per_src}). Videos are copied unchanged; "
                f"info.json will declare the majority codec {codecs.most_common(1)[0][0]!r}, "
                f"so that field will be wrong for the minority episodes.")

    # ---- global task table ------------------------------------------------
    task_to_new: dict[str, int] = {}
    for s in sources:
        for t in _read_jsonl(s / "meta" / "tasks.jsonl"):
            task_to_new.setdefault(t["task"], len(task_to_new))

    # ---- lay out the output ----------------------------------------------
    if out.exists():
        raise SystemExit(f"ERROR: {out} already exists — remove it or pick another --out")
    (out / "meta").mkdir(parents=True)
    chunks_size = base["chunks_size"]

    episodes_out: list[dict] = []
    ep_stats_out: list[dict] = []
    provenance: list[dict] = []
    extra_out: dict[str, list[dict]] = collections.defaultdict(list)

    new_ei = 0
    frame_cursor = 0
    for s, info in zip(sources, infos):
        eps = _read_jsonl(s / "meta" / "episodes.jsonl")
        src_stats = {r["episode_index"]: r for r in _read_jsonl(s / "meta" / "episodes_stats.jsonl")}
        # Non-standard sidecars (e.g. the recorder's rilab_episodes.jsonl) are
        # carried through when every source has them; dropped with a warning
        # otherwise, since a partial sidecar is worse than none.
        extras = {p.name: {r["episode_index"]: r for r in _read_jsonl(p)}
                  for p in (s / "meta").glob("*.jsonl")
                  if p.name not in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")}

        for ep in eps:
            old_ei = ep["episode_index"]
            old_chunk, new_chunk = old_ei // chunks_size, new_ei // chunks_size

            src_pq = s / info["data_path"].format(episode_chunk=old_chunk, episode_index=old_ei)
            dst_pq = out / base["data_path"].format(episode_chunk=new_chunk, episode_index=new_ei)

            new_task = task_to_new[ep["tasks"][0]]
            n = _reindex_parquet(src_pq, dst_pq, new_ei, frame_cursor, new_task)

            for vk in video_keys:
                src_v = s / info["video_path"].format(episode_chunk=old_chunk, video_key=vk, episode_index=old_ei)
                dst_v = out / base["video_path"].format(episode_chunk=new_chunk, video_key=vk, episode_index=new_ei)
                dst_v.parent.mkdir(parents=True, exist_ok=True)
                if copy_videos:
                    shutil.copy2(src_v, dst_v)
                else:  # same filesystem — a hardlink is instant and shares no risk of drift
                    dst_v.hardlink_to(src_v)

            episodes_out.append({"episode_index": new_ei, "tasks": ep["tasks"], "length": n})

            st = src_stats.get(old_ei)
            if st is None:
                warnings.append(f"{s.name} ep {old_ei}: no entry in episodes_stats.jsonl")
            else:
                stats = dict(st["stats"])
                # These three describe the columns the merge just rewrote.
                stats["episode_index"] = _stats_of(np.full(n, new_ei, dtype=np.int64))
                stats["index"] = _stats_of(np.arange(frame_cursor, frame_cursor + n, dtype=np.int64))
                stats["task_index"] = _stats_of(np.full(n, new_task, dtype=np.int64))
                ep_stats_out.append({"episode_index": new_ei, "stats": stats})

            for name, table in extras.items():
                row = table.get(old_ei)
                if row is not None:
                    extra_out[name].append({**row, "episode_index": new_ei})

            provenance.append({"episode_index": new_ei, "source_repo": s.name,
                               "source_episode_index": old_ei, "length": n})
            new_ei += 1
            frame_cursor += n

    # ---- meta -------------------------------------------------------------
    _write_jsonl(out / "meta" / "episodes.jsonl", episodes_out)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", ep_stats_out)
    _write_jsonl(out / "meta" / "tasks.jsonl",
                 [{"task_index": v, "task": k} for k, v in sorted(task_to_new.items(), key=lambda kv: kv[1])])
    _write_jsonl(out / "meta" / "source_episodes.jsonl", provenance)

    for name, rows in extra_out.items():
        if len(rows) != len(episodes_out):
            missing = [s.name for s in sources if not (s / "meta" / name).exists()]
            warnings.append(
                f"{name}: only {len(rows)}/{len(episodes_out)} episodes have an entry "
                f"(absent from {', '.join(missing) or 'some sources'}) — dropped from the merge "
                f"rather than shipping a partial sidecar")
            continue
        _write_jsonl(out / "meta" / name, rows)

    merged_info = json.loads(json.dumps(base))
    merged_info["total_episodes"] = len(episodes_out)
    merged_info["total_frames"] = frame_cursor
    merged_info["total_tasks"] = len(task_to_new)
    merged_info["total_videos"] = len(episodes_out) * len(video_keys)
    merged_info["total_chunks"] = (len(episodes_out) - 1) // chunks_size + 1
    merged_info["splits"] = {"train": f"0:{len(episodes_out)}"}
    for vk in video_keys:
        codecs = collections.Counter(i["features"][vk]["info"]["video.codec"] for i in infos)
        merged_info["features"][vk]["info"]["video.codec"] = codecs.most_common(1)[0][0]
    (out / "meta" / "info.json").write_text(json.dumps(merged_info, indent=4))

    # Copy modality.json from the first source that has one — none of the
    # recorder's outputs ship it, so usually there is nothing to copy.
    for s in sources:
        mj = s / "meta" / "modality.json"
        if mj.exists():
            shutil.copy2(mj, out / "meta" / "modality.json")
            break
    else:
        warnings.append("no source ships meta/modality.json — RLDX-1 requires it before training "
                        "(see CLAUDE.md 'Onboarding a new FFW-SH5 dataset', step 2)")

    return {"out": str(out), "episodes": len(episodes_out), "frames": frame_cursor,
            "tasks": len(task_to_new), "warnings": warnings}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hardlink-videos", action="store_true",
                    help="hardlink instead of copying videos (same filesystem only)")
    args = ap.parse_args()

    res = merge(args.sources, args.out, copy_videos=not args.hardlink_videos)
    print(f"\nmerged {len(args.sources)} datasets -> {res['out']}")
    print(f"  {res['episodes']} episodes, {res['frames']} frames, {res['tasks']} task(s)")
    if res["warnings"]:
        print("\nwarnings:")
        for w in res["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
