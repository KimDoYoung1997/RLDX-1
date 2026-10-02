#!/usr/bin/env python3
"""Split a LeRobot v2.1 dataset into a train part and a held-out part.

The trainer has no validation path — ``DatasetFactory.build`` asserts
``eval_strategy == "no"`` and always returns ``eval_dataset=None``. So the only
way to tell "still learning" from "now memorising" is to keep some episodes out
of training entirely and score checkpoints against them with
``open_loop_eval``. That requires two separate dataset directories, which is
what this produces.

Both outputs are re-indexed to be valid standalone datasets: episode indices
become contiguous from 0, the global frame ``index`` restarts, and every meta
file is regenerated. Videos are copied byte-for-byte and parquet payload
columns are preserved exactly (Arrow-level rewrite, same as the merge script).

Usage:
    python run_scripts/data/split_holdout.py \
        --src dataset/rlwrld_demo_0826_merge \
        --holdout 3,10,17,26,35,44,56,65,74 \
        --train-out dataset/rlwrld_demo_0826_train \
        --holdout-out dataset/rlwrld_demo_0826_holdout
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_lerobot_datasets import _reindex_parquet, _read_jsonl, _stats_of, _write_jsonl


def _write_part(src: Path, out: Path, episodes: list[int], info: dict) -> dict:
    """Write the subset ``episodes`` of ``src`` as a standalone dataset at ``out``."""
    if out.exists():
        raise SystemExit(f"ERROR: {out} already exists — remove it or pick another path")
    (out / "meta").mkdir(parents=True)

    chunks_size = info["chunks_size"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    src_eps = {r["episode_index"]: r for r in _read_jsonl(src / "meta" / "episodes.jsonl")}
    src_stats = {r["episode_index"]: r for r in _read_jsonl(src / "meta" / "episodes_stats.jsonl")}
    tasks = _read_jsonl(src / "meta" / "tasks.jsonl")
    task_to_idx = {t["task"]: t["task_index"] for t in tasks}
    # Sidecars keyed by episode_index (provenance, recorder logs, ...) follow the split.
    extras = {
        p.name: {r["episode_index"]: r for r in _read_jsonl(p)}
        for p in (src / "meta").glob("*.jsonl")
        if p.name not in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl")
    }

    eps_out, stats_out, frame_cursor = [], [], 0
    extra_out: dict[str, list[dict]] = {name: [] for name in extras}

    for new_ei, old_ei in enumerate(episodes):
        ep = src_eps[old_ei]
        old_chunk, new_chunk = old_ei // chunks_size, new_ei // chunks_size
        task_index = task_to_idx[ep["tasks"][0]]

        n = _reindex_parquet(
            src / info["data_path"].format(episode_chunk=old_chunk, episode_index=old_ei),
            out / info["data_path"].format(episode_chunk=new_chunk, episode_index=new_ei),
            new_ei, frame_cursor, task_index,
        )
        for vk in video_keys:
            s = src / info["video_path"].format(episode_chunk=old_chunk, video_key=vk, episode_index=old_ei)
            d = out / info["video_path"].format(episode_chunk=new_chunk, video_key=vk, episode_index=new_ei)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

        eps_out.append({**ep, "episode_index": new_ei, "length": n})
        st = dict(src_stats[old_ei]["stats"])
        st["episode_index"] = _stats_of(np.full(n, new_ei, dtype=np.int64))
        st["index"] = _stats_of(np.arange(frame_cursor, frame_cursor + n, dtype=np.int64))
        st["task_index"] = _stats_of(np.full(n, task_index, dtype=np.int64))
        stats_out.append({"episode_index": new_ei, "stats": st})
        for name, table in extras.items():
            if old_ei in table:
                extra_out[name].append({**table[old_ei], "episode_index": new_ei})
        frame_cursor += n

    _write_jsonl(out / "meta" / "episodes.jsonl", eps_out)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", stats_out)
    _write_jsonl(out / "meta" / "tasks.jsonl", tasks)
    for name, rows in extra_out.items():
        if rows:
            _write_jsonl(out / "meta" / name, rows)

    new_info = json.loads(json.dumps(info))
    new_info["total_episodes"] = len(eps_out)
    new_info["total_frames"] = frame_cursor
    new_info["total_videos"] = len(eps_out) * len(video_keys)
    new_info["total_chunks"] = (len(eps_out) - 1) // chunks_size + 1
    new_info["splits"] = {"train": f"0:{len(eps_out)}"}
    (out / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

    # modality.json is required by the loader and is split-independent, so copy it.
    #
    # stats.json is deliberately NOT copied. It holds the q01/q99 values the
    # policy normalises with, and copying the source's would mean the training
    # split's normalisation was fitted on the held-out episodes too — mild, but
    # it is leakage and there is no reason to accept it. Leaving the file absent
    # makes DatasetFactory._ensure_stats regenerate it per split on the first
    # run. (Note that `generate_stats` prints "Generating stats for ..." *before*
    # checking whether a valid file already exists, so that log line is not
    # evidence that anything was recomputed — see `stats.py:118` vs `:127`.)
    shutil.copy2(src / "meta" / "modality.json", out / "meta" / "modality.json")

    return {"episodes": len(eps_out), "frames": frame_cursor}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--holdout", required=True, help="comma-separated source episode indices")
    ap.add_argument("--train-out", required=True, type=Path)
    ap.add_argument("--holdout-out", required=True, type=Path)
    args = ap.parse_args()

    info = json.loads((args.src / "meta" / "info.json").read_text())
    total = info["total_episodes"]
    hold = sorted({int(x) for x in args.holdout.split(",")})
    bad = [e for e in hold if not 0 <= e < total]
    if bad:
        raise SystemExit(f"ERROR: episode indices out of range 0..{total - 1}: {bad}")
    train = [e for e in range(total) if e not in set(hold)]

    t = _write_part(args.src, args.train_out, train, info)
    h = _write_part(args.src, args.holdout_out, hold, info)

    print(f"train   -> {args.train_out}  {t['episodes']} ep / {t['frames']} frames")
    print(f"holdout -> {args.holdout_out}  {h['episodes']} ep / {h['frames']} frames "
          f"({100 * h['frames'] / (t['frames'] + h['frames']):.1f}%)")
    print(f"\nheld-out source episodes: {hold}")
    print("NOTE: holdout episode indices are renumbered 0..N-1 in the new dataset — "
          "use --traj-ids 0 1 2 ... with open_loop_eval, not the source indices.")


if __name__ == "__main__":
    main()
