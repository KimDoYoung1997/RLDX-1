#!/usr/bin/env python3
"""Audit LeRobot v2.1 datasets for the recording defects that survive a merge.

A merge is a mechanical re-index: it will happily renumber a dead episode, a
parquet whose row count disagrees with its mp4, or a run recorded at the wrong
fps. Those only surface much later — as a flat line in an open-loop eval, or as
a silent frame/state misalignment during training. So audit the *sources*
before merging, and audit the merged result again to confirm nothing shifted.

Usage:
    python run_scripts/data/audit_lerobot_dataset.py dataset/a dataset/b ...
    python run_scripts/data/audit_lerobot_dataset.py --json out.json dataset/a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# A joint that never moves more than this (radians) across a whole episode is
# treated as frozen. Teleop noise on a genuinely still joint is ~1e-3, so 1e-4
# flags "the value was written once and never updated", not "the arm was calm".
FROZEN_EPS = 1e-4
# Fraction of an episode's *used* joints (arms + hands, dims 0:54) that must be
# frozen before the episode itself is called dead.
DEAD_FRACTION = 0.98
# Episodes shorter than this many frames cannot cover an action horizon of 16
# with any margin; they are almost always aborted recordings.
SHORT_EPISODE = 40


def _ffprobe_frames(path: Path) -> int | None:
    """Exact frame count of an mp4. Returns None if ffprobe is unavailable."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    txt = out.stdout.strip()
    return int(txt) if txt.isdigit() else None


def audit(root: Path, check_videos: bool = True) -> dict:
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    episodes = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l.strip()]
    tasks = [json.loads(l) for l in (meta / "tasks.jsonl").read_text().splitlines() if l.strip()]

    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    findings: list[dict] = []
    per_ep: list[dict] = []

    def flag(sev: str, kind: str, msg: str, ep: int | None = None) -> None:
        findings.append({"severity": sev, "kind": kind, "episode": ep, "message": msg})

    # ---- metadata-level consistency ---------------------------------------
    declared = sum(e["length"] for e in episodes)
    if declared != info["total_frames"]:
        flag("high", "meta-mismatch",
             f"info.total_frames={info['total_frames']} but episodes.jsonl sums to {declared}")
    if len(episodes) != info["total_episodes"]:
        flag("high", "meta-mismatch",
             f"info.total_episodes={info['total_episodes']} but episodes.jsonl has {len(episodes)}")
    seen_idx = [e["episode_index"] for e in episodes]
    if seen_idx != list(range(len(episodes))):
        flag("high", "meta-mismatch", "episode_index in episodes.jsonl is not contiguous from 0")

    task_strings = {t["task"] for t in tasks}
    for t in tasks:
        # A task that is bare digits is a recording ID, not an instruction — the
        # VLA would receive the literal string as its language conditioning.
        if t["task"].strip().isdigit():
            flag("high", "placeholder-task",
                 f"task_index={t['task_index']} is the placeholder {t['task']!r}, not an instruction")

    # ---- per-episode data -------------------------------------------------
    global_index_cursor = 0
    for ep in episodes:
        ei = ep["episode_index"]
        chunk = ei // info["chunks_size"]
        pq = root / info["data_path"].format(episode_chunk=chunk, episode_index=ei)
        if not pq.exists():
            flag("high", "missing-file", f"parquet missing: {pq.relative_to(root)}", ei)
            continue
        df = pd.read_parquet(pq)

        n = len(df)
        if n != ep["length"]:
            flag("high", "length-mismatch",
                 f"episodes.jsonl says length={ep['length']} but parquet has {n} rows", ei)
        if n < SHORT_EPISODE:
            flag("high", "short-episode", f"only {n} frames ({n / fps:.1f}s) — likely an aborted take", ei)

        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())

        for name, arr in (("observation.state", state), ("action", action)):
            bad = ~np.isfinite(arr)
            if bad.any():
                flag("high", "non-finite",
                     f"{name} has {int(bad.sum())} NaN/Inf values in {int(bad.any(0).sum())} dims", ei)

        # Frozen joints among the dims this robot actually trains on (0:54 =
        # both arms + both hands; 54:57 are head/lift and are unused).
        used = state[:, :54]
        span = used.max(0) - used.min(0)
        frozen = int((span < FROZEN_EPS).sum())
        dead = frozen >= DEAD_FRACTION * 54
        if dead:
            flag("high", "dead-episode",
                 f"{frozen}/54 trained joints never move — recording is flat, not a demonstration", ei)
        elif frozen > 27:
            flag("medium", "mostly-frozen",
                 f"{frozen}/54 trained joints never move across the episode", ei)

        # Timestamps: LeRobot writes t = frame_index / fps, so any deviation
        # means the run was recorded or rewritten at a different rate.
        ts = df["timestamp"].to_numpy(dtype=np.float64)
        if n > 1:
            d = np.diff(ts)
            if (d <= 0).any():
                flag("high", "timestamp", f"timestamp is not strictly increasing ({int((d <= 0).sum())} steps)", ei)
            else:
                med = float(np.median(d))
                if abs(med - 1.0 / fps) > 0.1 / fps:
                    flag("high", "fps-mismatch",
                         f"median timestamp step {med:.4f}s implies {1 / med:.2f} fps, info.json says {fps}", ei)
                jumps = int((d > 3.0 / fps).sum())
                if jumps:
                    flag("medium", "dropped-frames",
                         f"{jumps} gaps larger than 3 frame intervals (max {d.max():.2f}s) — dropped frames", ei)

        # Bookkeeping columns must agree with the episode they live in.
        if not (df["episode_index"].to_numpy() == ei).all():
            flag("high", "index", "episode_index column disagrees with the file it is stored in", ei)
        if not (df["frame_index"].to_numpy() == np.arange(n)).all():
            flag("high", "index", "frame_index column is not 0..n-1", ei)
        gi = df["index"].to_numpy()
        if not (gi == np.arange(global_index_cursor, global_index_cursor + n)).all():
            flag("high", "index",
                 f"global `index` column is not continuous with the previous episode "
                 f"(expected to start at {global_index_cursor}, starts at {int(gi[0])})", ei)
        ti = df["task_index"].to_numpy()
        if len(set(ti.tolist())) != 1:
            flag("medium", "task", "task_index varies within the episode", ei)
        elif ep["tasks"] and tasks:
            want = next((t["task_index"] for t in tasks if t["task"] == ep["tasks"][0]), None)
            if want is not None and int(ti[0]) != want:
                flag("high", "task",
                     f"task_index={int(ti[0])} in parquet but episodes.jsonl names task_index={want}", ei)
        global_index_cursor += n

        # Videos: a frame-count disagreement means image and state are
        # misaligned for the whole episode.
        vid_frames = {}
        for vk in video_keys:
            mp4 = root / info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=ei)
            if not mp4.exists():
                flag("high", "missing-file", f"video missing: {mp4.relative_to(root)}", ei)
                continue
            if mp4.stat().st_size == 0:
                flag("high", "empty-video", f"video is 0 bytes: {mp4.relative_to(root)}", ei)
                continue
            if check_videos:
                got = _ffprobe_frames(mp4)
                vid_frames[vk] = got
                if got is not None and got != n:
                    flag("high", "video-desync",
                         f"{vk} has {got} frames but parquet has {n} rows (off by {got - n})", ei)

        # Tactile, when present, is often wired up but never populated.
        tac_zero = None
        if "observation.tactile" in df.columns:
            tac = np.stack(df["observation.tactile"].to_numpy())
            tac_zero = bool(np.all(tac == 0))

        per_ep.append({
            "episode_index": ei, "frames": n, "frozen_joints": frozen, "dead": dead,
            "tactile_all_zero": tac_zero,
            "state_hash": hashlib.md5(state.tobytes()).hexdigest(),
            "state_min": float(used.min()), "state_max": float(used.max()),
            "action_state_rmse": float(np.sqrt(((action[:, :54] - used) ** 2).mean())),
            "video_frames": vid_frames,
        })

    if per_ep and all(e["tactile_all_zero"] for e in per_ep if e["tactile_all_zero"] is not None):
        flag("low", "tactile-empty",
             "observation.tactile is all zeros in every episode — the sensor stream was not recorded")

    # Duplicate episodes inside this dataset (identical state trajectories).
    by_hash: dict[str, list[int]] = {}
    for e in per_ep:
        by_hash.setdefault(e["state_hash"], []).append(e["episode_index"])
    for h, eps in by_hash.items():
        if len(eps) > 1:
            flag("high", "duplicate", f"episodes {eps} have byte-identical observation.state")

    return {
        "root": str(root), "info": {k: info[k] for k in
                                    ("codebase_version", "robot_type", "fps", "total_episodes", "total_frames")},
        "video_keys": video_keys,
        "codecs": {k: info["features"][k]["info"].get("video.codec") for k in video_keys},
        "shapes": {k: info["features"][k]["shape"] for k in
                   ("observation.state", "action") if k in info["features"]},
        "tasks": sorted(task_strings),
        "episodes": per_ep, "findings": findings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--no-videos", action="store_true", help="skip ffprobe frame counting (much faster)")
    args = ap.parse_args()

    reports = []
    for root in args.roots:
        print(f"\n{'=' * 78}\n{root}\n{'=' * 78}")
        rep = audit(root, check_videos=not args.no_videos)
        reports.append(rep)
        print(f"  {rep['info']['total_episodes']} episodes / {rep['info']['total_frames']} frames "
              f"@ {rep['info']['fps']} fps   codecs={rep['codecs']}")
        if not rep["findings"]:
            print("  no findings")
        for sev in ("high", "medium", "low"):
            for f in [x for x in rep["findings"] if x["severity"] == sev]:
                loc = f"ep {f['episode']:>3}" if f["episode"] is not None else "dataset "
                print(f"  [{sev:<6}] {loc}  {f['kind']}: {f['message']}")

    # Cross-dataset checks only make sense with more than one source.
    if len(reports) > 1:
        print(f"\n{'=' * 78}\ncross-dataset\n{'=' * 78}")
        seen: dict[str, tuple[str, int]] = {}
        for rep in reports:
            for e in rep["episodes"]:
                key = e["state_hash"]
                if key in seen:
                    prev = seen[key]
                    print(f"  [high  ] duplicate: {Path(rep['root']).name} ep {e['episode_index']} is "
                          f"byte-identical to {prev[0]} ep {prev[1]}")
                else:
                    seen[key] = (Path(rep["root"]).name, e["episode_index"])
        for field, label in (("codecs", "video codec"), ("shapes", "state/action shape"),
                             ("tasks", "task strings")):
            vals = {json.dumps(r[field], sort_keys=True) for r in reports}
            if len(vals) > 1:
                print(f"  [high  ] {label} differs across sources:")
                for r in reports:
                    print(f"             {Path(r['root']).name}: {json.dumps(r[field], sort_keys=True)}")
        fpss = {r["info"]["fps"] for r in reports}
        if len(fpss) > 1:
            print(f"  [high  ] fps differs across sources: {fpss}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"\nreport written to {args.json}")


if __name__ == "__main__":
    main()
