#!/usr/bin/env python3
"""Decode every episode video and look for the failures ffprobe cannot see.

A frame count can be correct while the pixels are useless: the camera dropped
out and every frame is identical, the exposure collapsed to black, or the
stream decodes to garbage. This samples frames through torchcodec — the same
decoder the training data loader uses — so a video that fails here is a video
that will fail in training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

# Mean absolute inter-frame difference (0-255 scale) below which the camera is
# considered frozen. Real handheld/head-cam footage of a manipulation task sits
# well above 1.0; a stalled stream sits at exactly 0.
FROZEN_DIFF = 0.5
# Mean pixel value below which a frame is effectively black.
BLACK_LEVEL = 8.0


def check(root: Path, samples: int = 12) -> list[dict]:
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = [json.loads(l) for l in (root / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    out = []

    for ep in episodes:
        ei = ep["episode_index"]
        chunk = ei // info["chunks_size"]
        for vk in video_keys:
            mp4 = root / info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=ei)
            rec = {"episode": ei, "key": vk, "error": None}
            try:
                dec = VideoDecoder(str(mp4))
                n = len(dec)
                rec["frames"] = n
                idx = np.unique(np.linspace(0, n - 1, min(samples, n)).astype(int))
                frames = dec.get_frames_at(indices=idx.tolist()).data.float()
                means = frames.mean(dim=(1, 2, 3))
                diffs = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
                rec["mean_pixel"] = float(means.mean())
                rec["min_frame_mean"] = float(means.min())
                rec["mean_interframe_diff"] = float(diffs.mean()) if len(diffs) else 0.0
                rec["min_interframe_diff"] = float(diffs.min()) if len(diffs) else 0.0
                rec["frozen"] = rec["mean_interframe_diff"] < FROZEN_DIFF
                rec["black_frames"] = int((means < BLACK_LEVEL).sum())
            except Exception as e:  # decoder failures are the finding, not a crash
                rec["error"] = f"{type(e).__name__}: {e}"
            out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    print(f"torch {torch.__version__}\n")
    allrec = {}
    for root in args.roots:
        recs = check(root, args.samples)
        allrec[str(root)] = recs
        errs = [r for r in recs if r["error"]]
        frozen = [r for r in recs if not r["error"] and r["frozen"]]
        black = [r for r in recs if not r["error"] and r["black_frames"]]
        ok = [r for r in recs if not r["error"]]
        print(f"{root.name}: {len(recs)} videos  "
              f"decode-fail={len(errs)}  frozen={len(frozen)}  with-black-frames={len(black)}")
        if ok:
            d = np.array([r["mean_interframe_diff"] for r in ok])
            m = np.array([r["mean_pixel"] for r in ok])
            print(f"   interframe diff: min={d.min():.2f} med={np.median(d):.2f} max={d.max():.2f}")
            print(f"   mean pixel     : min={m.min():.1f} med={np.median(m):.1f} max={m.max():.1f}")
        for r in errs:
            print(f"   [high] ep {r['episode']:>3} {r['key']}: DECODE FAILED — {r['error']}")
        for r in frozen:
            print(f"   [high] ep {r['episode']:>3} {r['key']}: frozen video "
                  f"(interframe diff {r['mean_interframe_diff']:.3f})")
        for r in black:
            print(f"   [med ] ep {r['episode']:>3} {r['key']}: {r['black_frames']} sampled frames are black "
                  f"(min frame mean {r['min_frame_mean']:.1f})")
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(allrec, indent=2))
        print(f"report written to {args.json}")


if __name__ == "__main__":
    main()
