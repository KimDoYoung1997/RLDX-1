#!/usr/bin/env python3
"""Dump cam_head frames as contact sheets, at raw and at training resolution.

This script does **no** recognition. It decodes frames and tiles them into a
PNG; deciding what is in them (where the shipping label is, whether the box was
reoriented) is left to whoever looks at the sheet — a person or a VLM you feed
it to. There is no detector and no model call anywhere in here.

Two sheets are written per episode:

  ``ep0NN_raw.png``    the mp4 as recorded (376x672 for cam_head)
  ``ep0NN_train.png``  after the training-time resize + centre crop, i.e. what
                       the policy actually sees. Compare the two to check that
                       a detail you care about survives the downscale.

Sampling defaults to *front-loaded* fractions rather than a uniform grid,
because for this dataset the branch-deciding information (how the box was
placed) is near the start of the episode. Use ``--every-sec`` when you want a
plain uniform timeline instead.

Usage:
    # front-loaded default, 8 frames
    python run_scripts/data/peek_frames.py outputs/frames 5,30,70

    # uniform, one frame every 2 seconds
    python run_scripts/data/peek_frames.py outputs/frames 30 --every-sec 2

    # match a different training image budget
    python run_scripts/data/peek_frames.py outputs/frames 30 --max-area 65536
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torchcodec.decoders import VideoDecoder

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rldx.data.augmentations import resize_preserve_aspect_area_then_crop as _resize_crop

# Fractions of the episode to sample by default. Deliberately dense early and
# sparse late: the initial object pose is what decides the task branch, while
# the tail only needs one frame to confirm the end state.
DEFAULT_FRACTIONS = (0.0, 1 / 12, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 1.0)

# Anything wider than this is painful to open and exceeds the size limit of
# most image viewers/VLM upload paths, so sheets get scaled down at the end.
MAX_SHEET_WIDTH = 1900


def train_view(frame: np.ndarray, max_area: int, m: int = 32) -> Image.Image:
    """Reproduce the training-time resize + centre crop for one frame."""
    h, w = frame.shape[:2]
    (h_r, w_r), (h_c, w_c) = _resize_crop(h, w, max_area=max_area, m=m)
    img = Image.fromarray(frame).resize((w_r, h_r), Image.BILINEAR)
    left, top = (w_r - w_c) // 2, (h_r - h_c) // 2
    return img.crop((left, top, left + w_c, top + h_c))


def contact_sheet(images: list[Image.Image], captions: list[str], cols: int = 4) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    band = 16  # caption strip under each tile
    sheet = Image.new("RGB", (w * cols, (h + band) * rows), "black")
    draw = ImageDraw.Draw(sheet)
    for k, (im, cap) in enumerate(zip(images, captions)):
        x, y = (k % cols) * w, (k // cols) * (h + band)
        sheet.paste(im, (x, y))
        draw.text((x + 3, y + h + 3), cap, fill="white")
    if sheet.width > MAX_SHEET_WIDTH:
        s = MAX_SHEET_WIDTH / sheet.width
        sheet = sheet.resize((MAX_SHEET_WIDTH, int(sheet.height * s)), Image.LANCZOS)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("episodes", help="comma-separated merged episode indices, e.g. 5,30,70")
    ap.add_argument("--dataset", type=Path, default=Path("dataset/rlwrld_demo_0826_merge"))
    ap.add_argument("--video-key", default="observation.images.cam_head")
    ap.add_argument("--max-area", type=int, default=98304,
                    help="training image_max_area to reproduce (default 98304 = 224x384)")
    ap.add_argument("--every-sec", type=float, default=None,
                    help="uniform sampling every N seconds instead of the front-loaded default")
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    fps = info["fps"]
    chunks_size = info["chunks_size"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ei in [int(x) for x in args.episodes.split(",")]:
        chunk = ei // chunks_size
        mp4 = args.dataset / info["video_path"].format(
            episode_chunk=chunk, video_key=args.video_key, episode_index=ei
        )
        dec = VideoDecoder(str(mp4))
        n = len(dec)

        if args.every_sec:
            step = max(1, int(round(args.every_sec * fps)))
            idx = list(range(0, n, step))
        else:
            idx = sorted({min(n - 1, int(f * (n - 1))) for f in DEFAULT_FRACTIONS})

        frames = dec.get_frames_at(indices=idx).data.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        caps = [f"#{i}  {i / fps:.1f}s" for i in idx]

        raw = [Image.fromarray(f) for f in frames]
        trn = [train_view(f, args.max_area) for f in frames]
        contact_sheet(raw, caps, args.cols).save(args.out_dir / f"ep{ei:03d}_raw.png")
        contact_sheet(trn, caps, args.cols).save(args.out_dir / f"ep{ei:03d}_train.png")

        gaps = np.diff([i / fps for i in idx])
        print(f"ep {ei:>3}: {n} frames = {n / fps:.1f}s @ {fps}fps | {len(idx)} samples")
        print(f"        times(s): {', '.join(f'{i / fps:.1f}' for i in idx)}")
        print(f"        gaps(s) : {', '.join(f'{g:.1f}' for g in gaps)}")
        print(f"        raw {raw[0].size}  train {trn[0].size}  -> {args.out_dir}/ep{ei:03d}_*.png")


if __name__ == "__main__":
    main()
