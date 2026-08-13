#!/usr/bin/env python
"""``rldx/eval/open_loop_eval.py`` with its results actually printed.

``open_loop_eval.main()`` reports MSE/MAE through ``logging.info``, and calls
``logging.basicConfig(level=INFO)`` to enable it — but transformers has already
configured the root logger by the time that runs, which makes ``basicConfig`` a
no-op. The level stays at WARNING and every result line is swallowed, so the
eval appears to do nothing but write a plot. ``force=True`` overrides the
existing handler configuration.

Also prints peak VRAM, which is the number you want when sizing a deployment
GPU (see RLDX-tuning.md section 8).

Usage mirrors the upstream script:

    pixi run -e rldx-sm100 python run_scripts/eval/open_loop_eval_verbose.py \
        --model-path ckpt/rldx1/finetuned/<name>/<name>/checkpoint-<step> \
        --dataset-path dataset/ffw_sh5_0812 \
        --embodiment-tag GENERAL_EMBODIMENT \
        --traj-ids 1 2 --action-horizon 16 --steps 64 \
        --save-plot-path outputs/eval/openloop.png

Do NOT inline this as ``python -c "..." --model-path ...`` — the flags do not
reach tyro through ``pixi run``.

Note on --traj-ids: pick a trajectory that actually moves. For ffw_sh5_0812,
episode 0 is a dead recording (all 57 dims constant across its 49 frames), so
its ground truth plots as a flat line and the model looks broken.
"""

import logging
from pathlib import Path

import torch
import tyro

logging.basicConfig(level=logging.INFO, force=True)

from rldx.eval.open_loop_eval import ArgsConfig, main  # noqa: E402


if __name__ == "__main__":
    args = tyro.cli(ArgsConfig)

    if args.save_plot_path:
        Path(args.save_plot_path).parent.mkdir(parents=True, exist_ok=True)

    main(args)

    if torch.cuda.is_available():
        print(f"[VRAM] peak allocated : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        print(f"[VRAM] peak reserved  : {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
