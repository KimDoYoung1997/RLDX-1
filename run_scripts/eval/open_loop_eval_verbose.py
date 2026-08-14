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

import gc
import logging
import sys
from pathlib import Path

import torch
import tyro

logging.basicConfig(level=logging.INFO, force=True)


def _check_gpu_arch() -> None:
    """Fail fast when this torch build has no kernels for the installed GPU.

    Running from the uv/.venv interpreter on a B200 gets you a cu126 torch whose
    CUBINs stop at sm_90. torch only prints a UserWarning at import, then dies
    ~200 lines later inside the backbone with a bare "no kernel image is
    available for execution on the device" that points at an innocent line. Cheap
    to detect up front, so do that instead.
    """
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if arch in supported:
        return
    raise SystemExit(
        f"\nERROR: {torch.cuda.get_device_name(0)} is {arch}, but this torch "
        f"({torch.__version__}) only ships {', '.join(supported)}.\n"
        f"       Every CUDA launch will fail with 'no kernel image is available'.\n\n"
        f"       You are probably running the uv/.venv python directly. Use the\n"
        f"       pixi environment built for this GPU instead:\n\n"
        f"           pixi run -e rldx-sm100 python {' '.join(sys.argv)}\n\n"
        f"       See RLDX-tuning.md section 1.\n"
    )


from rldx.eval.open_loop_eval import ArgsConfig, main  # noqa: E402


if __name__ == "__main__":
    _check_gpu_arch()
    args = tyro.cli(ArgsConfig)

    # Upstream reuses one --save-plot-path for every trajectory in the loop
    # (open_loop_eval.py:348 -> :238), so with --traj-ids 1 2 the second
    # trajectory silently overwrites the first. Its *default* path does
    # interpolate the id (/tmp/open_loop_eval/traj_{id}.jpeg) — but /tmp does
    # not survive a session on this box. Run one trajectory per call and stamp
    # the id into the filename ourselves, so we get both properties: a distinct
    # plot per trajectory, on the persistent volume.
    if args.save_plot_path:
        out = Path(args.save_plot_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        traj_ids = list(args.traj_ids)
        peak = 0
        for traj_id in traj_ids:
            args.traj_ids = [traj_id]
            args.save_plot_path = str(out.with_name(f"{out.stem}_traj{traj_id}{out.suffix}"))
            # Each main() call builds its own RLDXPolicy. Drop the previous one
            # before the next load, otherwise both models are resident at once
            # and the reported peak is roughly double what a single deployed
            # process actually needs. reset_peak_memory_stats alone is not
            # enough — it resets the counter, not the live allocation.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            main(args)
            if torch.cuda.is_available():
                peak = max(peak, torch.cuda.max_memory_allocated())
            print(f"[plot] trajectory {traj_id} -> {args.save_plot_path}")
        if torch.cuda.is_available():
            print(f"[VRAM] peak allocated (per trajectory): {peak / 2**30:.2f} GiB")
        raise SystemExit(0)
    else:
        # No path given: upstream writes to /tmp/open_loop_eval/, which is wiped
        # between sessions. Warn rather than silently lose the output.
        print("[warn] no --save-plot-path; upstream will write to /tmp/open_loop_eval/,")
        print("[warn] which does not survive a session. Pass a path under outputs/.")
        main(args)

    if torch.cuda.is_available():
        print(f"[VRAM] peak allocated : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        print(f"[VRAM] peak reserved  : {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
