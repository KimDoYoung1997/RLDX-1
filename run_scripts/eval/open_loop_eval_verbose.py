#!/usr/bin/env python
"""``rldx/eval/open_loop_eval.py`` with its results actually printed.

``open_loop_eval.main()`` reports MSE/MAE through ``logging.info``, and calls
``logging.basicConfig(level=INFO)`` to enable it — but transformers has already
configured the root logger by the time that runs, which makes ``basicConfig`` a
no-op. The level stays at WARNING and every result line is swallowed, so the
eval appears to do nothing but write a plot. ``force=True`` overrides the
existing handler configuration.

Also prints peak VRAM, which is the number you want when sizing a deployment
GPU (see RLDX-1-FFW-SH5.md appendix A.6).

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
        f"       See RLDX-1-FFW-SH5.md appendix A.6.\n"
    )


def _check_model_path(model_path: str) -> None:
    """Disambiguate 'local checkpoint' from 'Hub repo id' before loading.

    ``policy_loader._resolve_model_dir`` already falls back to
    ``snapshot_download(model_path)`` when the path does not exist locally, so a
    repo id like ``learner1119/RLDX-1-FT-ffw-sh5`` just works.

    The failure mode is a *local* path that is gone — deleted by
    ``save_total_limit``, or a typo. That gets handed to the Hub as a repo id
    and comes back as::

        HFValidationError: Repo id must be in the form 'repo_name' or
        'namespace/repo_name': 'ckpt/rldx1/finetuned/.../checkpoint-20'

    which says nothing about the real problem. Catch it here and show what does
    exist instead.
    """
    p = Path(model_path)
    if p.exists():
        return
    # A Hub repo id is 'name' or 'namespace/name' — at most one slash, no dots
    # or path anchors. Anything else was meant to be a local path.
    looks_like_repo_id = (
        model_path.count("/") <= 1
        and not model_path.startswith((".", "/", "~"))
        and not p.suffix
    )
    if looks_like_repo_id:
        print(f"[i] {model_path} is not a local path; downloading from the Hub.")
        return

    msg = [f"\nERROR: no checkpoint at {model_path}"]
    parent = p.parent
    if parent.is_dir():
        siblings = sorted(
            (d.name for d in parent.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
            key=lambda n: int(n.split("-")[-1]),
        )
        if siblings:
            msg.append(f"       {parent} currently holds: {', '.join(siblings)}")
            msg.append("       (--save-total-limit evicts older checkpoints during training)")
        else:
            msg.append(f"       {parent} holds no checkpoint-* directories.")
    else:
        msg.append(f"       {parent} does not exist either.")
    msg.append("\n       To evaluate a published model instead, pass its repo id:")
    msg.append("           --model-path learner1119/RLDX-1-FT-ffw-sh5\n")
    raise SystemExit("\n".join(msg))


from rldx.eval.open_loop_eval import ArgsConfig, main  # noqa: E402


if __name__ == "__main__":
    _check_gpu_arch()
    args = tyro.cli(ArgsConfig)
    _check_model_path(args.model_path)

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
