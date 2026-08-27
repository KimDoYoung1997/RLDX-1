#!/usr/bin/env python3
"""Plot held-out error per checkpoint against the training loss.

The trainer has no validation path, so overfitting is invisible during the run —
train loss keeps falling either way. This puts the two curves side by side so the
point where they part company is obvious.

It parses the per-trajectory lines out of the eval logs rather than the
"Average MSE across all trajs" line: that line is printed once per trajectory
and each print covers only the trajectories seen so far in that call, so reading
the last one gives the final trajectory's score, not the mean. The mean here is
computed from the per-trajectory numbers.

Usage:
    python run_scripts/eval/plot_holdout_curve.py            # outputs/eval/*.log
    python run_scripts/eval/plot_holdout_curve.py --out foo.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAJ_RE = re.compile(r"MSE for trajectory (\d+): ([0-9.]+), MAE: ([0-9.]+)")


def parse_logs(log_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in sorted(log_dir.glob("holdout_*.log")):
        m = re.match(r"holdout_(\d+)\.log", f.name)
        if not m:
            continue
        rows = TRAJ_RE.findall(f.read_text(errors="ignore"))
        if not rows:
            continue
        step = int(m.group(1))
        out[step] = {
            "traj": [int(t) for t, _, _ in rows],
            "mse": np.array([float(a) for _, a, _ in rows]),
            "mae": np.array([float(b) for _, _, b in rows]),
        }
    return out


def train_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Training loss from the highest-step checkpoint's trainer_state."""
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        return np.array([]), np.array([])
    hist = json.loads(states[-1].read_text())["log_history"]
    pts = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", type=Path, default=Path("outputs/eval"))
    ap.add_argument("--run-dir", type=Path,
                    default=Path("ckpt/rldx1/finetuned/rldx1_ft_ffw_sh5_0826/rldx1_ft_ffw_sh5_0826"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/holdout_curve.png"))
    args = ap.parse_args()

    data = parse_logs(args.log_dir)
    if not data:
        raise SystemExit(f"no holdout_*.log with per-trajectory lines under {args.log_dir}")
    steps = sorted(data)
    mean = np.array([data[s]["mse"].mean() for s in steps])
    ts, tl = train_curve(args.run_dir)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # left: train loss vs held-out MSE ---------------------------------------
    if len(ts):
        ax.plot(ts, tl, color="0.65", lw=1, label="train loss (per 10 steps)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("train loss (log)", color="0.4")
    ax.tick_params(axis="y", labelcolor="0.4")

    axr = ax.twinx()
    for s in steps:  # every held-out episode, so the spread is visible
        axr.scatter([s] * len(data[s]["mse"]), data[s]["mse"], s=22,
                    color="tab:orange", alpha=0.45, zorder=3,
                    label="held-out, per episode" if s == steps[0] else None)
    axr.plot(steps, mean, "o-", color="tab:red", lw=2.2, ms=9, zorder=4,
             label="held-out MSE (mean of 9)")
    best = steps[int(np.argmin(mean))]
    axr.annotate(f"best: step {best:,}\nMSE {mean.min():.4f}",
                 xy=(best, mean.min()), xytext=(12, -34), textcoords="offset points",
                 fontsize=10, color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
    axr.set_ylabel("held-out MSE", color="tab:red")
    axr.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("train loss keeps falling; held-out does not", fontsize=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=9, framealpha=0.9)

    # right: paired per-episode view -----------------------------------------
    # Run-to-run MSE varies ~±15% (unseeded flow-matching sampling), so the mean
    # gap alone is weak. Pairing by episode removes that: if one checkpoint wins
    # on every episode, the ordering is not sampling noise.
    width = 0.8 / len(steps)
    x = np.arange(len(data[steps[0]]["traj"]))
    for i, s in enumerate(steps):
        ax2.bar(x + i * width - 0.4 + width / 2, data[s]["mse"], width,
                label=f"step {s:,}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"ep{t}" for t in data[steps[0]]["traj"]])
    ax2.set_xlabel("held-out episode")
    ax2.set_ylabel("MSE")
    wins = sum(int(np.argmin([data[s]["mse"][k] for s in steps]) == 0) for k in x)
    ax2.set_title(f"paired by episode — step {steps[0]:,} best on {wins}/{len(x)}", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)

    print(f"{'step':>8}  {'held-out MSE':>13}  {'MAE':>8}   vs best")
    for s, m in zip(steps, mean):
        print(f"{s:>8,}  {m:>13.5f}  {data[s]['mae'].mean():>8.5f}   {100 * (m / mean.min() - 1):+6.1f}%")
    print(f"\nstep {steps[0]:,} is best on {wins}/{len(x)} held-out episodes")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
