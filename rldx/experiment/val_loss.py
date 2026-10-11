# SPDX-License-Identifier: Apache-2.0
"""Held-out flow-matching loss logged during training (opt-in).

The sharded training pipeline has no evaluation path (DatasetFactory asserts
eval_strategy == "no"), so checkpoints used to be judged only after the fact.
This callback scores a fixed set of held-out frames every N optimizer steps
with the same masked flow-matching loss the trainer minimises, plus the same
number of frames from the training set ("train probe") so the gap between the
two curves shows over-fitting directly.

Enabled only when RLDX_VAL_DATASET is set; otherwise nothing changes.

    RLDX_VAL_DATASET   held-out LeRobot dataset (same embodiment / modality
                       config as training; needs meta/stats.json present —
                       copy the train split's, the processor normalises with
                       the training statistics anyway)
    RLDX_VAL_EVERY     score every N steps (default 1000; step 0 is scored at
                       train start, which also fails fast if the path is broken)
    RLDX_VAL_STRIDE    use every k-th frame of each episode (default 10)
    RLDX_VAL_DRAWS     noise/time draws per frame, fixed seeds (default 2)
    RLDX_VAL_BATCH     per-rank batch size (default 8)
    RLDX_VAL_TRAIN_PROBE_EPISODES  training episodes in the probe (default 9,
                       evenly spaced; 0 disables the probe)

Determinism: frames, noise and flow time are fixed per (frame, draw), so the
curve moves only because the weights move. Images use the processor's eval
transforms (no colour jitter / random crop) and the model runs in eval mode
(no state dropout), so absolute values sit somewhat off the logged train loss;
compare points of the same curve and the held-out vs train-probe gap.

Results go to trainer.state.log_history (hence trainer_state.json in every
checkpoint) and to <output_dir>/heldout_loss.jsonl.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback

from rldx.data.dataset.standard_single_step_dataset import StandardSingleStepDataset
from rldx.data.embodiment_tags import EmbodimentTag


def _rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class HeldoutLossCallback(TrainerCallback):
    def __init__(self, trainer, processor, config, output_dir, val_path, every, stride, draws,
                 batch_size, probe_episodes):
        self.trainer = trainer
        self.processor = processor
        self.config = config
        self.out = Path(output_dir) / "heldout_loss.jsonl"
        self.val_path = val_path
        self.every, self.stride, self.draws = every, stride, draws
        self.batch_size, self.probe_episodes = batch_size, probe_episodes
        self.sets = None  # built lazily on train begin (processor statistics are final by then)

    def _dataset(self, path):
        spec = self.config.data.datasets[0]
        tag = spec.embodiment_tag
        ds = StandardSingleStepDataset(
            dataset_path=path,
            embodiment_tag=EmbodimentTag(tag),
            modality_configs=self.config.data.modality_configs[tag],
            video_backend=self.config.data.video_backend,
            allow_padding=self.config.data.allow_padding,
        )
        proc = copy.deepcopy(self.processor)  # training statistics, eval transforms
        proc.eval()
        ds.set_processor(proc)
        return ds

    def _build(self):
        val = self._dataset(self.val_path)
        sets = {"heldout": (val, [i for i, (_, s) in enumerate(val.all_steps) if s % self.stride == 0])}
        if self.probe_episodes > 0:
            tr = self._dataset(self.config.data.datasets[0].dataset_paths[0])
            eps = sorted({e for e, _ in tr.all_steps})
            pick = set(np.asarray(eps)[np.linspace(0, len(eps) - 1, self.probe_episodes).round().astype(int)].tolist())
            sets["train_probe"] = (tr, [i for i, (e, s) in enumerate(tr.all_steps) if e in pick and s % self.stride == 0])
        return sets

    @torch.no_grad()
    def _score(self, model):
        rank, world = _rank_world()
        dev = next(model.parameters()).device
        res = {}
        was_training = model.training
        model.eval()
        try:
            for name, (ds, idx) in self.sets.items():
                mine = idx[rank::world]
                tot = torch.zeros(2, device=dev, dtype=torch.float64)  # sum(loss*n), n
                for d in range(self.draws):
                    for b in range(0, len(mine), self.batch_size):
                        chunk = mine[b:b + self.batch_size]
                        batch = self.trainer._prepare_inputs(ds.processor.collator([ds[i] for i in chunk]))
                        with torch.random.fork_rng(devices=[dev]):
                            torch.manual_seed(1_000_003 * (d + 1) + chunk[0])
                            out = model(**batch)
                        loss = out["loss"] if "loss" in out else out[0]  # BatchFeature (UserDict) or tuple
                        tot[0] += loss.double() * len(chunk)
                        tot[1] += len(chunk)
                if world > 1:
                    dist.all_reduce(tot)
                res[name] = (tot[0] / tot[1].clamp(min=1)).item()
                res[name + "_frames"] = len(idx)
        finally:
            model.train(was_training)
        return res

    def _run(self, state, model):
        if self.sets is None:
            self.sets = self._build()
        res = self._score(model)
        rec = {"step": int(state.global_step), "heldout_loss": res["heldout"]}
        if "train_probe" in res:
            rec["train_probe_loss"] = res["train_probe"]
        state.log_history.append(dict(rec))
        if _rank_world()[0] == 0:
            rec.update(heldout_frames=res["heldout_frames"], train_probe_frames=res.get("train_probe_frames"),
                       draws=self.draws, stride=self.stride, val_dataset=self.val_path)
            print(f"[heldout] {json.dumps(rec)}", flush=True)
            with open(self.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._run(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.every == 0:
            self._run(state, model)


def maybe_add_heldout_loss_callback(trainer, config, processor, output_dir, log=print):
    path = os.environ.get("RLDX_VAL_DATASET")
    if not path:
        return
    cb = HeldoutLossCallback(
        trainer, processor, config, output_dir, path,
        every=int(os.environ.get("RLDX_VAL_EVERY", 1000)),
        stride=int(os.environ.get("RLDX_VAL_STRIDE", 10)),
        draws=int(os.environ.get("RLDX_VAL_DRAWS", 2)),
        batch_size=int(os.environ.get("RLDX_VAL_BATCH", 8)),
        probe_episodes=int(os.environ.get("RLDX_VAL_TRAIN_PROBE_EPISODES", 9)),
    )
    trainer.add_callback(cb)
    log(f"[i] HeldoutLossCallback: {path} every {cb.every} steps, stride {cb.stride}, "
        f"{cb.draws} draws, train probe {cb.probe_episodes} ep -> {cb.out}")
