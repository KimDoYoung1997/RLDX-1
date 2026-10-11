"""Held-out / train-probe flow-matching loss of saved checkpoints (post hoc).

Same measurement as rldx/experiment/val_loss.py (every 10th frame, 2 fixed
noise/time draws, model in eval mode, processor in eval mode with the
checkpoint's own statistics), run on one GPU against checkpoints on disk.

    python posthoc_val_loss.py --ckpts DIR [DIR ...] --heldout DS --train DS --out out.jsonl
"""
import argparse, json, time

import numpy as np
import torch

from rldx.data.dataset.standard_single_step_dataset import StandardSingleStepDataset
from rldx.data.embodiment_tags import EmbodimentTag
from pathlib import Path

from transformers import AutoModel

from rldx.policy.policy_loader import _load_processor, _resolve_model_dir


def to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict) or hasattr(x, "items"):
        return {k: to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_dev(v, dev) for v in x)
    return x


def frames(ds, stride, episodes=None):
    return [i for i, (e, s) in enumerate(ds.all_steps) if s % stride == 0 and (episodes is None or e in episodes)]


@torch.no_grad()
def score(model, ds, idx, draws, bs, dev):
    tot, n = 0.0, 0
    for d in range(draws):
        for b in range(0, len(idx), bs):
            chunk = idx[b:b + bs]
            batch = to_dev(ds.processor.collator([ds[i] for i in chunk]), dev)
            with torch.random.fork_rng(devices=[dev]), torch.autocast("cuda", dtype=torch.bfloat16):
                torch.manual_seed(1_000_003 * (d + 1) + chunk[0])
                out = model(**batch)  # bf16 autocast: training runs bf16 under DeepSpeed
            loss = out["loss"] if "loss" in out else out[0]
            tot += loss.double().item() * len(chunk); n += len(chunk)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="GENERAL_EMBODIMENT")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--probe-episodes", type=int, default=9)
    a = ap.parse_args()
    tag = EmbodimentTag[a.tag] if a.tag in EmbodimentTag.__members__ else EmbodimentTag(a.tag)
    dev = torch.device("cuda:0")
    for ck in a.ckpts:
        t0 = time.time()
        # fp32 weights + bf16 autocast = the trainer's numerics (PolicyLoader would load
        # bf16 weights, and the flow-time Beta sampler has no bf16 kernel)
        mdir = _resolve_model_dir(ck)
        model = AutoModel.from_pretrained(mdir, device_map="cuda:0", torch_dtype=torch.float32, trust_remote_code=True)
        model.eval()
        proc = _load_processor(mdir, model)
        proc.eval()
        mcfg = proc.get_modality_configs()[tag.value]
        res = {"ckpt": ck}
        for name, path, probe in (("heldout", a.heldout, False), ("train_probe", a.train, True)):
            ds = StandardSingleStepDataset(dataset_path=path, embodiment_tag=tag,
                                           modality_configs=mcfg, video_backend="torchcodec")
            ds.set_processor(proc)
            eps = None
            if probe:
                all_eps = sorted({e for e, _ in ds.all_steps})
                eps = set(np.asarray(all_eps)[np.linspace(0, len(all_eps) - 1, a.probe_episodes).round().astype(int)].tolist())
            idx = frames(ds, a.stride, eps)
            res[name + "_loss"] = score(model, ds, idx, a.draws, a.bs, dev)
            res[name + "_frames"] = len(idx)
        res["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(res), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        del model, proc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
