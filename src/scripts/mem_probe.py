"""Measure SwinUNETR peak GPU memory for forward+backward at a range of
batch sizes, with cuDNN benchmark on and off — so the per-GPU batch
ceiling is *measured*, not guessed.

Usage:
    python src/scripts/mem_probe.py <experiment_path>
"""

import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models import build_model  # noqa: E402


def _probe(_cfg, _patch, _bs, _bench):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = _bench
    model = build_model('swin_unetr', _cfg, 3, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    try:
        x = torch.randn(_bs, 3, *_patch, device='cuda')
        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, _ = model(x)
            loss = logits.float().mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        peak = torch.cuda.max_memory_allocated() / 1e9
        result = f"PEAK {peak:5.1f} GB  OK"
    except torch.cuda.OutOfMemoryError:
        result = "OOM"
    finally:
        del model, opt
        torch.cuda.empty_cache()
    return result


def main():
    exp = sys.argv[1]
    cfg = yaml.safe_load(open(os.path.join(exp, 'configs.yaml')))
    patch = cfg['trainer']['train_ds']['sample_dimension']
    gpu = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu}  ({total:.0f} GB)   patch={patch}   model=swin_unetr\n")
    for bench in (False, True):
        for bs in (1, 2, 3, 4, 6):
            print(f"  cudnn.benchmark={str(bench):5s}  bs={bs}: "
                  f"{_probe(cfg, patch, bs, bench)}")
        print()


if __name__ == '__main__':
    main()
