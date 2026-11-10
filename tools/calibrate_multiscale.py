"""Fit score statistics using an explicit normal-only calibration JSONL.

Run from the original repository root; CLI paths resolve against cwd.
The image_dir in YAML resolves against the YAML parent directory.
Requires a checkpoint trained with MultiScaleMoEAD, not original MoEAD.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from easydict import EasyDict
from datasets.custom_dataset import build_custom_dataloader
from models.model_helper import ModelHelper
from utils.misc_helper import update_config


@torch.no_grad()
def calibrate(model, loader):
    model.eval()
    total = torch.zeros(3, dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = torch.zeros_like(total)
    for batch in loader:
        if not torch.all(batch['label'] == 0):
            raise ValueError('Calibration requires normal-only labels')
        for s, error in enumerate(model(batch)['scale_errors']):
            values = error.detach().double()
            total[s] += values.sum().cpu()
            squared[s] += values.square().sum().cpu()
            count[s] += values.numel()
    if (count == 0).any():
        raise ValueError('Empty calibration dataset')
    mean = total / count
    std = (squared / count - mean.square()).clamp_min(0).sqrt()
    model.reconstruction.score_fusion.set_statistics(mean, std)
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'checkpoint', 'normal-meta', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output checkpoint path')
    with open(args.normal_meta, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or any(row.get('label') != 0 for row in rows):
        raise ValueError('Supply a non-empty normal-only calibration JSONL')
    with open(args.config, encoding='utf-8') as f:
        cfg = update_config(EasyDict(yaml.safe_load(f)))
    model = ModelHelper(cfg.net).cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state = {k[7:] if k.startswith('module.') else k: v
             for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    data_cfg = copy.deepcopy(cfg.dataset)
    data_cfg.update(data_cfg.test)
    data_cfg.meta_file = str(Path(args.normal_meta).resolve())
    image_dir = Path(data_cfg.image_reader.kwargs.image_dir)
    if not image_dir.is_absolute():
        image_dir = Path(args.config).resolve().parent / image_dir
    data_cfg.image_reader.kwargs.image_dir = str(image_dir.resolve())
    loader = build_custom_dataloader(data_cfg, training=False, distributed=False)
    mean, std = calibrate(model, loader)
    checkpoint['state_dict'] = {'module.' + k: v.cpu() for k, v in model.state_dict().items()}
    checkpoint['calibration'] = {'mean': mean, 'std': std,
                                 'normal_meta': str(Path(args.normal_meta).resolve())}
    checkpoint['pre_calibration_best_metric'] = checkpoint.pop('best_metric', None)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print('Saved calibrated model to', output)


if __name__ == '__main__':
    main()
