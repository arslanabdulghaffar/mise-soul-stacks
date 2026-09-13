"""Train a task-specific ACT-style CVAE on verified, episode-disjoint contact data."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from mise.learning.data import ContactDataset, read_manifest, sha256, training_statistics
from mise.learning.model import ContactACT, PolicyConfig
from mise.learning.preprocessing import JOINT_ORDER


def evaluate(model, loader, device, action_std):
    model.eval()
    errors, first_errors, active_errors, weights = [], [], [], []
    with torch.inference_mode():
        for images, state, target, padding in loader:
            images, state, target, padding = (item.to(device) for item in (images, state, target, padding))
            prediction = model(images, state)
            error = (prediction - target).abs() * action_std
            valid = ~padding
            errors.append(float((error * valid[..., None]).sum().cpu()))
            active_errors.append(float((error[..., 6:] * valid[..., None]).sum().cpu()))
            first_errors.append(float(error[:, 0].sum().cpu()))
            weights.append((int(valid.sum().cpu()), len(state)))
    count = sum(x[0] for x in weights)
    samples = sum(x[1] for x in weights)
    return {'action_mae_rad': sum(errors) / (count * 12),
            'arm_B_action_mae_rad': sum(active_errors) / (count * 6),
            'first_action_mae_rad': sum(first_errors) / (samples * 12), 'observations': samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/contact'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/models/contact_act'))
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=30)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--kl-weight', type=float, default=10.0)
    parser.add_argument('--validation-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train-backbone', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.chunk_size < 1 or args.validation_every < 1:
        parser.error('steps, batch size, chunk size and validation interval must be positive')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    metadata, records, all_records = read_manifest(args.data)
    statistics = training_statistics(args.data, records)
    config = PolicyConfig(image_size=args.image_size, chunk_size=args.chunk_size)
    args.output.mkdir(parents=True, exist_ok=True)
    training = ContactDataset(args.data, records, statistics, split='train', image_size=config.image_size, chunk_size=config.chunk_size)
    validation = ContactDataset(args.data, records, statistics, split='validation', image_size=config.image_size, chunk_size=config.chunk_size)
    model = ContactACT(config, pretrained_backbone=args.resume is None).to(args.device)
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True)
        if previous['config'] != config.to_dict() or previous['statistics'] != statistics:
            raise ValueError('Resume requires identical model configuration and train-only normalization')
        model.load_state_dict(previous['model'])
    if not args.train_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    iterator = iter(train_loader)
    std = torch.tensor(statistics['action_std'], device=args.device)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow')}
    manifest = {'schema_version': 'mise.contact-act-training.v1', 'method': 'compact ACT-style CVAE',
                'task': 'mug pick_place with arm B to table_upper_right', 'config': config.to_dict(),
                'statistics': statistics, 'joint_order': JOINT_ORDER, 'joint_units': 'radians',
                'action_type': 'absolute position actuator targets', 'observation_alignment': 'pre-action',
                'source_metadata': metadata, 'versions': versions,
                'data_sha256': {row['path']: sha256(args.data / row['path']) for row in records},
                'split_episodes': {split: [row['path'] for row in records if row['split'] == split] for split in ('train', 'validation', 'test')},
                'retained_failed_episodes': len(all_records) - len(records),
                'backbone': 'torchvision ResNet18 ImageNet1K V1', 'backbone_trainable': args.train_backbone,
                'device': args.device, 'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'CPU',
                'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'source_sha256': {str(path): sha256(path) for path in (Path(__file__), *Path('src/mise/learning').glob('*.py'))},
                'closed_loop_success': None, 'intel_core_ultra_verified': False,
                'reference': 'https://arxiv.org/abs/2304.13705'}
    (args.output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    initial = evaluate(model, valid_loader, args.device, std)
    print(json.dumps({'step': 0, 'validation': initial, 'train_observations': len(training)}), flush=True)
    best, started = float('inf'), time.monotonic()
    logs = args.output / 'training_metrics.jsonl'
    with logs.open('w') as log:
        log.write(json.dumps({'step': 0, 'validation': initial}) + '\n')
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            model.train()
            if not args.train_backbone:
                model.backbone.eval()
            images, state, target, padding = (item.to(args.device) for item in batch)
            prediction, kl = model.training_prediction(images, state, target, padding)
            valid = (~padding).unsqueeze(-1)
            reconstruction = ((prediction - target).abs() * valid).sum() / (valid.sum() * config.joints)
            loss = reconstruction + args.kl_weight * kl
            if not torch.isfinite(loss):
                raise RuntimeError('Training produced a nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % args.validation_every == 0 or step == args.steps:
                metrics = evaluate(model, valid_loader, args.device, std)
                row = {'step': step, 'loss': float(loss.detach().cpu()), 'reconstruction': float(reconstruction.detach().cpu()),
                       'kl': float(kl.detach().cpu()), 'validation': metrics, 'wall_seconds': time.monotonic() - started}
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + '\n')
                log.flush()
                if metrics['arm_B_action_mae_rad'] < best:
                    best = metrics['arm_B_action_mae_rad']
                    payload = {'model': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                               'config': config.to_dict(), 'statistics': statistics, 'step': step,
                               'validation': metrics, 'task': manifest['task']}
                    temp = args.output / 'checkpoint.pt.tmp'
                    torch.save(payload, temp)
                    temp.replace(args.output / 'checkpoint.pt')
                    manifest.update(best_step=step, best_validation=metrics, checkpoint_sha256=sha256(args.output / 'checkpoint.pt'))
                    (args.output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    manifest.update(training_wall_seconds=time.monotonic() - started, finished_steps=args.steps)
    (args.output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
