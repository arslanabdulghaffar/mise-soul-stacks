"""Episode-disjoint contact demonstrations with pre-action frame alignment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import CAMERA_KEYS, preprocess_images, contextual_state


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(directory: Path):
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('contact_validated_demonstrations') is not True:
        raise ValueError('Training requires contact-validated demonstrations; drawer-actuator fixtures are prohibited')
    records = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines() if line.strip()]
    parents, paths = {}, set()
    verified = []
    for row in records:
        split = row['split']
        if split == 'val':
            split = 'validation'
        if split not in ('train', 'validation', 'test'):
            raise ValueError(f'Unknown episode split: {split}')
        row = {**row, 'split': split}
        parent = str(row.get('parent_episode_id', row.get('seed')))
        if parent in parents and parents[parent] != split:
            raise ValueError(f'Parent episode {parent} leaks across splits')
        parents[parent] = split
        path = (directory / row['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or path in paths:
            raise ValueError('Episode paths must be unique and stay inside the dataset')
        paths.add(path)
        if row.get('success', row.get('contact_skill_success')) is True:
            if not path.exists():
                raise ValueError(f'Missing verified episode: {path}')
            verified.append(row)
    if not any(row['split'] == 'train' for row in verified) or not any(row['split'] == 'validation' for row in verified):
        raise ValueError('At least one verified training and one separate validation episode are required')
    return metadata, verified, records


def load_episode(path: Path):
    with np.load(path, allow_pickle=False) as bundle:
        episode = {key: bundle[key] for key in bundle.files}
    state, actions = episode['observation.state'], episode['action']
    indices = episode.get('observation_action_indices')
    if indices is None:
        raise ValueError(f'{path.name}: explicit pre-action observation_action_indices are required')
    if state.ndim != 2 or state.shape[1] != 12 or actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError('Every state and action must contain the twelve robot joints')
    if indices.shape != (len(state),) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('One integer first-action index is required for every observation')
    if np.any(indices < 0) or np.any(indices >= len(actions)) or np.any(np.diff(indices) <= 0):
        raise ValueError('Observation-action indices must be strictly increasing and in range')
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError('Demonstration contains nonfinite robot joints or targets')
    for key in CAMERA_KEYS:
        images = episode[key]
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[0] != len(state) or images.shape[-1] != 3:
            raise ValueError(f'Camera {key} is not aligned RGB uint8')
    return episode


def training_statistics(directory: Path, records):
    states, actions = [], []
    for row in records:
        if row['split'] != 'train':
            continue
        episode = load_episode(directory / row['path'])
        states.append(episode['observation.state'])
        actions.append(episode['action'])
    state, action = np.concatenate(states), np.concatenate(actions)
    return {'state_mean': state.mean(0).tolist(), 'state_std': np.maximum(state.std(0), .05).tolist(),
            'action_mean': action.mean(0).tolist(), 'action_std': np.maximum(action.std(0), .05).tolist(),
            'fit_split': 'train', 'minimum_std_rad': .05}


class ContactDataset(Dataset):
    def __init__(self, directory: Path, records, statistics, *, split: str, image_size: int, chunk_size: int, action_context: bool = False):
        self.samples, self.episodes = [], []
        self.chunk_size = chunk_size
        self.state_mean = np.asarray(statistics['state_mean'], np.float32)
        self.state_std = np.asarray(statistics['state_std'], np.float32)
        self.action_mean = np.asarray(statistics['action_mean'], np.float32)
        self.action_std = np.asarray(statistics['action_std'], np.float32)
        for row in records:
            if row['split'] != split:
                continue
            raw = load_episode(directory / row['path'])
            states = (raw['observation.state'].astype(np.float32) - self.state_mean) / self.state_std
            if action_context:
                states = np.stack([contextual_state(
                    raw['observation.state'][i], statistics,
                    raw['action'][index - 1] if index else raw['observation.state'][i],
                    float(raw['observation.timestamp'][i]))
                    for i, index in enumerate(raw['observation_action_indices'])])
            images = np.stack([preprocess_images([raw[key][i] for key in CAMERA_KEYS], image_size) for i in range(len(raw['observation.state']))])
            episode = {'images': torch.from_numpy(images),
                       'states': torch.from_numpy(states),
                       'actions': torch.from_numpy((raw['action'].astype(np.float32) - self.action_mean) / self.action_std),
                       'indices': raw['observation_action_indices'], 'record': row}
            ep_index = len(self.episodes)
            self.episodes.append(episode)
            self.samples.extend((ep_index, frame) for frame in range(len(images)))
        if not self.samples:
            raise ValueError(f'No successful {split} observations')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_index, frame = self.samples[index]
        episode = self.episodes[episode_index]
        start = int(episode['indices'][frame])
        indices = torch.arange(start, start + self.chunk_size)
        padding = indices >= len(episode['actions'])
        actions = episode['actions'][indices.clamp(max=len(episode['actions']) - 1)]
        return episode['images'][frame], episode['states'][frame], actions, padding
