"""Task-specific learned inference. No simulator object poses or contacts enter."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .preprocessing import preprocess_images, normalize_state, denormalize_actions


class TorchContactPolicy:
    def __init__(self, checkpoint: str | Path, *, device: str = 'cpu'):
        import torch
        from .model import ContactACT, PolicyConfig
        self.torch, self.device = torch, device
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = PolicyConfig(**payload['config'])
        self.statistics = payload['statistics']
        self.model = ContactACT(self.config)
        self.model.load_state_dict(payload['model'])
        self.model.to(device).eval()

    def predict_chunk(self, observation):
        images = preprocess_images((observation.top, observation.wrist_a, observation.wrist_b), self.config.image_size)
        state = normalize_state(observation.joint_positions, self.statistics)
        with self.torch.inference_mode():
            output = self.model(self.torch.from_numpy(images[None]).to(self.device), self.torch.from_numpy(state[None]).to(self.device))
        return denormalize_actions(output[0].cpu().numpy(), self.statistics)


class OpenVINOContactPolicy:
    def __init__(self, directory: str | Path, *, device: str = 'CPU'):
        import openvino as ov
        directory = Path(directory)
        self.metadata = json.loads((directory / 'model_manifest.json').read_text())
        self.statistics = self.metadata['statistics']
        core = ov.Core()
        self.compiled = core.compile_model(str(directory / 'contact_act.xml'), device,
                                           {'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
        self.output = self.compiled.output(0)

    def predict_chunk(self, observation):
        images = preprocess_images((observation.top, observation.wrist_a, observation.wrist_b), self.metadata['config']['image_size'])
        state = normalize_state(observation.joint_positions, self.statistics)
        output = self.compiled({'images': images[None], 'state': state[None]})[self.output]
        return denormalize_actions(output[0], self.statistics)
