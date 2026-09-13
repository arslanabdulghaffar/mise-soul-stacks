"""Export a real contact checkpoint and measure OpenVINO CPU numerical parity."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import openvino as ov
import torch

from mise.learning.data import ContactDataset, read_manifest, sha256
from mise.learning.model import ContactACT, PolicyConfig
from mise.learning.preprocessing import JOINT_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/contact'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/models/contact_act/openvino'))
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--max-error-rad', type=float, default=1e-4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = PolicyConfig(**payload['config'])
    model = ContactACT(config)
    model.load_state_dict(payload['model'])
    model.eval()
    _, records, _ = read_manifest(args.data)
    validation = ContactDataset(args.data, records, payload['statistics'], split='validation', image_size=config.image_size, chunk_size=config.chunk_size)
    example = (torch.zeros(1, 3, 3, config.image_size, config.image_size), torch.zeros(1, 12))
    began = time.monotonic()
    converted = ov.convert_model(model, example_input=example)
    converted.inputs[0].get_tensor().set_names({'images'})
    converted.inputs[1].get_tensor().set_names({'state'})
    converted.outputs[0].get_tensor().set_names({'normalized_actions'})
    args.output.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, str(args.output / 'contact_act.xml'), compress_to_fp16=False)
    conversion_seconds = time.monotonic() - began
    core = ov.Core()
    began = time.monotonic()
    compiled = core.compile_model(str(args.output / 'contact_act.xml'), 'CPU', {'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2})
    compilation_seconds = time.monotonic() - began
    indices = np.linspace(0, len(validation) - 1, min(args.samples, len(validation)), dtype=int)
    results = []
    action_std = np.asarray(payload['statistics']['action_std'])
    for index in indices:
        images, state, _, _ = validation[int(index)]
        with torch.inference_mode():
            reference = model(images[None], state[None]).numpy()
        inputs = {'images': images[None].numpy(), 'state': state[None].numpy()}
        for _ in range(2):
            compiled(inputs)
        began = time.perf_counter()
        actual = compiled(inputs)[compiled.output(0)]
        latency = (time.perf_counter() - began) * 1000
        difference = np.abs(actual - reference) * action_std
        results.append({'validation_observation': int(index), 'max_absolute_error_rad': float(difference.max()),
                        'mean_absolute_error_rad': float(difference.mean()), 'warm_inference_ms': latency})
    maximum = max(row['max_absolute_error_rad'] for row in results)
    report = {'schema_version': 'mise.contact-act-export.v1', 'source_checkpoint_sha256': sha256(args.checkpoint),
              'config': config.to_dict(), 'statistics': payload['statistics'], 'joint_order': JOINT_ORDER,
              'joint_units': 'radians', 'camera_order': ['top', 'wrist_a', 'wrist_b'],
              'input_images': 'N,3,3,H,W float32 RGB, PIL bilinear resize, ImageNet normalization',
              'input_state': 'N,12 float32 normalized with training-split statistics',
              'output': 'normalized absolute actuator-target chunks; denormalize with action statistics',
              'device': 'CPU', 'host': platform.node(), 'processor': platform.processor(), 'precision': 'FP32',
              'openvino_version': ov.__version__, 'torch_version': str(torch.__version__),
              'conversion_seconds': conversion_seconds, 'compilation_seconds': compilation_seconds,
              'validation_samples': len(results), 'max_absolute_error_rad': maximum,
              'acceptance_max_error_rad': args.max_error_rad, 'numerical_parity_passed': maximum <= args.max_error_rad,
              'warm_p50_ms': float(np.median([r['warm_inference_ms'] for r in results])),
              'warm_p95_ms': float(np.percentile([r['warm_inference_ms'] for r in results], 95)),
              'per_sample': results, 'closed_loop_success': None, 'intel_core_ultra_verified': False,
              'files_sha256': {name: sha256(args.output / name) for name in ('contact_act.xml', 'contact_act.bin')},
              'reference': 'https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.convert_model.html'}
    (args.output / 'model_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('statistics', 'per_sample')}, indent=2))
    if not report['numerical_parity_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
