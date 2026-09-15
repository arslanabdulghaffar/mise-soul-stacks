"""The shared, recorded compilation contract for ACT timing and execution."""
from __future__ import annotations

import re
from typing import Any


def compile_config(device: str, precision: str, threads: int = 2) -> dict[str, Any]:
    if precision not in {'auto', 'f32', 'f16', 'bf16'}:
        raise ValueError(f'Unsupported inference precision: {precision}')
    if threads < 1:
        raise ValueError('Inference thread count must be positive.')
    config: dict[str, Any] = {'PERFORMANCE_HINT': 'LATENCY'}
    if precision != 'auto':
        config['INFERENCE_PRECISION_HINT'] = precision
    # INFERENCE_NUM_THREADS is a CPU property; GPU/NPU plugins may reject it.
    if device.split('.')[0] == 'CPU':
        config['INFERENCE_NUM_THREADS'] = threads
    return config


def bind_inputs(model: Any, metadata: dict[str, Any]) -> None:
    size = int(metadata['config']['image_size'])
    if size < 1:
        raise ValueError('Model image_size must be positive.')
    shapes = {'images': [1, 3, 3, size, size], 'state': [1, 25 if metadata['config'].get('action_context', False) else 12]}
    names = [port.any_name for port in model.inputs]
    if len(names) != 2 or set(names) != set(shapes):
        raise ValueError(f'Unsupported model inputs: {names}; expected images and state')
    model.reshape(shapes)


def resolved_properties(compiled: Any) -> dict[str, Any]:
    """Retain resolved hints; these do not prove every operator uses that type."""
    result = {}
    for name in ('EXECUTION_DEVICES', 'INFERENCE_PRECISION_HINT', 'PERFORMANCE_HINT',
                 'INFERENCE_NUM_THREADS'):
        try:
            value = compiled.get_property(name)
            result[name] = list(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
        except Exception:  # Not every device plugin exposes every property.
            result[name] = None
    return result


def detected_core_ultra_series(cpu: str) -> int | None:
    """Conservatively recognize Intel's three-digit Core Ultra SKU convention.

    Unknown or virtualized CPU strings remain unverified. A declaration alone is
    insufficient: e.g. a 155H is Series 1 even when the operator passes Series 2.
    """
    if 'intel' not in cpu.lower():
        return None
    cleaned = re.sub(r'\((?:tm|r)\)|[™®]', '', cpu, flags=re.IGNORECASE)
    match = re.search(r'core\s+ultra\s+(?:x?[579])\s+(?:processor\s+)?([123])\d{2}[a-z]*\b',
                      cleaned, re.IGNORECASE)
    return int(match.group(1)) if match else None
