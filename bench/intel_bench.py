"""Benchmark an exported OpenVINO policy and retain submission-grade evidence.

The script deliberately separates a measured OpenVINO result from Intel Core Ultra
verification. Passing ``--intel-core-ultra-series`` records the declared target
series, but verification is true only when the host CPU also identifies itself as
an Intel Core Ultra processor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("At least one timing value is required.")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "Unavailable"


def inventory() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": cpu_name(),
    }


def _shape(port: Any) -> tuple[int, ...]:
    shape = tuple(int(value) for value in port.shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError(f"Dynamic input {port.any_name!r} must be reshaped before benchmarking: {shape}")
    return shape


def benchmark(
    model_path: Path,
    *,
    device: str,
    precision: str,
    warmup: int,
    iterations: int,
    seed: int,
    declared_series: int | None,
) -> dict[str, Any]:
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO is not installed. Install requirements-learning.txt before running the benchmark."
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(
            f"OpenVINO model not found: {model_path}. Train and export it with `make train export`."
        )

    core = ov.Core()
    available = list(core.available_devices)
    if device not in available and not any(item.startswith(f"{device}.") for item in available):
        raise ValueError(f"Requested device {device!r} is unavailable. Available devices: {available}")

    read_started = time.perf_counter()
    model = core.read_model(str(model_path))
    read_ms = (time.perf_counter() - read_started) * 1000
    manifest_path = model_path.with_name("model_manifest.json")
    try:
        model_manifest = json.loads(manifest_path.read_text())
        image_size = int(model_manifest["config"]["image_size"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"A valid {manifest_path.name} with config.image_size is required to bind the exported model inputs."
        ) from exc
    expected_shapes = {
        "images": [1, 3, 3, image_size, image_size],
        "state": [1, 12],
    }
    unknown_inputs = [port.any_name for port in model.inputs if port.any_name not in expected_shapes]
    if unknown_inputs:
        raise ValueError(f"Unsupported model inputs: {unknown_inputs}; expected images and state")
    model.reshape({port.any_name: expected_shapes[port.any_name] for port in model.inputs})
    config: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
    if precision != "auto":
        config["INFERENCE_PRECISION_HINT"] = precision
    compile_started = time.perf_counter()
    compiled = core.compile_model(model, device, config)
    compile_ms = (time.perf_counter() - compile_started) * 1000

    rng = np.random.default_rng(seed)
    inputs: dict[Any, np.ndarray] = {}
    input_schema = []
    for port in compiled.inputs:
        shape = _shape(port)
        # The exported ACT contract is float32. Deterministic bounded noise avoids
        # benchmarking a special all-zero input while remaining reproducible.
        value = rng.normal(0.0, 0.1, size=shape).astype(np.float32)
        inputs[port] = value
        input_schema.append({"name": port.any_name, "shape": list(shape), "dtype": str(value.dtype)})

    for _ in range(warmup):
        compiled(inputs)
    latencies = []
    measured_started = time.perf_counter()
    for _ in range(iterations):
        started = time.perf_counter()
        compiled(inputs)
        latencies.append((time.perf_counter() - started) * 1000)
    measured_seconds = time.perf_counter() - measured_started

    hardware = inventory()
    cpu = hardware["cpu"].lower()
    host_matches_target = "intel" in cpu and "core" in cpu and "ultra" in cpu
    target_verified = bool(declared_series in (2, 3) and host_matches_target)
    bin_path = model_path.with_suffix(".bin")
    try:
        device_name = str(core.get_property(device, "FULL_DEVICE_NAME"))
    except Exception:  # Device plugins expose different property sets.
        device_name = device

    record = {
        "schema_version": "mise.openvino-benchmark.v2",
        "component": "contact_act_policy",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware,
        "declared_intel_core_ultra_series": declared_series,
        "intel_core_ultra_series_2_or_3_verified": target_verified,
        "device": device,
        "device_name": device_name,
        "available_devices": available,
        "precision": precision,
        "openvino_version": ov.__version__,
        "model_path": str(model_path),
        "model_hash": file_sha256(model_path),
        "weights_hash": file_sha256(bin_path) if bin_path.is_file() else None,
        "model_manifest_hash": file_sha256(manifest_path),
        "input_schema": input_schema,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "model_read_ms": read_ms,
        "compile_ms": compile_ms,
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 95),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "throughput_fps": iterations / measured_seconds,
        "timings_ms": latencies,
        "closed_loop_success": None,
        "quality_preservation_verified": False,
        "note": (
            "Intel Core Ultra Series 2/3 host identity verified; paired closed-loop quality remains separate."
            if target_verified
            else "Measured OpenVINO inference only; this host is not verified as Intel Core Ultra Series 2/3."
        ),
    }
    return record


def write_report(record: dict[str, Any], json_output: Path, csv_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mise.benchmark-archive.v1",
        "hardware": record["hardware"],
        "records": [{key: value for key, value in record.items() if key != "timings_ms"}],
        "raw_timings_ms": record["timings_ms"],
    }
    json_output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "measured_at_utc", "component", "device", "device_name", "precision",
        "model_hash", "measured_iterations", "median_ms", "p95_ms", "min_ms",
        "max_ms", "throughput_fps", "compile_ms", "openvino_version",
        "declared_intel_core_ultra_series", "intel_core_ultra_series_2_or_3_verified",
        "quality_preservation_verified",
    ]
    with csv_output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/contact_act_final/openvino/contact_act.xml"),
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device, for example CPU, GPU, or NPU")
    parser.add_argument("--precision", choices=["auto", "f32", "f16"], default="f32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--intel-core-ultra-series", type=int, choices=[2, 3])
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/benchmarks/latest.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("artifacts/benchmarks/latest.csv"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        record = benchmark(
            args.model,
            device=args.device,
            precision=args.precision,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
            declared_series=args.intel_core_ultra_series,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    write_report(record, args.json_output, args.csv_output)
    printable = {key: value for key, value in record.items() if key not in {"timings_ms", "input_schema"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
