"""Verify retained OpenVINO timing and matched closed-loop evidence without reruns.

This writes a separate report. It never changes benchmark or attempt evidence and
never promotes a failed/zero-success or incomplete pilot into a quality claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mise.learning.openvino_config import detected_core_ultra_series
from mise.telemetry import file_hash


def read_evaluation(directory: Path) -> tuple[dict, dict]:
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    results = json.loads((directory / 'results.json').read_text())
    if manifest.get('schema_version') != 'mise.learned-contact-eval.v2':
        raise ValueError('Evaluation must use the v2 frozen evidence contract.')
    if results.get('manifest_sha256') != file_hash(manifest_path):
        raise ValueError('Evaluation manifest checksum mismatch.')
    seeds = manifest['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Evaluation seed list is empty or invalid.')
    attempts = results['attempts']
    if (not results.get('complete') or results['total'] != len(seeds)
            or results['predeclared'] != len(seeds) or [r['seed'] for r in attempts] != seeds):
        raise ValueError('Evaluation is incomplete or does not match every predeclared seed in order.')
    if set(results['files_sha256']) != {f'{seed}.json' for seed in seeds}:
        raise ValueError('Attempt checksum list differs from the predeclared seeds.')
    for attempt in attempts:
        path = directory / f"{attempt['seed']}.json"
        if results['files_sha256'][path.name] != file_hash(path):
            raise ValueError(f'Attempt checksum mismatch: {path.name}')
        recorded = json.loads(path.read_text())
        if recorded['result'] != attempt:
            raise ValueError('Summary does not match the retained attempt.')
        expected_success = bool(attempt['controller_done'] and attempt['physics'].get('success'))
        if type(attempt['success']) is not bool or attempt['success'] != expected_success:
            raise ValueError('Success is inconsistent with controller and independent physics evidence.')
        if not recorded.get('trace') or not recorded.get('inference_timings_ms'):
            raise ValueError('Retained attempt has no physical trace or measured policy inference.')
    if results['successes'] != sum(row['success'] for row in attempts):
        raise ValueError('Success count does not match the retained attempts.')
    return manifest, results


def read_benchmark(path: Path, manifest: dict) -> dict:
    payload = json.loads(path.read_text())
    if len(payload.get('records', [])) != 1:
        raise ValueError('Expected one benchmark record per configuration.')
    record = payload['records'][0]
    for field, filename in (('model_hash', 'contact_act.xml'), ('weights_hash', 'contact_act.bin'),
                            ('model_manifest_hash', 'model_manifest.json')):
        if record.get(field) != manifest['files_sha256'].get(filename) or not record.get(field):
            raise ValueError(f'Benchmark/evaluation model mismatch: {field}')
    for field in ('device', 'precision', 'compile_config', 'resolved_properties'):
        if record.get(field) is None or record[field] != manifest['runtime_config'].get(field):
            raise ValueError(f'Benchmark/evaluation runtime mismatch: {field}')
    # ov.__version__ includes a build suffix that distribution metadata omits.
    package_version = manifest['hardware']['versions']['openvino']
    measured_version = record.get('openvino_version', '')
    if (record.get('component') != 'contact_act_policy' or not package_version
            or measured_version.split('-', 1)[0] != package_version.split('-', 1)[0]):
        raise ValueError('Benchmark/evaluation component or OpenVINO version mismatch.')
    for field in ('cpu', 'hostname'):
        if record['hardware'][field] != manifest['hardware'][field]:
            raise ValueError(f'Benchmark/evaluation host mismatch: {field}')
    timings = payload.get('raw_timings_ms', [])
    if (len(timings) != record['measured_iterations'] or not timings
            or any(not math.isfinite(t) or t <= 0 for t in timings)):
        raise ValueError('Raw benchmark timings are missing or inconsistent.')
    import numpy as np
    for field, expected in (('median_ms', float(np.median(timings))),
                            ('p95_ms', float(np.percentile(timings, 95)))):
        if not math.isfinite(record[field]) or abs(record[field] - expected) > 1e-6:
            raise ValueError(f'Benchmark {field} does not match raw timings.')
    return record


def compare_pair(reference: Path, candidate: Path, reference_benchmark: Path,
                 candidate_benchmark: Path) -> dict:
    ref, ref_results = read_evaluation(reference)
    opt, opt_results = read_evaluation(candidate)
    for field in ('seeds', 'files_sha256', 'command', 'max_steps', 'scope', 'hardware'):
        if ref[field] != opt[field]:
            raise ValueError(f'Paired evaluation mismatch: {field}')
    if ref['backend'] != 'openvino' or opt['backend'] != 'openvino':
        raise ValueError('This comparison requires reference and candidate OpenVINO evaluations.')
    if ref['runtime_config']['precision'] != 'f32':
        raise ValueError('Reference precision must be f32.')
    if ref['runtime_config'] == opt['runtime_config']:
        raise ValueError('Candidate configuration is identical to reference; no optimization was evaluated.')
    ref_bench = read_benchmark(reference_benchmark, ref)
    opt_bench = read_benchmark(candidate_benchmark, opt)
    if ref_bench['openvino_version'] != opt_bench['openvino_version']:
        raise ValueError('Paired benchmarks use different OpenVINO builds.')
    if ref_bench.get('input_seed') != opt_bench.get('input_seed') or ref_bench.get('input_seed') is None:
        raise ValueError('Benchmarks must use the same recorded input seed.')
    pairs = [dict(seed=r['seed'], reference_success=r['success'], candidate_success=o['success'])
             for r, o in zip(ref_results['attempts'], opt_results['attempts'])]
    regressions = [row['seed'] for row in pairs if row['reference_success'] and not row['candidate_success']]
    # Strict submission gate: ten or more complete attempts, all reference and
    # candidate placements succeed, and at least 100 warmed timing samples each.
    quality = len(pairs) >= 10 and all(row['reference_success'] and row['candidate_success'] for row in pairs)
    timing_ready = all(r['measured_iterations'] >= 100 and r['warmup_iterations'] >= 10
                       for r in (ref_bench, opt_bench))
    faster = opt_bench['median_ms'] < ref_bench['median_ms'] and opt_bench['p95_ms'] <= ref_bench['p95_ms']
    series = detected_core_ultra_series(ref['hardware']['cpu'])
    target = series in (2, 3) and all(
        r.get('declared_intel_core_ultra_series') == series
        and r.get('intel_core_ultra_series_2_or_3_verified') is True for r in (ref_bench, opt_bench))
    evidence = {}
    for label, directory in (('reference', reference), ('candidate', candidate)):
        evidence[label] = dict(directory=str(directory), manifest_sha256=file_hash(directory / 'manifest.json'),
                               results_sha256=file_hash(directory / 'results.json'))
    return dict(schema_version='mise.intel-paired-validation.v1', scope=ref['scope'], full_task_success=None,
                evidence=evidence, benchmark_sha256=dict(reference=file_hash(reference_benchmark),
                                                          candidate=file_hash(candidate_benchmark)),
                reference_config=ref['runtime_config'], candidate_config=opt['runtime_config'],
                seeds=ref['seeds'], pairs=pairs, regressions=regressions,
                reference_successes=ref_results['successes'], candidate_successes=opt_results['successes'],
                quality_preservation_verified=quality, timing_protocol_satisfied=timing_ready,
                measured_median_speedup=ref_bench['median_ms'] / opt_bench['median_ms'],
                latency_improvement_measured=faster, intel_core_ultra_series_2_or_3_verified=target,
                submission_optimization_verified=bool(quality and timing_ready and faster and target),
                note='Gate requires >=10/10 paired mug successes, warmed timing improvement, and matching target host. '
                     'This is experimental learned-mug evidence; it does not establish full-table VLA success.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-benchmark', type=Path, required=True)
    parser.add_argument('--candidate-benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new report path; existing evidence cannot be overwritten.')
    try:
        report = compare_pair(args.reference, args.candidate, args.reference_benchmark, args.candidate_benchmark)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as destination:
        json.dump(report, destination, indent=2)
        destination.write('\n')
    print(json.dumps(report, indent=2))
    if not report['submission_optimization_verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
