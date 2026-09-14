"""Run a frozen learned mug policy with visual stopping and isolated scoring."""
import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LP_NUM_THREADS', '1')

from mise.contact_evaluation import ContactSkillEvaluator
from mise.learned_controller import LearnedContactController
from mise.learning.runtime import OpenVINOContactPolicy, TorchContactPolicy
from mise.manipulation import create_contact_env
from mise.planner import RuleBasedPlanner
from mise.telemetry import CONTACT_COMMAND, atomic_json, file_hash, hardware_identity


def frozen_files(model: Path, scene: Path, backend: str) -> dict[str, str]:
    files = {'checkpoint.pt': model / 'checkpoint.pt', 'scene.xml': scene}
    # Meshes and controller/evaluator dependencies also affect physical outcomes.
    root = ET.parse(scene).getroot()
    compiler = root.find('compiler')
    meshdir = scene.parent / compiler.get('meshdir', '')
    for mesh in root.findall('./asset/mesh'):
        name = mesh.get('file')
        if name:
            files[f'mesh/{name}'] = meshdir / name
    for source in Path('src/mise').rglob('*.py'):
        files[str(source)] = source
    files['evaluation_script'] = Path(__file__)
    if backend == 'openvino':
        for name in ('contact_act.xml', 'contact_act.bin', 'model_manifest.json'):
            files[name] = model / 'openvino' / name
        exported = json.loads(files['model_manifest.json'].read_text())
        if exported['source_checkpoint_sha256'] != file_hash(files['checkpoint.pt']):
            raise ValueError('Export manifest does not match the source checkpoint.')
        for name in ('contact_act.xml', 'contact_act.bin'):
            if exported['files_sha256'][name] != file_hash(files[name]):
                raise ValueError(f'Exported file differs from its manifest: {name}')
    return {name: file_hash(path) for name, path in sorted(files.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('artifacts/models/contact_act_final'))
    parser.add_argument('--backend', choices=['torch', 'openvino'], default='openvino')
    parser.add_argument('--device', default=None, help='OpenVINO CPU/GPU/NPU or torch cpu/cuda; defaults to CPU/cpu')
    parser.add_argument('--precision', choices=['auto', 'f32', 'f16', 'bf16'], default='f32')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--scene', type=Path, default=Path('data/contact/frozen/scene.xml'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(40000, 40010)))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('seeds must be distinct nonnegative integers')
    if args.threads < 1 or (args.backend == 'torch' and args.precision != 'f32'):
        parser.error('threads must be positive; torch reference supports f32 only')
    if args.output.exists():
        parser.error('Use a new output directory; attempts cannot be overwritten.')
    scene = args.scene
    hashes = frozen_files(args.model, scene, args.backend)
    args.output.mkdir(parents=True)
    manifest = dict(schema_version='mise.learned-contact-eval.v2', seeds=args.seeds,
                    backend=args.backend, model=str(args.model),
                    checkpoint_sha256=file_hash(args.model / 'checkpoint.pt'),
                    scene_sha256=file_hash(scene), hardware=hardware_identity(),
                    controller_sha256=file_hash(Path('src/mise/learned_controller.py')),
                    files_sha256=hashes, command=CONTACT_COMMAND, max_steps=1200,
                    created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    runtime_config=None,
                    scope='learned_mug_contact_skill', full_task_success=None,
                    completion_source='RGB and robot joints; independent physics scoring')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.backend == 'torch':
        import torch
        torch.set_num_threads(args.threads)
        policy = TorchContactPolicy(args.model / 'checkpoint.pt', device=args.device or 'cpu')
        manifest['runtime_config'] = dict(device=args.device or 'cpu', precision='f32', threads=args.threads)
    else:
        policy = OpenVINOContactPolicy(args.model / 'openvino', device=args.device or 'CPU',
                                      precision=args.precision, threads=args.threads)
        manifest['runtime_config'] = policy.runtime_config
    atomic_json(args.output / 'manifest.json', manifest)
    results = []
    attempt_hashes = {}
    for seed in args.seeds:
        began = time.monotonic()
        env = create_contact_env(seed, scene_path=scene)
        controller = LearnedContactController(env, RuleBasedPlanner().plan(CONTACT_COMMAND), policy)
        verifier = ContactSkillEvaluator(env.model)
        trace = []
        try:
            for step in range(1200):
                action = controller.advance()
                if controller.done or controller.failed_reason:
                    break
                env.step(action, observe=False)
                verifier.update(env.data)
                trace.append(dict(time_s=float(env.data.time), targets=action.tolist(), physics=dict(verifier.evidence)))
            failure = controller.failed_reason
            if not controller.done and not failure:
                failure = 'Evaluation step limit reached.'
            result = dict(seed=seed, success=bool(controller.done and verifier.success),
                          controller_done=controller.done, failure=failure,
                          physics=verifier.evidence, controller=controller.evidence,
                          wall_seconds=time.monotonic() - began)
            atomic_json(args.output / f'{seed}.json', dict(result=result, trace=trace,
                inference_timings_ms=controller.inference_ms))
            attempt_hashes[f'{seed}.json'] = file_hash(args.output / f'{seed}.json')
            results.append(result)
            atomic_json(args.output / 'results.json', dict(attempts=results,
                successes=sum(row['success'] for row in results), total=len(results), predeclared=len(args.seeds),
                files_sha256=attempt_hashes, manifest_sha256=file_hash(args.output / 'manifest.json'),
                complete=len(results) == len(args.seeds)))
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
