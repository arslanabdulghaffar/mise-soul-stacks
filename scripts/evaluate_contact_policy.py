"""Run a frozen learned mug policy with visual stopping and isolated scoring."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LP_NUM_THREADS', '1')

from mise.contact_evaluation import ContactSkillEvaluator
from mise.learned_controller import LearnedContactController
from mise.learning.runtime import OpenVINOContactPolicy, TorchContactPolicy
from mise.manipulation import create_contact_env
from mise.planner import RuleBasedPlanner
from mise.telemetry import CONTACT_COMMAND, file_hash, hardware_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('artifacts/models/contact_act_final'))
    parser.add_argument('--backend', choices=['torch', 'openvino'], default='openvino')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(40000, 40010)))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('seeds must be distinct nonnegative integers')
    if args.output.exists():
        parser.error('Use a new output directory; attempts cannot be overwritten.')
    args.output.mkdir(parents=True)
    scene = Path('data/contact/frozen/scene.xml')
    manifest = dict(schema_version='mise.learned-contact-eval.v1', seeds=args.seeds,
                    backend=args.backend, model=str(args.model),
                    checkpoint_sha256=file_hash(args.model / 'checkpoint.pt'),
                    scene_sha256=file_hash(scene), hardware=hardware_identity(),
                    controller_sha256=file_hash(Path('src/mise/learned_controller.py')),
                    scope='learned_mug_contact_skill', full_task_success=None,
                    completion_source='RGB and robot joints; independent physics scoring')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.backend == 'torch':
        import torch
        torch.set_num_threads(2)
        policy = TorchContactPolicy(args.model / 'checkpoint.pt', device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        policy = OpenVINOContactPolicy(args.model / 'openvino')
    results = []
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
            result = dict(seed=seed, success=bool(controller.done and verifier.success),
                          controller_done=controller.done, failure=controller.failed_reason,
                          physics=verifier.evidence, controller=controller.evidence,
                          wall_seconds=time.monotonic() - began)
            (args.output / f'{seed}.json').write_text(json.dumps(dict(result=result, trace=trace), indent=2))
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(dict(attempts=results,
                successes=sum(row['success'] for row in results), total=len(results), predeclared=len(args.seeds)), indent=2))
            print(json.dumps(result), flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
