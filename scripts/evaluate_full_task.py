"""Run the fixed-seed full task without video and retain every outcome."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from mise.evaluation import wilson_interval
from mise.full_task import RECOVERY_MODES, FullTaskController, FullTaskEvaluator, create_full_env
from mise.planner import RuleBasedPlanner
from mise.telemetry import atomic_json, file_hash, utc_now


def evaluate(seed: int, *, recovery_mode: str = "adaptive", scene_path: Path | None = None) -> dict:
    started = time.monotonic()
    env = create_full_env(seed, scene_path=scene_path)
    try:
        controller = FullTaskController(env, RuleBasedPlanner().plan("Set the table."),
                                        recovery_mode=recovery_mode,
                                        episode_id=f"headless-{recovery_mode}-{seed}",
                                        memory_write=False)
        evaluator = FullTaskEvaluator(env)
        object_addresses = []
        for name in ("plate", "fork", "spoon", "mug"):
            start = env.model.joint(f"{name}_free").qposadr[0]
            object_addresses.extend(range(start, start + 7))
        direct_object_write = False
        for _ in range(7500):
            before = env.data.qpos[object_addresses].copy()
            action = controller.advance()
            if not np.array_equal(before, env.data.qpos[object_addresses]):
                direct_object_write = True
                break
            if controller.done or controller.failed_reason:
                break
            env.step(action, observe=False)
            evaluator.update(env.data)
        passed = bool(controller.done and evaluator.success and not direct_object_write)
        failure = controller.failed_reason
        if not passed and not failure:
            if direct_object_write:
                failure = "Controller modified free-object state."
            elif not controller.done:
                failure = "Controller did not finish within the evaluation step budget."
            elif evaluator.snapshot:
                snapshot = evaluator.snapshot
                unmet = [f"placement:{name}" for name, valid in snapshot.objects_in_goal.items() if not valid]
                if not snapshot.drawer_open:
                    unmet.append("drawer_open")
                if not snapshot.handoff_complete:
                    unmet.append("physical_handoff")
                unmet.extend(snapshot.violations)
                failure = "Independent evaluation failed: " + ", ".join(unmet or ["contact provenance or geometry certification"])
            else:
                failure = "No independent evaluation snapshot was produced."
        return {
            "seed": seed,
            "recovery_mode": recovery_mode,
            "success": passed,
            "failure": failure,
            "completed_steps": list(controller.completed_steps),
            "recovery_attempts": controller.recovery_attempts,
            "simulation_seconds": float(env.data.time),
            "wall_seconds": time.monotonic() - started,
            "direct_object_write": direct_object_write,
            "randomization": {
                "object_offsets_m": env.last_randomization.object_offsets_m,
                "mass_scale": env.last_randomization.mass_scale,
                "friction_scale": env.last_randomization.friction_scale,
            },
            "evaluation": evaluator.snapshot.to_dict() if evaluator.snapshot else None,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=Path("eval/seeds.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/full_task_fixed_seeds.json"))
    parser.add_argument("--recovery-mode", choices=RECOVERY_MODES, default="adaptive")
    args = parser.parse_args()
    seeds = yaml.safe_load(args.seeds.read_text())["required"]
    results = []
    for seed in seeds:
        result = evaluate(int(seed), recovery_mode=args.recovery_mode)
        results.append(result)
        print(f"seed={seed} success={result['success']} recovery={result['recovery_attempts']}", flush=True)
    successes = sum(result["success"] for result in results)
    report = {
        "schema_version": "mise.full-evaluation.v1",
        "created_at": utc_now(),
        "scope": "headless_fixed_seed_regression",
        "command": "Set the table.",
        "recovery_mode": args.recovery_mode,
        "successes": successes,
        "total": len(results),
        "wilson95": wilson_interval(successes, len(results)),
        "source_sha256": file_hash(Path(__file__).resolve().parents[1] / "src/mise/full_task.py"),
        "note": "No videos are captured here; use `mise full` or the console for camera artifacts.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "success", "completed_steps", "recovery_attempts",
                                                    "simulation_seconds", "wall_seconds", "direct_object_write", "failure"))
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    print(f"full task: {successes}/{len(results)}; wrote {args.output} and {csv_path}")
    if successes != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
