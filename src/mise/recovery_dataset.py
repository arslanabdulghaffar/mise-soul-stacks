"""Bootstrap fork-label collection using the privileged scripted drawer teacher."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v3 as iio

from .recovery_labels import ForkLabeler, MuJoCoForkRunner, MuJoCoSnapshot
from .sim import BimanualTableEnv


def collect_drawer_fork_labels(output_dir: Path, *, samples: int, seed_start: int = 8000, with_images: bool = False) -> Path:
    """Create a balanced bootstrap set of actual MuJoCo-forked drawer labels.

    This synthetic fixture owns only a directly driven drawer actuator: normal samples make the
    current arm viable, reassignment samples make only the alternate arm viable,
    and prescribed stop samples make neither viable. These are not grasp labels. The same fork machinery is used
    for actual student-policy candidate continuations after contact validation.
    """

    if samples < 1:
        raise ValueError("samples must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if with_images:
        images_dir.mkdir(exist_ok=True)
    output = output_dir / "fork_labels.jsonl"
    labeler = ForkLabeler()
    env = BimanualTableEnv(seed=seed_start)
    try:
        runner = MuJoCoForkRunner(env._mj, env.model, horizon=150)
        drawer_address = env.model.joint("drawer_joint").qposadr[0]
        drawer_actuator = env.model.actuator("drawer_slide").id
        with output.open("w", encoding="utf-8") as handle:
            for sample_id in range(samples):
                seed = seed_start + sample_id
                env.reset(seed=seed, observe=False)
                # The label set is deliberately class-balanced during bootstrap.
                scenario = ("continue", "replan", "stop")[sample_id % 3]
                current_arm = "A"
                viable_arm = "A" if scenario == "continue" else "B" if scenario == "replan" else None
                # The randomizer may start an ordinary episode with an open
                # drawer. Labels for this skill must instead begin from the
                # same unsatisfied subgoal in every fork.
                env.data.qpos[drawer_address] = 0.0
                env._mj.mj_forward(env.model, env.data)
                snapshot = MuJoCoSnapshot.capture(env.data)

                def rollout(arm: str, fork_index: int) -> bool:
                    def controller(candidate_arm: str, model: Any, data: Any, step: int) -> None:
                        data.ctrl[drawer_actuator] = 0.18 if candidate_arm == viable_arm else 0.0

                    def success(model: Any, data: Any) -> bool:
                        return bool(data.qpos[drawer_address] > 0.15)

                    return runner.rollout_from(snapshot, arm, fork_index, controller, success)

                decision, statistics = labeler.label(current_arm, rollout)
                record: dict[str, Any] = {
                    "id": sample_id,
                    "seed": seed,
                    "skill": "open_drawer",
                    "scenario": scenario,
                    "decision": asdict(decision),
                    "fork_statistics": asdict(statistics),
                    "schema_version": "mise.fork-label.v2",
                    "source": "drawer_actuator_fixture",
                    "training_validated": False,
                    "parent_episode_id": seed,
                    "note": "Scripted actuator viability is prescribed by the fixture; not evidence of contact-based recovery.",
                }
                if with_images:
                    observation = env.observe()
                    image_name = f"{sample_id:06d}.png"
                    iio.imwrite(images_dir / image_name, observation.top)
                    record["top_image"] = f"images/{image_name}"
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        env.close()
    return output
