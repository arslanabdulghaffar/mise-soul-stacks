"""Scene-aware symbolic teacher used to produce verified planner SFT records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio

from .planner import parse_plan, RuleBasedPlanner, DEFAULT_COMMAND
from .sim import BimanualTableEnv


@dataclass(frozen=True, slots=True)
class SceneFacts:
    drawer_open: bool
    plate_side: str
    mug_side: str


def inspect_scene(env: BimanualTableEnv) -> SceneFacts:
    drawer_address = env.model.joint("drawer_joint").qposadr[0]
    plate_address = env.model.joint("plate_free").qposadr[0]
    mug_address = env.model.joint("mug_free").qposadr[0]
    # Arms are separated along Y: A is the positive-Y arm and B is negative-Y.
    return SceneFacts(
        drawer_open=bool(env.data.qpos[drawer_address] > 0.09),
        plate_side="A" if env.data.qpos[plate_address + 1] >= 0 else "B",
        mug_side="A" if env.data.qpos[mug_address + 1] >= 0 else "B",
    )


class SceneAwarePlannerTeacher:
    """Emit valid dependency graphs from command intent plus visible scene facts."""

    def plan(self, command: str, facts: SceneFacts) -> dict[str, list[dict[str, Any]]]:
        # Privileged scene facts may satisfy implicit prerequisites, but never
        # override an explicit arm assignment or ordered action.
        plan = RuleBasedPlanner().plan(command, drawer_open=facts.drawer_open)
        records = []
        for step in plan.steps:
            record = asdict(step)
            record["to"] = record.pop("target")
            record["needs"] = list(record["needs"])
            record["preconditions"] = list(record["preconditions"])
            records.append(record)
        payload = {"steps": records}
        parse_plan(payload)
        return payload



def generate_planner_records(output_dir: Path, *, count: int, seed_start: int = 5000, with_images: bool = True) -> Path:
    """Generate scene/command/verified-plan records, optionally with top images."""

    if count < 1:
        raise ValueError("count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if with_images:
        images_dir.mkdir(exist_ok=True)
    manifest = output_dir / "planner_records.jsonl"
    teacher = SceneAwarePlannerTeacher()
    env = BimanualTableEnv(seed=seed_start)
    try:
        with manifest.open("w", encoding="utf-8") as handle:
            for record_id in range(count):
                seed = seed_start + record_id
                observation = env.reset(seed=seed, observe=with_images)
                facts = inspect_scene(env)
                plan = teacher.plan(DEFAULT_COMMAND, facts)
                record: dict[str, Any] = {
                    "id": record_id,
                    "seed": seed,
                    "command": DEFAULT_COMMAND,
                    "scene_facts": asdict(facts),
                    "plan": plan,
                    "schema_valid": True,
                    "source": "privileged_symbolic_teacher",
                    "visual_grounding_validated": False,
                }
                if with_images:
                    assert observation is not None
                    image_name = f"{record_id:06d}.png"
                    iio.imwrite(images_dir / image_name, observation.top)
                    record["top_image"] = f"images/{image_name}"
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        env.close()
    return manifest
