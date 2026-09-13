"""Collect reproducible, pre-action contact demonstrations with isolated outcomes.

Example: python scripts/collect_contact_data.py --output data/contact --limit 5
Run the same command without --limit to resume all predeclared attempts. Existing
attempts, including failed attempts, are never replaced or silently retried.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("top", "wrist_a", "wrist_b")
SCHEMA = "mise.contact-demonstrations.v1"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS")
SOURCE_FILES = (
    "src/mise/manipulation.py", "src/mise/contact_vision.py", "src/mise/kinematics.py",
    "src/mise/contact_evaluation.py", "src/mise/evaluation.py", "src/mise/sim.py",
    "src/mise/planner.py", "src/mise/types.py", "src/mise/randomization.py",
    "scripts/build_contact_scene.py", "scripts/build_scene.py", "scripts/collect_contact_data.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def episode_plan(train: int, validation: int, test: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    episode_index = 0
    for split, count, start in (("train", train, 20000), ("validation", validation, 30000), ("test", test, 40000)):
        groups[split] = []
        for offset in range(count):
            identifier = f"contact-{split}-{start + offset}"
            groups[split].append({"episode_index": episode_index, "episode_id": identifier,
                                  "parent_episode_id": identifier, "seed": start + offset,
                                  "split": split, "path": f"episodes/{identifier}.npz"})
            episode_index += 1
    # Make a small complete split available for integration without changing any
    # seed or split assignment after observing an outcome.
    prefix = groups["train"][:3] + groups["validation"][:1] + groups["test"][:1]
    return prefix + groups["train"][3:] + groups["validation"][1:] + groups["test"][1:]


def create_manifest(output: Path, args: argparse.Namespace) -> dict[str, Any]:
    from runpy import run_path
    from mise.telemetry import CONTACT_COMMAND, hardware_identity

    source_before = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    generated_scene = run_path(str(ROOT / "scripts/build_contact_scene.py"))["build_contact_scene"]()
    scene_tree = ET.parse(generated_scene)
    compiler = scene_tree.getroot().find("compiler")
    meshdir = (generated_scene.parent / compiler.get("meshdir", "")).resolve()
    mesh_hashes = {}
    for mesh in scene_tree.getroot().findall("asset/mesh"):
        path = meshdir / mesh.get("file")
        mesh_hashes[str(path.relative_to(ROOT))] = sha256(path)
    frozen_directory = output / "frozen"
    frozen_directory.mkdir(parents=True, exist_ok=True)
    # Preserve portability within the repository while changing no physical
    # parameter. Workers only read this immutable, dataset-specific XML.
    compiler.set("meshdir", os.path.relpath(meshdir, frozen_directory))
    scene_path = frozen_directory / "scene.xml"
    scene_tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    for name in SOURCE_FILES:
        target = frozen_directory / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    source_after = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    if source_before != source_after:
        raise RuntimeError("Collection sources changed while freezing the manifest; rerun before collecting.")
    manifest = {
        "schema_version": SCHEMA, "created_at": utc_now(), "task": CONTACT_COMMAND,
        "scope": "contact_skill", "controller": "contact_expert", "learned_policy": False,
        "splits": {"train": args.train, "validation": args.validation, "test": args.test},
        "episodes": episode_plan(args.train, args.validation, args.test),
        "seed_policy": "All attempts are predeclared. Failures are retained, never replaced by successful seeds.",
        "split_policy": "Whole parent episodes remain in one split; no adjacent-frame split leakage.",
        "initial_randomization": "Mug XY uniformly +/-8 mm; nominal contact scene only.",
        "scene_path": "frozen/scene.xml", "scene_sha256": sha256(scene_path),
        "source_sha256": source_before, "mesh_sha256": mesh_hashes, "hardware": hardware_identity(),
        "control_hz": 30, "record_hz": args.record_hz, "cameras": list(CAMERAS),
        "max_simulation_seconds": args.max_simulation_seconds,
        "worker_threads": {key: os.environ[key] for key in THREAD_VARIABLES},
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    manifest_path = output / "manifest.json"
    # This file is frozen before even the first policy step is attempted.
    write_json(manifest_path, manifest)
    metadata = {
        "schema_version": SCHEMA, "manifest_sha256": sha256(manifest_path), "task": CONTACT_COMMAND,
        "control_hz": 30, "record_hz": args.record_hz, "image_shape": [256, 256, 3],
        "image_dtype": "uint8", "cameras": list(CAMERAS), "color_order": "RGB",
        "observation_keys": [*(f"observation.images.{camera}" for camera in CAMERAS), "observation.state"],
        "action_key": "action", "action_alignment": "pre-action; observation_action_indices selects first future 30 Hz action",
        "units": "radians", "joint_order": [f"arm_{arm}.{joint}" for arm in ("a", "b")
            for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")],
        "array_schema": {
            "observation.images.*": "uint8 [T_obs,256,256,3]", "observation.state": "float32 [T_obs,12]",
            "action": "float32 [T_action,12]", "observation_action_indices": "int64 [T_obs]",
            "observation.timestamp": "float64 [T_obs], elapsed simulation seconds before action",
            "action.timestamp": "float64 [T_action], elapsed simulation seconds before action",
        },
        "chunk_contract": "At observation i, take actions[index:index+chunk_size]. Pad past episode end only with an explicit is_pad mask; never cross episodes.",
        "contact_validated_demonstrations": True,
        "contact_validation_note": "Every attempt is checked by an independent physical verifier. Train imitation only on episodes with contact_skill_success=true; retain failures for separate diagnostics.",
        "observation_boundary": "NPZ observations contain RGB and twelve robot joint positions only. Object poses/contact labels are confined to outcome JSON and never become policy inputs.",
        "controller_provenance": "deterministic_camera_guided_contact_expert",
        "geometry_limitation": "Finite frictional finger pads and a 44 mm solid mug cylinder; hardware calibration pending.",
        "full_task_success": None,
    }
    write_json(output / "metadata.json", metadata)
    return manifest


def verify_manifest(output: Path, manifest: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {"train": args.train, "validation": args.validation, "test": args.test}
    if manifest["splits"] != expected or manifest["record_hz"] != args.record_hz:
        raise ValueError("Existing manifest is immutable. Use its original counts/rate or a new output directory.")
    for name, expected_hash in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected_hash:
            raise ValueError(f"Pinned source changed: {name}. Use a new dataset directory for the new revision.")
    if sha256(output / manifest["scene_path"]) != manifest["scene_sha256"]:
        raise ValueError("Frozen scene hash changed; refusing to mix scene revisions.")
    for name, expected_hash in manifest["mesh_sha256"].items():
        if sha256(ROOT / name) != expected_hash:
            raise ValueError(f"Pinned mesh changed: {name}")


def collect_episode(output_string: str, spec: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    # Spawned processes receive the thread limits before importing numerical or
    # rendering libraries. No shell profile or global runtime setting is edited.
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    import numpy as np
    from mise.contact_evaluation import ContactSkillEvaluator
    from mise.manipulation import ManipulationController, create_contact_env
    from mise.planner import RuleBasedPlanner

    output = Path(output_string)
    start = time.monotonic()
    attempt = {**spec, "task": manifest["task"], "started_at": utc_now(),
               "success": False, "contact_skill_success": False, "full_task_success": None,
               "scope": "contact_skill", "assisted": False, "controller": "contact_expert",
               "source_manifest_sha256": sha256(output / "manifest.json")}
    # A persisted started record makes an interrupted worker an explicit attempt.
    attempt_path = output / "attempts" / f"{spec['episode_id']}.json"
    write_json(attempt_path, {**attempt, "status": "running"})
    images: dict[str, list[Any]] = {camera: [] for camera in CAMERAS}
    states, actions, indices, observation_times, action_times = [], [], [], [], []
    phase_events = []
    env = controller = verifier = None
    failure = None
    timings = {"render_wall_seconds": 0.0, "control_wall_seconds": 0.0, "physics_wall_seconds": 0.0}
    simulation_seconds = 0.0
    try:
        env = create_contact_env(spec["seed"], scene_path=output / manifest["scene_path"])
        controller = ManipulationController(env, RuleBasedPlanner().plan(manifest["task"]))
        verifier = ContactSkillEvaluator(env.model)
        origin = float(env.data.time)
        record_every = env.control_hz // manifest["record_hz"]
        previous_phase = None
        for action_index in range(round(manifest["max_simulation_seconds"] * env.control_hz)):
            before_control = time.monotonic()
            target = np.asarray(controller.advance(), dtype=np.float64)
            timings["control_wall_seconds"] += time.monotonic() - before_control
            if controller.phase != previous_phase:
                phase_events.append({"action_index": action_index, "simulation_time": float(env.data.time) - origin,
                                     "phase": controller.phase})
                previous_phase = controller.phase
            if controller.done or controller.failed_reason:
                break
            if not np.isfinite(target).all() or target.shape != (12,):
                raise ValueError("Controller returned invalid joint targets")
            elapsed = float(env.data.time) - origin
            limits = env.model.actuator_ctrlrange[env._arm_ctrl.ravel()]
            applied_action = np.clip(target, limits[:, 0], limits[:, 1])
            if action_index % record_every == 0:
                before_render = time.monotonic()
                observation = env.observe()
                for camera in CAMERAS:
                    images[camera].append(np.asarray(getattr(observation, camera), dtype=np.uint8))
                states.append(np.asarray(observation.joint_positions, dtype=np.float32))
                indices.append(action_index)
                observation_times.append(elapsed)
                timings["render_wall_seconds"] += time.monotonic() - before_render
            actions.append(applied_action.astype(np.float32))
            action_times.append(elapsed)
            before_physics = time.monotonic()
            env.step(applied_action, observe=False)
            timings["physics_wall_seconds"] += time.monotonic() - before_physics
            # This evaluator only writes outcome evidence. Its output is never
            # passed to the controller or included in observation arrays.
            verifier.update(env.data)
        else:
            failure = "Predeclared simulation-time budget exhausted"
        simulation_seconds = float(env.data.time) - origin
        success = bool(controller.done and not controller.failed_reason and verifier.success and not failure)
        failure = failure or controller.failed_reason
        if not success and not failure:
            failure = "Control sequence ended without independently verified grasp, sustained lift, and stable released placement"
        attempt.update(success=success, contact_skill_success=success,
                       reason="Contact skill independently verified" if success else failure,
                       status="completed" if success else "failed",
                       completed_steps=list(controller.completed_steps) if success else [],
                       controller_completed_steps=list(controller.completed_steps),
                       controller_evidence=dict(controller.evidence),
                       physics_verification=dict(verifier.evidence))
    except Exception as exc:
        attempt.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            env.close()
    arrays = {f"observation.images.{camera}": np.stack(values) if values else np.empty((0, 256, 256, 3), dtype=np.uint8)
              for camera, values in images.items()}
    arrays.update({"observation.state": np.asarray(states, dtype=np.float32).reshape(-1, 12),
                   "action": np.asarray(actions, dtype=np.float32).reshape(-1, 12),
                   "observation_action_indices": np.asarray(indices, dtype=np.int64),
                   "observation.timestamp": np.asarray(observation_times, dtype=np.float64),
                   "action.timestamp": np.asarray(action_times, dtype=np.float64)})
    data_path = output / spec["path"]
    temporary = data_path.with_suffix(".npz.tmp")
    before_write = time.monotonic()
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(data_path)
    timings["archive_wall_seconds"] = time.monotonic() - before_write
    attempt.update(finished_at=utc_now(), observations=len(states), actions=len(actions),
                   simulation_seconds=simulation_seconds, wall_seconds=time.monotonic() - start,
                   size_bytes=data_path.stat().st_size, sha256=sha256(data_path),
                   timing=timings, phase_events=phase_events,
                   outcome_path=f"attempts/{spec['episode_id']}.json")
    write_json(attempt_path, attempt)
    return attempt


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/contact")
    parser.add_argument("--train", type=int, default=30)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--record-hz", type=int, choices=(3, 5, 6, 10, 15, 30), default=10)
    parser.add_argument("--max-simulation-seconds", type=float, default=40.0)
    parser.add_argument("--limit", type=int, default=0, help="Maximum pending attempts this invocation; 0 means all")
    args = parser.parse_args()
    if any(count < 1 or count > 5000 for count in (args.train, args.validation, args.test)) or args.limit < 0:
        parser.error("Split counts must be 1..5000 and limit must be nonnegative")
    if not 1 <= args.max_simulation_seconds <= 180:
        parser.error("Simulation budget must be 1..180 seconds")
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(ROOT / "src"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("attempts", "episodes"):
        (output / name).mkdir(exist_ok=True)
    # Lock only this collection directory. Multiple workers share immutable
    # manifest/scene files while the parent alone appends the episode index.
    import fcntl
    with (output / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another collector owns this dataset directory") from exc
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else create_manifest(output, args)
        verify_manifest(output, manifest, args)
        index_path = output / "episodes.jsonl"
        existing = read_records(index_path)
        known = {record["episode_id"] for record in existing}
        # Recover durable worker records after parent interruption, retaining
        # unfinished workers explicitly rather than silently reattempting seeds.
        for spec in manifest["episodes"]:
            attempt_path = output / "attempts" / f"{spec['episode_id']}.json"
            if spec["episode_id"] not in known and attempt_path.exists():
                record = json.loads(attempt_path.read_text())
                if record["status"] == "running":
                    record.update(status="interrupted", reason="Collection process ended before an outcome was recorded",
                                  success=False, contact_skill_success=False, finished_at=utc_now())
                    write_json(attempt_path, record)
                with index_path.open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                existing.append(record)
                known.add(spec["episode_id"])
        pending = [spec for spec in manifest["episodes"] if spec["episode_id"] not in known]
        if args.limit:
            pending = pending[:args.limit]
        print(json.dumps({"event": "collection_started", "output": str(output), "pending": len(pending),
                          "already_recorded": len(existing), "workers": args.workers,
                          "manifest_sha256": sha256(manifest_path)}), flush=True)
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(collect_episode, str(output), spec, manifest): spec for spec in pending}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {**spec, "task": manifest["task"], "status": "failed", "success": False,
                              "contact_skill_success": False, "full_task_success": None,
                              "reason": f"Worker failure: {type(exc).__name__}: {exc}", "finished_at": utc_now()}
                    write_json(output / "attempts" / f"{spec['episode_id']}.json", record)
                with index_path.open("a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                existing.append(record)
                print(json.dumps({"event": "episode_finished", "episode_id": record["episode_id"],
                                  "split": record["split"], "success": record["success"],
                                  "reason": record.get("reason"), "wall_seconds": record.get("wall_seconds"),
                                  "observations": record.get("observations"), "size_bytes": record.get("size_bytes")}), flush=True)
        counts = {split: {"attempts": sum(r["split"] == split for r in existing),
                          "successes": sum(r["split"] == split and r["success"] for r in existing)}
                  for split in ("train", "validation", "test")}
        summary = {"updated_at": utc_now(), "scope": "contact_skill", "full_task_success": None,
                   "splits": counts, "predeclared_attempts": len(manifest["episodes"]),
                   "recorded_attempts": len(existing), "pending_attempts": len(manifest["episodes"]) - len(existing),
                   "this_invocation_seconds": time.monotonic() - started,
                   "total_bytes": sum(record.get("size_bytes", 0) for record in existing)}
        write_json(output / "collection_summary.json", summary)
        print(json.dumps({"event": "collection_finished", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
