"""Isolated MuJoCo execution and durable multi-camera evidence recording."""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .telemetry import CAMERAS, CONTACT_NOTE, FIXTURE_NOTE, FULL_NOTE, RunRecorder, utc_now
from .types import Plan, SkillStep
from .presentation import PresentationRenderer, capture_settings


def _saved_plan(run: dict[str, Any]) -> Plan:
    """Execute the accepted, recorded graph rather than reinterpret its command."""

    record = run["plan"]
    steps = []
    for entry in record["steps"]:
        entry = dict(entry)
        entry["needs"] = tuple(entry.get("needs", ()))
        entry["preconditions"] = tuple(entry.get("preconditions", ()))
        steps.append(SkillStep(**entry))
    return Plan(steps, record["source"])


def run_worker(directory_string: str, display_queue: Any, controls: Any) -> None:
    # Import MuJoCo only after choosing the renderer, in the worker process.
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MISE_MUJOCO_GL", "osmesa"))
    import imageio.v2 as iio
    import numpy as np
    from PIL import Image

    from .evaluation import TaskEvaluator
    from .scripted import DrawerExpert
    from .sim import BimanualTableEnv

    directory = Path(directory_string)
    recorder = RunRecorder(directory, json.loads((directory / "run.json").read_text()))
    contact_run = recorder.run["controller"] == "contact_expert"
    full_run = False
    scope = "contact_skill" if contact_run else "drawer_fixture"
    disclosure = CONTACT_NOTE if contact_run else FIXTURE_NOTE
    env, controller, contact_evaluator, recovery_memory = None, None, None, None
    writers: dict[str, Any] = {}
    started = time.monotonic()
    paused = assisted = False
    status, reason = "completed", disclosure
    timings: list[float] = []
    snapshot = None
    completed_steps: list[int] = []
    contact_evidence: dict[str, Any] = {}
    frame_count = {camera: 0 for camera in CAMERAS}
    latest_images: dict[str, bytes] = {}
    latest_camera_timestamps: dict[str, str] = {}
    index = 0
    plan = None
    presentation = None
    settings = capture_settings(recorder.run.get("view_quality", "economy"))
    capture_hz = settings["camera_capture_hz"]
    try:
        plan = _saved_plan(recorder.run)
        if contact_run:
            from .full_task import is_full_plan
            full_run = is_full_plan(plan)
            if full_run:
                scope, disclosure = "full_task", FULL_NOTE
        capture_hz = settings["camera_capture_hz"]
        recorder.run["camera_configuration"] = settings
        recorder.run.update(status="running", phase="initializing")
        recorder.event("run_started", disclosure=disclosure)
        if contact_run:
            from .skills import create_skill_env, create_skill_controller, create_skill_evaluator, validate_contact_plan, is_drawer_plan
            validate_contact_plan(plan)
            env = create_skill_env(plan, seed=recorder.run["seed"])
        else:
            env = BimanualTableEnv(seed=recorder.run["seed"])
        # Presets are applied once to the episode's initial state and logged.
        # Object changes never happen after the first controller action.
        perturbations: list[dict[str, Any]] = []
        if recorder.run["preset"] == "low_friction":
            for name in ("plate", "mug", "spoon", "fork"):
                body = env.model.body(name).id
                env.model.geom_friction[env.model.geom_bodyid == body] *= 0.7
            perturbations.append({"type": "object_friction_scale", "factor": 0.7})
        elif recorder.run["preset"] == "displaced_objects":
            objects = sorted({step.object for step in plan.steps if step.object and step.object != "drawer"}) if contact_run else ["plate"]
            for name in objects:
                address = env.model.joint(f"{name}_free").qposadr[0]
                env.data.qpos[address] += 0.02
                perturbations.append({"type": "initial_position_offset", "object": name, "offset_m": [0.02, 0, 0]})
            env._mj.mj_forward(env.model, env.data)
        recorder.run["scene_configuration"] = {
            "randomization": asdict(env.last_randomization) if env.last_randomization is not None else None,
            "scene": ("full_table_setting" if full_run else "physical_drawer" if is_drawer_plan(plan) else "contact") if contact_run else "legacy_drawer",
            "perturbations": perturbations,
        }
        recorder.event("scene_ready", float(env.data.time), preset=recorder.run["preset"],
                       scene_configuration=recorder.run["scene_configuration"])
        if contact_run:
            if full_run and recorder.run.get("recovery_mode") == "adaptive":
                from .recovery_memory import RecoveryMemory
                recovery_memory = RecoveryMemory(directory.parent / "recovery_memory.sqlite3")
            controller = create_skill_controller(
                env, plan, recovery_mode=recorder.run.get("recovery_mode", "adaptive"),
                recovery_memory=recovery_memory, episode_id=recorder.run["id"], memory_write=True)
            contact_evaluator = create_skill_evaluator(env, plan)
            expert, fixture_steps = None, None
        else:
            expert = DrawerExpert(env)
            fixture_steps = round(expert.duration_seconds * env.control_hz)
        presentation = PresentationRenderer(env, settings["width"])
        evaluator = TaskEvaluator(env.model)
        for camera in CAMERAS:
            writers[camera] = iio.get_writer(str(directory / f"{camera}.mp4"), fps=capture_hz[camera],
                                              codec="libx264", quality=7, macro_block_size=16)
        capture_every = {camera: max(1, round(env.control_hz / capture_hz[camera])) for camera in CAMERAS}
        previous_step, previous_phase = None, None

        def update_context() -> None:
            nonlocal previous_step, previous_phase, completed_steps, contact_evidence
            active = controller.active_step if controller is not None else plan.steps[0]
            phase = controller.phase if controller is not None else "drawer_actuator_fixture"
            recorder.run["active_step"] = asdict(active) if active is not None else None
            recorder.run["phase"] = phase
            if active is not None and active.id != previous_step:
                recorder.event("step_started", float(env.data.time), step_id=active.id,
                               skill=active.skill, arm=active.arm, object=active.object, target=active.target)
                previous_step = active.id
            if phase != previous_phase:
                recorder.event("phase_changed", float(env.data.time), phase=phase)
                previous_phase = phase
            if controller is not None:
                completed_steps = list(controller.completed_steps)
                contact_evidence = dict(controller.evidence)
                if contact_evaluator is not None:
                    contact_evidence["physics_verification"] = dict(contact_evaluator.evidence)
                recorder.run["completed_steps"] = completed_steps
                recorder.run["contact_evidence"] = contact_evidence
                pending = list(controller.events)
                controller.events.clear()
                for entry in pending:
                    payload = dict(entry)
                    kind = payload.pop("type", "controller_event")
                    event_time = float(payload.pop("simulation_time", env.data.time))
                    recorder.event(kind, event_time, **payload)

        def capture(cameras=CAMERAS) -> None:
            captured_at = utc_now()
            recorder.run["last_observation_at"] = captured_at
            recorder.run.setdefault("camera_observed_at", {})
            for camera in cameras:
                frame = presentation.render(camera)
                writers[camera].append_data(frame)
                encoded = io.BytesIO()
                Image.fromarray(frame).save(encoded, format="JPEG", quality=85)
                latest_images[camera] = encoded.getvalue()
                latest_camera_timestamps[camera] = captured_at
                recorder.run["camera_observed_at"][camera] = captured_at
                temp = directory / f"{camera}.jpg.tmp"
                temp.write_bytes(latest_images[camera])
                temp.replace(directory / f"{camera}.jpg")
                frame_count[camera] += 1
            progress = len(completed_steps) / len(plan.steps) if contact_run else min(1.0, index / fixture_steps)
            recorder.event("observation", float(env.data.time), progress=progress,
                           frame_timestamp=captured_at, cameras=list(cameras),
                           frame_index=dict(frame_count),
                           recording_time_s={camera: frame_count[camera] / capture_hz[camera] for camera in CAMERAS},
                           joint_positions=env.joint_positions().tolist())
            if len(latest_images) == len(CAMERAS):
                try:
                    display_queue.put_nowait({"run_id": recorder.run["id"], "timestamp": captured_at,
                                              "camera_timestamps": dict(latest_camera_timestamps),
                                              "simulation_time": float(env.data.time),
                                              "images": dict(latest_images)})
                except queue.Full:
                    pass  # Display drops do not remove persisted video or trace evidence.

        update_context()
        capture()
        while True:
            if time.monotonic() - started > settings["wall_guard_s"]:
                status, reason = "failed", f"{settings['wall_guard_s']} second worker wall-clock guard exceeded."
                recorder.event("episode_timeout", float(env.data.time), reason=reason)
                break
            if controller is not None:
                if controller.failed_reason:
                    status, reason = "failed", controller.failed_reason
                    break
                if controller.done:
                    if contact_evaluator is None or not contact_evaluator.success:
                        status, reason = "failed", ("Full sequence ended without satisfying every independent task predicate."
                                                    if full_run else "Contact sequence ended without verified grasp and the requested physical outcome.")
                    else:
                        reason = ("All seven task steps and full-task predicates were independently verified."
                                  if full_run else "Requested contact skill independently verified.")
                    break
            elif index >= fixture_steps:
                break
            try:
                while True:
                    command = controls.get_nowait()
                    if command == "stop":
                        status, reason = "stopped", "Operator stopped the run at a control boundary."
                        recorder.event("operator_stop", float(env.data.time))
                        break
                    if command == "pause" and not paused:
                        paused, assisted = True, True
                        recorder.run["status"] = "paused"
                        recorder.event("operator_pause", float(env.data.time),
                                       reason="Simulation paused at a control boundary; this run is assisted.")
                    elif command == "resume" and paused:
                        paused = False
                        recorder.run["status"] = "running"
                        recorder.event("operator_resume", float(env.data.time))
            except queue.Empty:
                pass
            if status == "stopped":
                break
            if paused:
                time.sleep(0.02)
                continue
            tick = time.monotonic()
            update_context()
            if controller is not None:
                targets = controller.advance()
                update_context()
                # A terminal observation may finish the controller without another
                # action. Do not execute a stale target after a stop or completion.
                if controller.done or controller.failed_reason:
                    continue
            else:
                progress = min(1.0, index / (fixture_steps * 0.55))
                targets = expert.home.copy()
                targets[:6] = expert.home[:6] * (1.0 - progress) + expert.open_pose * progress
                env.set_drawer(0.18 * min(1.0, max(0.0, (index - fixture_steps * 0.35) / (fixture_steps * 0.4))))
            targets = np.asarray(targets, dtype=np.float64)
            env.step(targets, observe=False)
            index += 1
            # The append-only trace durably retains every 30 Hz action. Avoid
            # rewriting the much larger run snapshot at the same rate; camera
            # observations and state transitions checkpoint it frequently.
            recorder.event("action", float(env.data.time), persist_run=False, action_sequence=index,
                           joint_targets=targets.tolist(), source=recorder.run["controller"])
            # TaskEvaluator never supplies a target or progress decision to the
            # controller. Its independent full-task predicates remain evidence.
            if full_run:
                contact_evaluator.update(env.data)
                snapshot = contact_evaluator.snapshot
            else:
                snapshot = evaluator.update(env.data)
                if contact_evaluator is not None:
                    contact_evaluator.update(env.data)
            if contact_evaluator is not None:
                update_context()
            due = tuple(camera for camera in CAMERAS if index % capture_every[camera] == 0)
            if due:
                capture(due)
            timings.append((time.monotonic() - tick) * 1000)
            if not full_run:
                time.sleep(max(0.0, 1.0 / env.control_hz - (time.monotonic() - tick)))
        update_context()
        capture()
        if snapshot is not None:
            recorder.event("evaluation", float(env.data.time), scope=scope,
                           evaluator=snapshot.to_dict(), controller_input=False,
                           contact_evidence=contact_evidence)
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
        recorder.event("worker_error", recorder.run.get("simulation_time", 0), reason=reason)
    finally:
        for camera, writer in writers.items():
            try:
                writer.close()
                recorder.run["artifacts"][f"video_{camera}"] = f"/api/runs/{recorder.run['id']}/artifacts/{camera}.mp4"
            except Exception as exc:
                recorder.event("artifact_error", recorder.run.get("simulation_time", 0),
                               camera=camera, reason=f"{type(exc).__name__}: {exc}")
        if (directory / "top.mp4").exists():
            recorder.run["artifacts"]["video"] = recorder.run["artifacts"].get("video_top")
        if env is not None:
            try:
                try:
                    if presentation is not None:
                        presentation.close()
                finally:
                    env.close()
            except Exception as exc:
                status, reason = "failed", f"Environment cleanup failed ({type(exc).__name__}: {exc})."
                recorder.event("worker_error", recorder.run.get("simulation_time", 0), reason=reason)
        wall = time.monotonic() - started
        simulated = recorder.run.get("simulation_time", 0)
        timing = {"p50_loop_work_ms": float(np.percentile(timings, 50)) if timings else None,
                  "p95_loop_work_ms": float(np.percentile(timings, 95)) if timings else None,
                  "wall_seconds": wall, "simulation_seconds": simulated,
                  "real_time_factor": simulated / wall if wall else None,
                  "policy_latency_ms": None, "capture_hz": capture_hz["top"],
                  "frames_per_camera": frame_count["top"],
                  "camera_capture_hz": capture_hz, "camera_frame_counts": frame_count,
                  "camera_width": settings["width"], "camera_height": settings["height"],
                  "view_quality": settings["profile"],
                  "scope": f"{scope}_with_cameras_and_recording"}
        with (directory / "timing.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timing))
            writer.writeheader()
            writer.writerow(timing)
        skill_success = bool(status == "completed" and plan is not None and controller is not None and controller.done
                             and contact_evaluator is not None and contact_evaluator.success
                             and not controller.failed_reason and len(completed_steps) == len(plan.steps))
        recovery_attempts = int(getattr(controller, "recovery_attempts", 0)) if controller is not None else 0
        summary = {"scope": scope, "full_task_success": skill_success if full_run else None,
                   "completed_steps": completed_steps, "collision_count": snapshot.forbidden_collision_count if full_run and snapshot else None, "recovery_attempts": 0,
                   "assisted": assisted, "reason": reason, "wall_seconds": wall,
                   "timing": timing, "checkpoint_hash": None}
        summary["recovery_attempts"] = recovery_attempts
        if full_run:
            summary.update(full_task_evidence=contact_evidence,
                           recovery_mode=recorder.run.get("recovery_mode", "adaptive"),
                           recovery_memory=recovery_memory.summary() if recovery_memory is not None else None,
                           evaluation=snapshot.to_dict() if snapshot else None,
                           controller_completed_steps=completed_steps,
                           completed_steps=completed_steps,
                           autonomous_full_task_success=skill_success and not assisted)
        elif contact_run:
            summary.update(contact_skill_success=skill_success, contact_evidence=contact_evidence,
                           controller_completed_steps=completed_steps,
                           completed_steps=completed_steps if skill_success else [],
                           autonomous_contact_skill_success=skill_success and not assisted)
        else:
            summary["fixture_drawer_open"] = snapshot.drawer_open if snapshot else None
        recorder.finish(status, summary)


def run_fixture(directory_string: str, display_queue: Any, controls: Any) -> None:
    """Compatibility entry point for archived fixture launch integrations."""

    run_worker(directory_string, display_queue, controls)
