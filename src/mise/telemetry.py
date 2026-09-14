"""Durable run evidence. Browser delivery never owns the source of truth."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
from .presentation import CAMERAS
TERMINAL = {"completed", "failed", "stopped"}
FIXTURE_NOTE = (
    "Scripted integration fixture: a separate actuator drives the drawer directly. "
    "This does not demonstrate a robot grasp, a learned skill, or full table-setting success."
)
CONTACT_NOTE = (
    "Deterministic contact expert: the instruction selects a supported manipulation and arm. "
    "Robot joint targets move the object through physical contacts; no object attachment or teleport is used. "
    "RGB color localization and calibrated robot kinematics guide this deterministic baseline; "
    "contact outcomes are recorded as physics evidence. It is not a learned ACT policy."
)
FULL_NOTE = (
    "Verified deterministic full-task expert: seven camera-grounded contact steps physically open the drawer, "
    "set four objects, and transfer the spoon from arm A to arm B. An isolated evaluator checks contact "
    "provenance, stable goals, handoff, retrieval, and forbidden collisions. No object attachment or teleport "
    "is used. This is an engineering expert, not the learned ACT policy."
)
CONTACT_COMMAND = "Place the mug in the upper-right with arm B."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def hardware_identity() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    versions = {}
    for package in ("mujoco", "numpy", "fastapi", "openvino"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"hostname": socket.gethostname(), "cpu": cpu, "platform": platform.platform(),
            "device": "CPU", "python": platform.python_version(), "versions": versions,
            "intel_target_verified": False, "inference_device": None, "precision": None}


class RunRecorder:
    def __init__(self, directory: Path, run: dict[str, Any]):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.sequence = run.get("last_sequence", 0)
        self.save()

    def save(self) -> None:
        atomic_json(self.directory / "run.json", self.run)

    def event(self, kind: str, simulation_time: float = 0.0, *,
              persist_run: bool = True, **payload: Any) -> dict[str, Any]:
        self.sequence += 1
        active = self.run.get("active_step") or {}
        active_arm = active.get("arm")
        controller = self.run.get("controller", "scripted_drawer")
        arms = {arm: controller if active_arm in (arm, "both") else "hold_target" for arm in ("A", "B")}
        if self.run.get("phase") is not None:
            payload.setdefault("phase", self.run["phase"])
        if active.get("id") is not None:
            payload.setdefault("step_id", active["id"])
        event = {"schema_version": "mise.event.v1", "run_id": self.run["id"],
                 "sequence": self.sequence, "timestamp": utc_now(),
                 "monotonic_time": time.monotonic(), "simulation_time": simulation_time,
                 "type": kind, "active_skill": active.get("skill"),
                 "arm_state": arms,
                 "monitor": {"state": "unavailable", "confidence": None},
                 "config_hash": self.run["config_hash"], "payload": payload}
        with (self.directory / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        self.run["last_sequence"] = self.sequence
        self.run["simulation_time"] = simulation_time
        if persist_run:
            self.save()
        return event

    def finish(self, status: str, summary: dict[str, Any]) -> None:
        if status not in TERMINAL:
            raise ValueError("finish requires a terminal run status")
        self.run.update(status=status, phase=status, active_step=None, summary=summary, finished_at=utc_now())
        atomic_json(self.directory / "summary.json", summary)
        self.event("run_finished", self.run.get("simulation_time", 0.0), status=status, summary=summary)
        files = {p.name: file_hash(p) for p in self.directory.iterdir()
                 if p.is_file() and p.suffix in {".json", ".jsonl", ".mp4", ".csv"}
                 and p.name != "manifest.json"}
        atomic_json(self.directory / "manifest.json", {
            "schema_version": "mise.manifest.v1", "run_id": self.run["id"],
            "config_hash": self.run["config_hash"], "files_sha256": files,
            "hardware": self.run["hardware"], "checkpoint_hash": None,
            "scope": self.run.get("scope", "drawer_fixture"),
            "disclosure": self.run.get("disclosure", FIXTURE_NOTE),
            "sources_sha256": self.run.get("sources_sha256", {})})


def read_trace(directory: Path, after: int = 0) -> list[dict[str, Any]]:
    path = directory / "trace.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A writer may still be appending the final line.
        if event["sequence"] > after:
            events.append(event)
    return events
