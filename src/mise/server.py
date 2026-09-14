"""Single-operator local API and read-only replay server."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import queue
import re
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .planner import RuleBasedPlanner
from .telemetry import (CAMERAS, CONTACT_COMMAND, CONTACT_NOTE, FIXTURE_NOTE, FULL_NOTE, ROOT, TERMINAL, RunRecorder, digest,
                        file_hash, hardware_identity, read_trace, utc_now)
from .worker import run_worker


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    command: str = Field(default=CONTACT_COMMAND, min_length=3, max_length=500)
    seed: int = Field(default=1001, ge=0, le=2**31 - 1, strict=True)
    preset: Literal["nominal", "low_friction", "displaced_objects"] = "nominal"
    controller: Literal["contact_expert", "scripted_drawer"] = "contact_expert"
    view_quality: Literal["economy", "balanced", "detail"] = "economy"
    recovery_mode: Literal["none", "blind_retry", "adaptive"] = "adaptive"


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "stop"]


class PlanRequest(BaseModel):
    command: str = Field(min_length=3, max_length=500)


def _interrupted_summary(run: dict[str, Any], reason: str) -> dict[str, Any]:
    """Preserve observed partial progress when a worker cannot finalize itself."""

    summary = {"scope": run.get("scope", "drawer_fixture"), "full_task_success": None,
               "reason": reason, "completed_steps": run.get("completed_steps", []),
               "wall_seconds": None}
    if run.get("scope") == "full_task":
        summary["full_task_success"] = False
    elif run.get("controller") == "contact_expert":
        summary.update(contact_skill_success=False, autonomous_contact_skill_success=False,
                       contact_evidence=run.get("contact_evidence", {}))
    return summary


class RunManager:
    def __init__(self, root: Path, *, read_only: bool = False, launch_workers: bool = True):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lease = None
        if not read_only:
            import fcntl
            self._lease = (root / '.runtime.lock').open('a+')
            try:
                fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lease.close()
                raise RuntimeError(f'Another runtime owns {root.resolve()}. Use its console or choose a different --output directory.') from None
        self.read_only = read_only
        self.launch_workers = launch_workers
        self.hardware = hardware_identity()
        self.context = mp.get_context("spawn")
        self.display = self.context.Queue(maxsize=8)
        self.controls = None
        self.process = None
        self.active_id: str | None = None
        self.frames: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()
        # A previous process cannot remain live across a server restart.
        if not read_only:
            for run in self.list():
                if run["status"] not in TERMINAL:
                    recorder = RunRecorder(self.directory(run["id"]), run)
                    recorder.finish("failed", _interrupted_summary(
                        run, "Server restarted before the run completed."))

    def _drain(self) -> None:
        while not self.closing.is_set():
            try:
                item = self.display.get(timeout=0.2)
                with self.lock:
                    self.frames = {item["run_id"]: item}
            except queue.Empty:
                pass

    def directory(self, run_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{12}", run_id):
            raise HTTPException(404, "Unknown run")
        return self.root / run_id

    def get(self, run_id: str) -> dict[str, Any]:
        path = self.directory(run_id) / "run.json"
        if not path.exists():
            raise HTTPException(404, "Unknown run")
        run = json.loads(path.read_text())
        if (run_id == self.active_id and self.process is not None
                and not self.process.is_alive() and run["status"] not in TERMINAL):
            recorder = RunRecorder(path.parent, run)
            recorder.finish("failed", _interrupted_summary(
                run, f"Worker exited unexpectedly ({self.process.exitcode})."))
            run = recorder.run
        return run

    def list(self) -> list[dict[str, Any]]:
        records = []
        for path in self.root.glob("*/run.json"):
            try:
                records.append(self.get(path.parent.name))
            except (ValueError, HTTPException):
                continue
        return sorted(records, key=lambda run: run["created_at"], reverse=True)

    def create(self, request: RunRequest) -> dict[str, Any]:
        if self.read_only:
            raise HTTPException(403, "Replay mode is read-only")
        try:
            plan = RuleBasedPlanner().plan(request.command)
            if request.controller == "contact_expert":
                from .skills import validate_contact_plan, is_drawer_plan
                from .full_task import is_full_plan
                validate_contact_plan(plan)
                if (is_drawer_plan(plan) or is_full_plan(plan)) and request.preset != 'nominal':
                    raise ValueError('This physical controller currently supports the nominal scene only.')
            elif not re.fullmatch(r"open (?:the )?(?:top )?drawer(?: with arm a)?[.!]?", request.command.lower()):
                raise ValueError("The scripted_drawer fixture supports only 'Open the drawer with arm A.'.")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        full_run = request.controller == "contact_expert" and is_full_plan(plan)
        scope = "full_task" if full_run else "contact_skill" if request.controller == "contact_expert" else "drawer_fixture"
        disclosure = FULL_NOTE if full_run else CONTACT_NOTE if request.controller == "contact_expert" else FIXTURE_NOTE
        with self.lock:
            if self.active_id and self.get(self.active_id)["status"] not in TERMINAL:
                raise HTTPException(409, "One run is already active")
            if self.process is not None and self.process.is_alive():
                raise HTTPException(409, "Previous worker is still closing")
            if self.process is not None:
                self.process.join(timeout=1)
            if self.controls is not None:
                self.controls.close()
            run_id = uuid.uuid4().hex[:12]
            if request.controller == 'contact_expert':
                # Freeze the generated scene before hashing the accepted run.
                # The worker rebuilds the same deterministic scene at startup.
                from runpy import run_path
                run_path(str(ROOT / 'scripts/build_contact_scene.py'))['build_contact_scene']()
                if full_run:
                    run_path(str(ROOT / 'scripts/build_full_scene.py'))['build_full_scene']()
                elif is_drawer_plan(plan):
                    run_path(str(ROOT / 'scripts/build_drawer_scene.py'))['build_drawer_scene']()
            sources = {str(path.relative_to(ROOT)): file_hash(path)
                       for path in [ROOT / "configs/scene_randomization.yaml", ROOT / "configs/evaluator.yaml",
                                    ROOT / "assets/generated/mise_bimanual.xml", ROOT / "src/mise/scripted.py",
                                    ROOT / "src/mise/worker.py", ROOT / "src/mise/presentation.py", ROOT / "src/mise/evaluation.py",
                                    ROOT / "src/mise/randomization.py", ROOT / "src/mise/sim.py",
                                    ROOT / "src/mise/manipulation.py", ROOT / "src/mise/planner.py",
                                    ROOT / "src/mise/kinematics.py", ROOT / "src/mise/contact_vision.py",
                                    ROOT / "src/mise/contact_evaluation.py",
                                    ROOT / "src/mise/supervisor.py", ROOT / "src/mise/recovery_memory.py",
                                    ROOT / "src/mise/skills.py", ROOT / "src/mise/drawer_skill.py",
                                    ROOT / "src/mise/full_task.py", ROOT / "src/mise/full_vision.py",
                                    ROOT / "src/mise/scene_io.py",
                                    ROOT / "src/mise/drawer_evaluation.py", ROOT / "scripts/build_drawer_scene.py",
                                    ROOT / "assets/generated/mise_drawer_contact.xml",
                                    ROOT / "configs/contact_scene.yaml", ROOT / "assets/generated/mise_contact.xml",
                                    ROOT / "configs/full_evaluator.yaml", ROOT / "assets/generated/mise_full_task.xml",
                                    ROOT / "scripts/build_scene.py", ROOT / "scripts/build_contact_scene.py",
                                    ROOT / "scripts/build_full_scene.py",
                                    ROOT / "vendor/SO-ARM100/Simulation/SO101/so101_new_calib.xml"] if path.exists()}
            config = request.model_dump()
            run = {"schema_version": "mise.run.v1", "id": run_id, "status": "queued", **config,
                   "created_at": utc_now(), "hardware": self.hardware, "scope": scope,
                   "plan": asdict(plan), "active_step": None, "phase": "queued",
                   "config_hash": digest({"request": config, "sources": sources}), "sources_sha256": sources,
                   "summary": None, "last_sequence": 0, "last_observation_at": None,
                   "checkpoint_hash": None, "disclosure": disclosure,
                   "artifacts": {name: f"/api/runs/{run_id}/artifacts/{filename}"
                                 for name, filename in {"trace": "trace.jsonl", "summary": "summary.json",
                                                        "manifest": "manifest.json", "timing": "timing.csv"}.items()}}
            recorder = RunRecorder(self.directory(run_id), run)
            recorder.event("run_created", configuration=config, disclosure=disclosure, plan=asdict(plan))
            self.active_id = run_id
            self.controls = self.context.Queue(maxsize=8)
            if self.launch_workers:
                self.process = self.context.Process(target=run_worker,
                    args=(str(self.directory(run_id)), self.display, self.controls), daemon=True)
                try:
                    self.process.start()
                except Exception as exc:
                    self.process = None
                    reason = f"Worker could not start ({type(exc).__name__}: {exc})."
                    recorder.finish("failed", _interrupted_summary(run, reason))
                    raise HTTPException(503, reason) from exc
            return run

    def control(self, run_id: str, action: str) -> dict[str, Any]:
        if self.read_only:
            raise HTTPException(403, "Replay mode is read-only")
        run = self.get(run_id)
        if run_id != self.active_id or run["status"] in TERMINAL or self.controls is None:
            raise HTTPException(409, "Run is not active")
        if action == "resume" and run["status"] != "paused":
            raise HTTPException(409, "Only a paused run can resume")
        if action == "pause" and run["status"] == "paused":
            raise HTTPException(409, "Run is already paused")
        try:
            self.controls.put_nowait(action)
        except queue.Full:
            raise HTTPException(429, "Control queue is full") from None
        return {"accepted": True, "action": action, "run_id": run_id}

    def close(self) -> None:
        if self.process is not None and self.process.is_alive():
            try:
                self.controls.put_nowait("stop")
            except queue.Full:
                pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            if self.active_id:
                self.get(self.active_id)  # Persist failure if forced termination prevented finalization.
        self.closing.set()
        self.thread.join(timeout=1)
        self.display.close()
        if self.controls is not None:
            self.controls.close()
        if self._lease is not None:
            self._lease.close()


def create_app(*, runs_dir: Path | None = None, read_only: bool = False,
               launch_workers: bool = True, web_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = RunManager(runs_dir or ROOT / "artifacts/runs", read_only=read_only,
                                      launch_workers=launch_workers)
        yield
        app.state.manager.close()

    app = FastAPI(title="MISE Robotics Console", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def local_operator(request: Request, call_next):
        # Local single-origin operation, including protection from cross-site POSTs.
        origin = request.headers.get("origin")
        if request.method == "POST" and origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != request.headers.get("host"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Operator commands must come from the same origin"}, 403)
        return await call_next(request)

    @app.get("/api/health")
    def health():
        from .skills import SUPPORTED_CONTACT_COMMANDS
        manager = app.state.manager
        return {"status": "ok", "mode": "replay" if read_only else "local",
                "hardware": manager.hardware,
                "capabilities": {"live_control": not read_only, "controller": "contact_expert",
                                 "controllers": ["contact_expert", "scripted_drawer"],
                                 "supported_commands": {"contact_expert": list(SUPPORTED_CONTACT_COMMANDS),
                                                        "scripted_drawer": ["Open the drawer with arm A."]},
                                 "contact_expert": True,
                                 "bounded_recovery": True,
                                 "adaptive_recovery_memory": True,
                                 "recovery_modes": ["none", "blind_retry", "adaptive"],
                                 "learned_skills": False, "visual_monitor": True,
                                 "full_task": True, "intel_verified": False},
                "disclosure": CONTACT_NOTE}

    @app.post("/api/plans")
    def preview_plan(request: PlanRequest):
        try:
            plan = RuleBasedPlanner().plan(request.command)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        from .skills import validate_contact_plan
        from .full_task import is_full_plan
        try:
            validate_contact_plan(plan)
            executable, reason = True, FULL_NOTE if is_full_plan(plan) else CONTACT_NOTE
        except ValueError as exc:
            executable, reason = False, str(exc)
        return {"plan": asdict(plan), "executable": executable, "controller": "contact_expert",
                "note": reason}

    @app.get("/api/runs")
    def list_runs():
        return {"runs": app.state.manager.list()}

    @app.post("/api/runs", status_code=201)
    def start_run(request: RunRequest):
        return app.state.manager.create(request)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return app.state.manager.get(run_id)

    @app.post("/api/runs/{run_id}/control", status_code=202)
    def control_run(run_id: str, request: ControlRequest):
        return app.state.manager.control(run_id, request.action)

    @app.get("/api/runs/{run_id}/trace")
    def trace(run_id: str):
        app.state.manager.get(run_id)
        return read_trace(app.state.manager.directory(run_id))

    @app.websocket("/api/runs/{run_id}/events")
    async def events(websocket: WebSocket, run_id: str):
        manager = app.state.manager
        try:
            run = manager.get(run_id)
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        sequence = run["last_sequence"]
        try:
            await websocket.send_json({"type": "snapshot", "run": run, "last_sequence": sequence})
            while True:
                for event in read_trace(manager.directory(run_id), after=sequence):
                    await websocket.send_json(event)
                    sequence = event["sequence"]
                await websocket.send_json({"type": "heartbeat", "timestamp": utc_now(),
                                           "last_sequence": sequence})
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.2)
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.get("/api/runs/{run_id}/camera/{camera}")
    async def camera_stream(run_id: str, camera: str, request: Request):
        manager = app.state.manager
        manager.get(run_id)
        if camera not in CAMERAS:
            raise HTTPException(404, "Unknown camera")
        async def stream():
            last = None
            while not await request.is_disconnected():
                item = manager.frames.get(run_id)
                camera_timestamp = item.get("camera_timestamps", {}).get(camera, item["timestamp"]) if item else None
                if item and camera_timestamp != last:
                    last = camera_timestamp
                    jpeg = item["images"].get(camera)
                    if jpeg is None:
                        return
                    header = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n"
                              f"X-Timestamp: {last}\r\nX-Simulation-Time: {item['simulation_time']}\r\n\r\n")
                    yield header.encode() + jpeg + b"\r\n"
                if manager.get(run_id)["status"] in TERMINAL:
                    break
                await asyncio.sleep(0.1)
        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/runs/{run_id}/frame/{camera}")
    def camera_frame(run_id: str, camera: str):
        manager = app.state.manager
        run = manager.get(run_id)
        path = manager.directory(run_id) / f"{camera}.jpg"
        if camera not in CAMERAS or not path.exists():
            raise HTTPException(404, "No frame available")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store",
                            "X-Timestamp": run.get("camera_observed_at", {}).get(camera,
                                           run.get("last_observation_at") or "unavailable")})

    @app.get("/api/runs/{run_id}/artifacts/{filename}")
    def artifact(run_id: str, filename: str):
        manager = app.state.manager
        manager.get(run_id)
        allowed = {"trace.jsonl", "run.json", "summary.json", "manifest.json", "timing.csv"}
        allowed.update(f"{camera}.mp4" for camera in CAMERAS)
        path = manager.directory(run_id) / filename
        if filename not in allowed or not path.is_file():
            raise HTTPException(404, "Artifact is not available")
        return FileResponse(path, filename=filename,
                            media_type="video/mp4" if filename.endswith(".mp4") else None)

    @app.get("/api/evaluations")
    def evaluations():
        from .evaluation import wilson_interval
        seeds = yaml.safe_load((ROOT / "eval/seeds.yaml").read_text())["required"]
        runs = app.state.manager.list()
        evaluated = [run for run in runs if isinstance(run.get("summary"), dict)
                     and run["summary"].get("scope") == "full_task"
                     and isinstance(run["summary"].get("full_task_success"), bool)]
        successes = sum(run["summary"]["full_task_success"] for run in evaluated)
        return {"required_seeds": seeds, "runs": runs,
                "summary": {"successes": successes, "total": len(evaluated),
                            "wilson95": wilson_interval(successes, len(evaluated))},
                "note": "Only independently evaluated full-task runs enter this total; contact skills and fixtures retain separate scopes."}

    @app.get("/api/benchmarks")
    def benchmarks():
        path = ROOT / "artifacts/benchmarks/latest.json"
        if not path.exists():
            return {"records": [], "hardware": app.state.manager.hardware,
                    "note": "No model benchmarks recorded. Run `make bench` after exporting the policy."}
        try:
            report = json.loads(path.read_text())
            records = report.get("records")
            if not isinstance(records, list):
                raise ValueError("records must be a list")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(500, f"Benchmark artifact is invalid: {exc}") from exc
        return {**report, "csv": "/api/benchmark-artifacts/latest.csv"}

    @app.get("/api/benchmark-artifacts/latest.csv")
    def benchmark_csv():
        path = ROOT / "artifacts/benchmarks/latest.csv"
        if not path.is_file():
            raise HTTPException(404, "Benchmark CSV is not available")
        return FileResponse(path, filename="mise-openvino-benchmark.csv", media_type="text/csv")

    @app.get("/api/recovery-comparison")
    def recovery_comparison():
        path = ROOT / "artifacts/evaluations/recovery_matched_seeds.json"
        if not path.exists():
            return {"available": False, "variants": {},
                    "note": "Run `make eval-recovery` to create the matched-seed comparison."}
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(500, f"Recovery comparison artifact is invalid: {exc}") from exc
        return {"available": True, **report}

    frontend = web_dir or ROOT / "web/dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="console")
    else:
        @app.get("/")
        def no_frontend():
            return {"message": "Build the console with: cd web && npm ci && npm run build", "api": "/docs"}
    return app
