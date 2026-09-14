from __future__ import annotations

import hashlib
import json
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from fastapi.testclient import TestClient

from mise.server import create_app
from mise.telemetry import RunRecorder, read_trace
from mise.worker import run_worker


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.app = create_app(runs_dir=self.root, launch_workers=False)
        self.client = TestClient(self.app).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_second_writer_cannot_relabel_an_active_run(self):
        run = self.create()
        with self.assertRaisesRegex(RuntimeError, 'Another runtime owns'):
            with TestClient(create_app(runs_dir=self.root, launch_workers=False)):
                pass
        self.assertEqual(self.client.get(f"/api/runs/{run['id']}").json()['status'], 'queued')
        with TestClient(create_app(runs_dir=self.root, read_only=True)) as replay:
            self.assertEqual(replay.get(f"/api/runs/{run['id']}").json()['status'], 'queued')

    def create(self, **kwargs):
        response = self.client.post("/api/runs", json={"command": "Open the drawer with arm A.",
                                                       "controller": "scripted_drawer", **kwargs})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_single_operator_and_executable_contract(self):
        self.assertEqual(self.client.post("/api/runs", json={"command": "Pour water"}).status_code, 422)
        self.assertEqual(self.client.post("/api/runs", json={"command": "Open the drawer with arm B"}).status_code, 422)
        self.assertEqual(self.client.post("/api/runs", json={"seed": True}).status_code, 422)
        self.assertEqual(self.client.post("/api/runs", json={"preset": "arbitrary"}).status_code, 422)
        self.assertEqual(self.client.post("/api/runs", json={"recovery_mode": "unbounded"}).status_code, 422)
        run = self.create(seed=1001)
        self.assertEqual(self.client.post("/api/runs", json={}).status_code, 409)
        self.assertIsNone(run["summary"])
        self.assertIsNone(run["checkpoint_hash"])
        self.assertEqual(run["recovery_mode"], "adaptive")
        self.assertEqual(len(run["config_hash"]), 64)

    def test_contact_command_records_actual_instruction_and_scope(self):
        response = self.client.post("/api/runs", json={})
        self.assertEqual(response.status_code, 201, response.text)
        run = response.json()
        self.assertEqual(run["controller"], "contact_expert")
        self.assertEqual(run["scope"], "contact_skill")
        step = run["plan"]["steps"][0]
        self.assertEqual((step["skill"], step["object"], step["arm"]), ("pick_place", "mug", "B"))
        self.assertEqual(step["target"], "table_upper_right")
        self.assertIsNone(run["active_step"])
        self.assertIn("RGB", run["disclosure"])
        self.assertIsNone(run["checkpoint_hash"])
        for name in ("scripts/build_contact_scene.py", "src/mise/kinematics.py", "src/mise/contact_vision.py",
                     "src/mise/manipulation.py", "src/mise/worker.py", "configs/evaluator.yaml"):
            self.assertIn(name, run["sources_sha256"])

    def test_contact_plan_is_checked_before_worker_launch(self):
        with patch("mise.manipulation.validate_contact_plan", side_effect=ValueError("Unvalidated hand-off")):
            response = self.client.post("/api/runs", json={})
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unvalidated hand-off", response.json()["detail"])
        self.assertEqual(self.client.get("/api/runs").json()["runs"], [])

    def test_health_and_preview_identify_executable_contact_baseline(self):
        health = self.client.get("/api/health").json()
        self.assertEqual(health["capabilities"]["controller"], "contact_expert")
        self.assertIn("scripted_drawer", health["capabilities"]["controllers"])
        self.assertFalse(health["capabilities"]["learned_skills"])
        self.assertTrue(health["capabilities"]["visual_monitor"])
        self.assertTrue(health["capabilities"]["adaptive_recovery_memory"])
        self.assertEqual(health["capabilities"]["recovery_modes"], ["none", "blind_retry", "adaptive"])
        self.assertTrue(health["capabilities"]["full_task"])
        command = health["capabilities"]["supported_commands"]["contact_expert"][0]
        preview = self.client.post("/api/plans", json={"command": command})
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertTrue(preview.json()["executable"])
        comparison = self.client.get("/api/recovery-comparison")
        self.assertEqual(comparison.status_code, 200)
        self.assertIn("available", comparison.json())
        self.assertIn("variants", comparison.json())

    def test_full_command_records_complete_scope_and_sources(self):
        response = self.client.post("/api/runs", json={"command": "Set the table."})
        self.assertEqual(response.status_code, 201, response.text)
        run = response.json()
        self.assertEqual(run["scope"], "full_task")
        self.assertEqual(len(run["plan"]["steps"]), 7)
        self.assertIn("seven camera-grounded contact steps", run["disclosure"])
        for name in ("scripts/build_full_scene.py", "src/mise/full_task.py",
                     "src/mise/full_vision.py", "configs/full_evaluator.yaml"):
            self.assertIn(name, run["sources_sha256"])

    def test_only_full_scope_enters_evaluation_total(self):
        run = self.client.post("/api/runs", json={"command": "Set the table."}).json()
        recorder = RunRecorder(self.root / run["id"], run)
        recorder.finish("completed", {"scope": "full_task", "full_task_success": True,
                                       "completed_steps": list(range(1, 8))})
        evidence = self.client.get("/api/evaluations").json()
        self.assertEqual(evidence["summary"]["successes"], 1)
        self.assertEqual(evidence["summary"]["total"], 1)
        self.assertEqual(len(evidence["summary"]["wilson95"]), 2)

    def test_snapshot_reconnect_and_trace_are_sequenced(self):
        run = self.create()
        with self.client.websocket_connect(f"/api/runs/{run['id']}/events") as socket:
            snapshot = socket.receive_json()
            self.assertEqual(snapshot["type"], "snapshot")
            self.assertEqual(snapshot["last_sequence"], 1)
            self.assertEqual(snapshot["run"]["id"], run["id"])
        events = self.client.get(f"/api/runs/{run['id']}/trace").json()
        self.assertEqual(events[0]["sequence"], 1)
        self.assertEqual(events[0]["config_hash"], run["config_hash"])
        self.assertEqual(events[0]["monitor"]["confidence"], None)

    def test_camera_quality_is_validated_and_bound_to_run(self):
        self.assertEqual(self.client.post("/api/runs", json={"view_quality": "unbounded"}).status_code, 422)
        run = self.create(view_quality="detail")
        self.assertEqual(run["view_quality"], "detail")
        self.assertIn("src/mise/presentation.py", run["sources_sha256"])

    def test_controls_unknown_runs_and_artifacts(self):
        run = self.create()
        base = f"/api/runs/{run['id']}"
        self.assertEqual(self.client.post(base + "/control", json={"action": "stop"}).status_code, 202)
        self.assertEqual(self.client.post(base + "/control", json={"action": "reset"}).status_code, 422)
        self.assertEqual(self.client.post(base + "/control", json={"action": "resume"}).status_code, 409)
        self.assertEqual(self.client.get(base + "/artifacts/trace.jsonl").status_code, 200)
        self.assertEqual(self.client.get(base + "/artifacts/worker.py").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/missing").status_code, 404)
        self.assertEqual(self.client.get(base + "/camera/missing").status_code, 404)

    def test_spawn_failure_is_finalized_and_does_not_block_next_run(self):
        manager = self.app.state.manager
        manager.launch_workers = True
        process = Mock()
        process.start.side_effect = OSError("process limit")
        with patch.object(manager.context, "Process", return_value=process):
            response = self.client.post("/api/runs", json={"command": "Open the drawer with arm A.",
                                                          "controller": "scripted_drawer"})
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(manager.process)
        failed = self.client.get("/api/runs").json()["runs"][0]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("process limit", failed["summary"]["reason"])
        manager.launch_workers = False
        self.assertEqual(self.create()["status"], "queued")

    def test_unexpected_exit_retains_partial_contact_evidence(self):
        run = self.create()
        # Restore a partial persisted contact run without executing its controller.
        run.update(controller="contact_expert", scope="contact_skill", completed_steps=[1],
                   contact_evidence={"retention_observed": True}, active_step={"id": 2, "skill": "pick_place", "arm": "B"})
        RunRecorder(self.root / run["id"], run)
        process = Mock(exitcode=7)
        process.is_alive.return_value = False
        self.app.state.manager.process = process
        failed = self.client.get(f"/api/runs/{run['id']}").json()
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["phase"], "failed")
        self.assertIsNone(failed["active_step"])
        self.assertEqual(failed["summary"]["completed_steps"], [1])
        self.assertEqual(failed["summary"]["contact_evidence"], {"retention_observed": True})
        self.assertFalse(failed["summary"]["contact_skill_success"])
        self.assertIsNone(failed["summary"]["full_task_success"])

    def test_fixture_does_not_count_as_full_task_evidence(self):
        run = self.create()
        recorder = RunRecorder(self.root / run["id"], run)
        recorder.finish("completed", {"full_task_success": None, "fixture_drawer_open": True})
        evidence = self.client.get("/api/evaluations").json()
        self.assertEqual(evidence["summary"]["total"], 0)
        self.assertEqual(len(evidence["required_seeds"]), 10)
        self.assertEqual(len(evidence["runs"]), 1)
        manifest = json.loads((recorder.directory / "manifest.json").read_text())
        for name, expected in manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((recorder.directory / name).read_bytes()).hexdigest(), expected)
        self.assertEqual([e["sequence"] for e in read_trace(recorder.directory)], [1, 2])

    def test_contact_evidence_context_and_manifest_preserve_scope(self):
        run = self.create(command="Place the mug in the upper-right with arm B.", controller="contact_expert")
        recorder = RunRecorder(self.root / run["id"], run)
        recorder.run["active_step"] = run["plan"]["steps"][0]
        recorder.run["phase"] = "grasp"
        event = recorder.event("phase_changed", 1.5)
        self.assertEqual(event["active_skill"], "pick_place")
        self.assertEqual(event["arm_state"], {"A": "hold_target", "B": "contact_expert"})
        self.assertEqual(event["payload"]["phase"], "grasp")
        self.assertEqual(event["payload"]["step_id"], run["plan"]["steps"][0]["id"])
        recorder.finish("completed", {"scope": "contact_skill", "contact_skill_success": True,
                                       "full_task_success": None, "completed_steps": [1]})
        evidence = self.client.get("/api/evaluations").json()
        self.assertEqual(evidence["summary"]["total"], 0)
        manifest = json.loads((recorder.directory / "manifest.json").read_text())
        self.assertEqual(manifest["scope"], "contact_skill")
        self.assertEqual(manifest["disclosure"], run["disclosure"])

    def test_high_rate_action_trace_can_defer_snapshot_checkpoint(self):
        run = self.create(command="Place the mug in the upper-right with arm B.", controller="contact_expert")
        recorder = RunRecorder(self.root / run["id"], run)
        on_disk_before = json.loads((recorder.directory / "run.json").read_text())
        event = recorder.event("action", 0.033, persist_run=False, joint_targets=[0.0] * 12)
        on_disk_after = json.loads((recorder.directory / "run.json").read_text())
        self.assertEqual(on_disk_after["last_sequence"], on_disk_before["last_sequence"])
        self.assertEqual(read_trace(recorder.directory)[-1]["sequence"], event["sequence"])
        recorder.save()
        self.assertEqual(json.loads((recorder.directory / "run.json").read_text())["last_sequence"],
                         event["sequence"])

    def _run_contact_worker_fixture(self, *, physics_success=True, cleanup_error=False):
        run = self.create(command="Place the mug in the upper-right with arm B.", controller="contact_expert")
        step = self.app.state.manager.get(run["id"])["plan"]["steps"][0]
        from mise.types import SkillStep
        active = SkillStep(**step)
        env = Mock()
        env.control_hz = 30
        env.data = SimpleNamespace(time=0.0)
        env.last_randomization = None
        if cleanup_error:
            env.close.side_effect = RuntimeError("renderer cleanup error")
        env.render.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        env.joint_positions.return_value = np.zeros(12)
        env.step.side_effect = lambda *args, **kwargs: setattr(env.data, "time", env.data.time + 1 / 30)
        targets = np.arange(12, dtype=float) / 10
        controller = SimpleNamespace(active_step=active, phase="grasp", completed_steps=[],
                                     evidence={"contact_retained": True}, events=[], done=False, failed_reason=None)
        advances = 0

        def advance():
            nonlocal advances
            advances += 1
            if advances == 2:
                controller.done = True
                controller.phase = "done"
                controller.completed_steps = [active.id]
                controller.events.append({"type": "step_completed", "step_id": active.id})
            return targets

        controller.advance = advance
        snapshot = SimpleNamespace(to_dict=lambda: {"full_task_success": False}, drawer_open=False)
        with patch("mise.manipulation.create_contact_env", return_value=env), \
                patch("mise.manipulation.ManipulationController", return_value=controller), \
                patch("mise.contact_evaluation.ContactSkillEvaluator") as contact_verifier, \
                patch("mise.evaluation.TaskEvaluator") as evaluator, \
                patch("mise.worker.PresentationRenderer") as display_renderer, \
                patch("imageio.v2.get_writer") as videos, patch("mise.worker.time.sleep"):
            display_renderer.return_value.render.side_effect = env.render
            evaluator.return_value.update.return_value = snapshot
            contact_verifier.return_value.success = physics_success
            contact_verifier.return_value.evidence = {"grasp_observed": physics_success, "lift_observed": physics_success}
            run_worker(str(self.root / run["id"]), queue.Queue(), queue.Queue())
        env.set_drawer.assert_not_called()
        self.assertEqual(env.step.call_count, 1, "No stale action may execute after the controller finishes")
        np.testing.assert_array_equal(env.step.call_args.args[0], targets)
        self.assertEqual(videos.call_count, 5)
        recorded = self.client.get(f"/api/runs/{run['id']}").json()
        contact_verifier.return_value.update.assert_called_once_with(env.data)
        actions = [event for event in read_trace(self.root / run["id"]) if event["type"] == "action"]
        self.assertEqual(actions[0]["payload"]["joint_targets"], targets.tolist())
        self.assertEqual(actions[0]["arm_state"]["B"], "contact_expert")
        return recorded, active.id

    def test_worker_executes_contact_targets_and_retains_physical_scope(self):
        recorded, step_id = self._run_contact_worker_fixture()
        self.assertEqual(recorded["status"], "completed")
        self.assertTrue(recorded["summary"]["contact_skill_success"])
        self.assertIsNone(recorded["summary"]["full_task_success"])
        self.assertEqual(recorded["summary"]["completed_steps"], [step_id])
        self.assertTrue(recorded["summary"]["contact_evidence"]["physics_verification"]["lift_observed"])

    def test_camera_completion_cannot_override_failed_physics_verification(self):
        recorded, step_id = self._run_contact_worker_fixture(physics_success=False)
        self.assertEqual(recorded["status"], "failed")
        self.assertFalse(recorded["summary"]["contact_skill_success"])
        self.assertEqual(recorded["summary"]["controller_completed_steps"], [step_id])
        self.assertEqual(recorded["summary"]["completed_steps"], [])
        self.assertIn("without verified grasp", recorded["summary"]["reason"])

    def test_cleanup_error_still_produces_failure_summary(self):
        recorded, _ = self._run_contact_worker_fixture(cleanup_error=True)
        self.assertEqual(recorded["status"], "failed")
        self.assertIn("renderer cleanup error", recorded["summary"]["reason"])
        self.assertIsNone(recorded["summary"]["full_task_success"])

    def test_cross_origin_commands_are_rejected(self):
        response = self.client.post("/api/runs", json={}, headers={"Origin": "https://unrelated.example"})
        self.assertEqual(response.status_code, 403)

    def test_replay_server_is_read_only(self):
        with TestClient(create_app(runs_dir=self.root / "replay", read_only=True)) as client:
            self.assertFalse(client.get("/api/health").json()["capabilities"]["live_control"])
            self.assertEqual(client.post("/api/runs", json={}).status_code, 403)
            self.assertEqual(client.post("/api/runs/aaaaaaaaaaaa/control", json={"action": "stop"}).status_code, 403)


if __name__ == "__main__":
    unittest.main()
