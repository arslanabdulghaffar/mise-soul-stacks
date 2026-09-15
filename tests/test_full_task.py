import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mise.full_task import FullTaskController, FullTaskEvaluator, create_full_env, validate_full_plan
from mise.planner import DEFAULT_COMMAND, RuleBasedPlanner, validate_plan_graph
from mise.recovery_memory import RecoveryMemory


class FullTaskTests(unittest.TestCase):
    def test_full_plan_accepts_registered_parallel_and_sequential_dependencies(self):
        for command in ("Set the table.", DEFAULT_COMMAND):
            with self.subTest(command=command):
                validate_full_plan(RuleBasedPlanner().plan(command))

        plan = RuleBasedPlanner().plan("Set the table.")
        # A later listed mug is already completed in the first parallel group.
        plan.steps[1] = replace(plan.steps[1], needs=(plan.steps[-1].id,))
        # VLM plans may use different positive IDs.
        plan.steps = [replace(step, id=step.id + 20,
                              needs=tuple(dependency + 20 for dependency in step.needs))
                      for step in plan.steps]
        validate_full_plan(plan)

    def test_full_plan_rejects_acyclic_dependencies_the_executor_would_ignore(self):
        for dependent in ("drawer", "plate", "fork"):
            with self.subTest(dependent=dependent):
                plan = RuleBasedPlanner().plan("Set the table.")
                mug = plan.steps[-1]
                if dependent != "drawer":
                    # A dependent mug is no longer part of the initial group;
                    # the specialized executor would leave it until last.
                    plan.steps[-1] = replace(mug, needs=(plan.steps[0].id,))
                index = next(index for index, step in enumerate(plan.steps)
                             if step.object == dependent)
                plan.steps[index] = replace(plan.steps[index], needs=(mug.id,))
                validate_plan_graph(plan)
                with self.assertRaisesRegex(ValueError, "cannot honor.*dependencies"):
                    validate_full_plan(plan)

    def test_interrupted_recovery_is_retained_without_scoring_an_unobserved_outcome(self):
        env = create_full_env(seed=1001)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for interruption in ("timeout", "occluded"):
                    with self.subTest(interruption=interruption):
                        memory = RecoveryMemory(Path(directory) / f"{interruption}.sqlite3")
                        controller = FullTaskController(
                            env, RuleBasedPlanner().plan("Set the table."),
                            recovery_memory=memory, episode_id=interruption,
                        )
                        controller.active_step = controller.plan.steps[4]
                        controller._handle_spoon_failure(np.asarray([.3, 0]))
                        candidate, _, context = controller._pending_recovery
                        initial_estimate = memory.estimate(context, candidate)

                        if interruption == "timeout":
                            controller._step_started = float(env.data.time) - controller.active_step.timeout_s - 1
                            controller.advance()
                        else:
                            controller._segment_index = len(controller._segments)
                            with patch.object(env, "render", side_effect=ValueError("Spoon is occluded")):
                                controller.advance()

                        self.assertIsNotNone(controller.failed_reason)
                        self.assertIsNone(controller._pending_recovery)
                        self.assertEqual(controller.evidence["recoveries"][-1]["outcome"], "unknown")
                        self.assertEqual(memory.summary()["unknown"], 1)
                        self.assertEqual(memory.estimate(context, candidate), initial_estimate)
                        self.assertEqual(
                            [event["type"] for event in controller.events if "recovery" in event["type"]],
                            ["recovery_selected", "recovery_unverified"],
                        )
                        controller.advance()
                        self.assertEqual(memory.summary()["total_outcomes"], 1)
        finally:
            env.close()

    def test_camera_grounded_physical_sequence_satisfies_every_predicate(self):
        env = create_full_env(seed=1001)
        temporary = tempfile.TemporaryDirectory()
        try:
            plan = RuleBasedPlanner().plan("Set the table.")
            memory = RecoveryMemory(Path(temporary.name) / "recovery.sqlite3")
            controller = FullTaskController(env, plan, recovery_memory=memory,
                                            episode_id="physical-seed-1001")
            evaluator = FullTaskEvaluator(env)
            phases = []
            addresses = []
            for name in ("plate", "fork", "spoon", "mug"):
                start = env.model.joint(f"{name}_free").qposadr[0]
                addresses.extend(range(start, start + 7))

            for _ in range(7000):
                before = env.data.qpos[addresses].copy()
                targets = controller.advance()
                if not phases or controller.phase != phases[-1]:
                    phases.append(controller.phase)
                np.testing.assert_array_equal(
                    before, env.data.qpos[addresses],
                    "The controller must never write free-object state.",
                )
                if controller.done or controller.failed_reason:
                    break
                env.step(targets, observe=False)
                evaluator.update(env.data)

            self.assertIsNone(controller.failed_reason)
            self.assertTrue(controller.done)
            self.assertEqual(controller.completed_steps, [1, 7, 2, 3, 4, 5, 6])
            self.assertIn("parallel_group_completed", [event["type"] for event in controller.events])
            self.assertTrue(evaluator.success, evaluator.evidence)
            self.assertTrue(all(evaluator.snapshot.objects_in_goal.values()))
            self.assertLessEqual(evaluator.evidence["objects"]["spoon"]["axis_error_degrees"], 20)
            self.assertLessEqual(evaluator.evidence["objects"]["fork"]["axis_error_degrees"], 20)
            self.assertTrue(evaluator.snapshot.handoff_complete)
            self.assertEqual(evaluator.snapshot.forbidden_collision_count, 0)
            self.assertEqual(evaluator.snapshot.violations, ())
            self.assertEqual(evaluator.evidence["utensils_retrieved_from_drawer"], ["fork", "spoon"])
            self.assertEqual(controller.recovery_attempts, 1)
            self.assertLess(phases.index("retract_from_spoon"), phases.index("recovery_approach_spoon"))
            self.assertLess(phases.index("recovery_retract_spoon"), phases.index("parallel_fork_and_arm_B_park"))
            self.assertNotIn("park_after_spoon", phases)
            self.assertIn("parallel_cleanup_started", [event["type"] for event in controller.events])
            self.assertEqual(controller.evidence["recovery"]["mode"], "adaptive")
            self.assertEqual(len(controller.evidence["recovery"]["candidate_registry"]), 3)
            self.assertEqual(memory.summary()["successes"], 1)
            self.assertEqual(
                [event["type"] for event in controller.events if "recovery" in event["type"]],
                ["recovery_selected", "recovery_succeeded"],
            )
            self.assertFalse(controller.evidence["object_state_used_for_targets"])
            self.assertFalse(controller.evidence["object_attachment_used"])
            self.assertEqual(env.model.nu, 12)
            self.assertEqual(env.model.neq, 0)
        finally:
            env.close()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
