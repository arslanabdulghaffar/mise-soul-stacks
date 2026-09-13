import unittest
import tempfile
from pathlib import Path

import numpy as np

from mise.full_task import FullTaskController, FullTaskEvaluator, create_full_env
from mise.planner import RuleBasedPlanner
from mise.recovery_memory import RecoveryMemory


class FullTaskTests(unittest.TestCase):
    def test_camera_grounded_physical_sequence_satisfies_every_predicate(self):
        env = create_full_env(seed=1001)
        temporary = tempfile.TemporaryDirectory()
        try:
            plan = RuleBasedPlanner().plan("Set the table.")
            memory = RecoveryMemory(Path(temporary.name) / "recovery.sqlite3")
            controller = FullTaskController(env, plan, recovery_memory=memory,
                                            episode_id="physical-seed-1001")
            evaluator = FullTaskEvaluator(env)
            addresses = []
            for name in ("plate", "fork", "spoon", "mug"):
                start = env.model.joint(f"{name}_free").qposadr[0]
                addresses.extend(range(start, start + 7))

            for _ in range(7000):
                before = env.data.qpos[addresses].copy()
                targets = controller.advance()
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
            self.assertEqual(controller.evidence["recovery"]["mode"], "adaptive")
            self.assertEqual(len(controller.evidence["recovery"]["candidate_registry"]), 2)
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
