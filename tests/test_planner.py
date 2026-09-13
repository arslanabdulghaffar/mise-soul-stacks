from __future__ import annotations

import unittest

from mise.planner import DEFAULT_COMMAND, PlanValidationError, RuleBasedPlanner, parse_plan


class PlannerTests(unittest.TestCase):
    def test_valid_json_graph_is_parsed(self) -> None:
        plan = parse_plan('{"steps":[{"id":1,"skill":"open_drawer","arm":"A","zone":"drawer"},{"id":2,"skill":"pick","arm":"B","needs":[],"zone":"right"}]}')
        self.assertEqual([step.id for step in plan.steps], [1, 2])
        self.assertEqual(plan.source, "vlm")

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaises(PlanValidationError):
            parse_plan('{"steps":[{"id":1,"skill":"pick","arm":"A","needs":[2]},{"id":2,"skill":"pick","arm":"B","needs":[1]}]}')

    def test_table_setting_preserves_handoff_and_order(self) -> None:
        plan = RuleBasedPlanner().plan(DEFAULT_COMMAND)
        self.assertEqual(plan.steps[0].skill, "open_drawer")
        handoff = next(step for step in plan.steps if step.skill == "handoff")
        self.assertEqual((handoff.arm, handoff.donor, handoff.receiver, handoff.object), ("both", "A", "B", "spoon"))
        self.assertEqual(plan.steps[-1].object, "mug")
        for first, second in zip(plan.steps, plan.steps[1:]):
            self.assertIn(first.id, second.needs)

    def test_goal_only_command_exposes_independent_tasks(self) -> None:
        plan = RuleBasedPlanner().plan("set the table")
        ready_objects = {step.object for step in plan.steps if not step.needs}
        self.assertEqual(ready_objects, {"drawer", "plate", "mug"})
        handoff = next(step for step in plan.steps if step.skill == "handoff")
        pickup = next(step for step in plan.steps if step.object == "spoon" and step.skill == "pick")
        self.assertEqual(handoff.needs, (pickup.id,))

    def test_unsupported_actions_are_not_silently_dropped(self) -> None:
        for command in ("Open drawer, pour water", "pick the knife", "open drawer before placing the plate", "do not open drawer"):
            with self.subTest(command=command), self.assertRaises(PlanValidationError):
                RuleBasedPlanner().plan(command)

    def test_requested_arm_is_preserved(self) -> None:
        plan = RuleBasedPlanner().plan("Open the drawer with arm B, retrieve the fork with arm B, place it on the left with arm B")
        self.assertTrue(all(step.arm == "B" for step in plan.steps))



if __name__ == "__main__":
    unittest.main()
