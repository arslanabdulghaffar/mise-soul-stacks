import unittest
from mise.planner import parse_plan
from mise.planner_teacher import DEFAULT_COMMAND, SceneAwarePlannerTeacher, SceneFacts

class PlannerTeacherTests(unittest.TestCase):
    def test_scene_side_does_not_override_explicit_arms(self):
        parsed = parse_plan(SceneAwarePlannerTeacher().plan(DEFAULT_COMMAND, SceneFacts(False, "B", "A")))
        self.assertEqual(next(s for s in parsed.steps if s.object == "plate").arm, "A")
        self.assertEqual(parsed.steps[-1].arm, "B")
        self.assertIn("handoff", [s.skill for s in parsed.steps])

    def test_open_drawer_only_skips_implicit_prerequisite(self):
        teacher = SceneAwarePlannerTeacher()
        facts = SceneFacts(True, "A", "B")
        self.assertEqual(teacher.plan(DEFAULT_COMMAND, facts)["steps"][0]["skill"], "open_drawer")
        self.assertNotIn("open_drawer", [s["skill"] for s in teacher.plan("retrieve the fork with arm A", facts)["steps"]])
