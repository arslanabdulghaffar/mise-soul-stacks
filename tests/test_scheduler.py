from __future__ import annotations

import unittest

from mise.scheduler import Scheduler
from mise.types import Plan, SkillStep


class SchedulerTests(unittest.TestCase):
    def test_different_arms_and_zones_can_start_together(self) -> None:
        scheduler = Scheduler(Plan([SkillStep(1, "open_drawer", "A", zone="drawer"), SkillStep(2, "pick", "B", zone="right")]), parallel=True, geometry_clear=lambda step, others: True)
        self.assertEqual([step.id for step in scheduler.start_ready()], [1, 2])

    def test_two_arm_skill_waits_for_all_arms(self) -> None:
        scheduler = Scheduler(Plan([SkillStep(1, "pick", "A", zone="left"), SkillStep(2, "handoff", "both", zone="middle", donor="A", receiver="B")]))
        self.assertEqual([step.id for step in scheduler.start_ready()], [1])


if __name__ == "__main__":
    unittest.main()

