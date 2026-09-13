import unittest
from dataclasses import replace

from mise.scheduler import Scheduler
from mise.supervisor import MonitorEvidence, RecoveryCandidate, RecoverySupervisor
from mise.types import Plan, SkillStep, RecoverabilityVerdict as Verdict


class RuntimeContractTests(unittest.TestCase):
    def candidate(self, **kwargs):
        base = RecoveryCandidate("regrasp", Verdict.RETRY, "A", 0.8, 4.0,
                                 registered=True, feasible=True, preserves_constraints=True,
                                 preserves_completed_goals=True, evidence_count=24)
        return replace(base, **kwargs)

    def test_cheap_forbidden_switch_loses_to_valid_regrasp(self):
        supervisor = RecoverySupervisor()
        grasp = self.candidate()
        switch = self.candidate(name="switch_arm", verdict=Verdict.REPLAN, arm="B", duration_s=1)
        decision = supervisor.decide(1, MonitorEvidence(failure_detected=True, stable_for_replan=True),
                                     (switch, grasp), required_arm="A")
        self.assertEqual(decision.verdict, Verdict.RETRY)
        self.assertEqual(supervisor.selected.name, "regrasp")
        self.assertIsNone(decision.confidence)

    def test_expected_cost_requires_success_threshold(self):
        supervisor = RecoverySupervisor()
        candidates = (self.candidate(name="fast_failure", duration_s=0.01, success_rate=0.1),
                      self.candidate(name="slow", duration_s=10, success_rate=1),
                      self.candidate(name="cheap", duration_s=3, success_rate=0.75))
        supervisor.decide(1, MonitorEvidence(failure_detected=True), candidates)
        self.assertEqual(supervisor.selected.name, "cheap")

    def test_recovery_budget_and_invalid_evidence_stop(self):
        supervisor = RecoverySupervisor()
        evidence = MonitorEvidence(failure_detected=True)
        for _ in range(2):
            self.assertEqual(supervisor.decide(1, evidence, (self.candidate(),)).verdict, Verdict.RETRY)
            supervisor.begin_recovery(1)
        self.assertEqual(supervisor.decide(1, evidence, (self.candidate(),)).verdict, Verdict.STOP)
        self.assertEqual(supervisor.decide(2, MonitorEvidence(fresh=False), (self.candidate(),)).verdict, Verdict.STOP)
        self.assertEqual(supervisor.decide(2, evidence, (self.candidate(evidence_count=0),)).verdict, Verdict.STOP)
        self.assertEqual(supervisor.decide(2, MonitorEvidence(terminal_satisfied=True)).verdict, Verdict.DONE)

    def test_sequential_default_parallel_clearance_and_object_reservation(self):
        plan = Plan([SkillStep(1, "pick", "A", zone="left", object="spoon"),
                     SkillStep(2, "pick", "B", zone="right", object="spoon")])
        self.assertEqual(len(Scheduler(plan).start_ready()), 1)
        with self.assertRaises(ValueError):
            Scheduler(plan, parallel=True)
        self.assertEqual(len(Scheduler(plan, parallel=True, geometry_clear=lambda *_: True).start_ready()), 1)
        different = Plan([plan.steps[0], replace(plan.steps[1], object="mug")])
        self.assertEqual(len(Scheduler(different, parallel=True, geometry_clear=lambda *_: True).start_ready()), 2)
        self.assertEqual(len(Scheduler(different, parallel=True, geometry_clear=lambda *_: False).start_ready()), 0)

    def test_replan_preserves_completed_work_and_requested_arms(self):
        plan = Plan([SkillStep(1, "pick", "A", object="plate"), SkillStep(2, "pick", "B", (1,), object="mug")])
        scheduler = Scheduler(plan)
        scheduler.start_ready()
        with self.assertRaises(ValueError):
            scheduler.replan(plan)
        scheduler.complete(1)
        with self.assertRaises(ValueError):
            scheduler.replan(Plan([replace(plan.steps[0], arm="B"), plan.steps[1]]))
        with self.assertRaises(ValueError):
            scheduler.replan(Plan([plan.steps[0], replace(plan.steps[1], arm="A")]))
        scheduler.replan(plan)
        self.assertEqual(scheduler.completed, {1})
