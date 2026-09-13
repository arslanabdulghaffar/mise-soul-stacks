import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mise.recovery_memory import AdaptiveRecoverySupervisor, RecoveryContext, RecoveryMemory
from mise.supervisor import MonitorEvidence, RecoveryCandidate
from mise.types import RecoverabilityVerdict as Verdict


class RecoveryMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'history.sqlite3'
        self.memory = RecoveryMemory(self.path)
        self.context = RecoveryContext('pick_place', 'mug', 'lift', 'not_retained',
                                       'upright_pickup_region', 'scene-v1', 'policy-v1')
        self.first = RecoveryCandidate('regrasp_front', Verdict.RETRY, 'B', .9, 4,
                                       True, True, True, True, 10)
        self.other = replace(self.first, name='regrasp_side', success_rate=.8, duration_s=5)
        self.evidence = MonitorEvidence(failure_detected=True, stable_for_replan=True)

    def record(self, candidate=None, outcome='failure', attempt='attempt1', episode='episode1'):
        return self.memory.record(self.context, candidate or self.first,
                                  attempt_id=attempt, episode_id=episode, outcome=outcome,
                                  duration_s=4, observation_ref='trace.jsonl#visual-check-7')

    def test_failures_change_next_episode_choice_and_survive_restart(self):
        supervisor = AdaptiveRecoverySupervisor(self.memory)
        supervisor.decide_with_memory(1, self.evidence, (self.first, self.other),
                                      context=self.context, episode_id='before')
        self.assertEqual(supervisor.selected.name, 'regrasp_front')
        self.record()
        self.record(attempt='attempt2', episode='episode2')
        restarted = AdaptiveRecoverySupervisor(RecoveryMemory(self.path))
        decision = restarted.decide_with_memory(1, self.evidence, (self.first, self.other),
                                                context=self.context, episode_id='episode3')
        self.assertEqual(decision.verdict, Verdict.RETRY)
        self.assertEqual(restarted.selected.name, 'regrasp_side')

    def test_no_identical_retry_in_same_episode_even_below_threshold_disabled(self):
        self.record()
        supervisor = AdaptiveRecoverySupervisor(self.memory, success_threshold=.1)
        decision = supervisor.decide_with_memory(1, self.evidence, (self.first,),
                                                context=self.context, episode_id='episode1')
        self.assertEqual(decision.verdict, Verdict.STOP)
        self.assertIsNone(supervisor.selected)

    def test_history_cannot_relax_required_arm_or_register_actions(self):
        for index in range(5):
            self.record(outcome='success', attempt=f'success{index}', episode=f'run{index}')
        supervisor = AdaptiveRecoverySupervisor(self.memory)
        decision = supervisor.decide_with_memory(1, self.evidence, (self.first,),
                                                context=self.context, episode_id='new', required_arm='A')
        self.assertEqual(decision.verdict, Verdict.STOP)
        untested = replace(self.first, registered=False, evidence_count=0)
        self.assertEqual(self.memory.estimate(self.context, untested), untested)
        with self.assertRaises(ValueError):
            self.record(untested, attempt='invalid')

    def test_unknown_outcomes_and_other_contexts_do_not_change_estimate(self):
        self.record(outcome='unknown')
        self.assertEqual(self.memory.estimate(self.context, self.first), self.first)
        self.record(attempt='failed')
        for context in (replace(self.context, policy_version='new'),
                        replace(self.context, scene_version='new'),
                        replace(self.context, state_bucket='different_pose')):
            self.assertEqual(self.memory.estimate(context, self.first), self.first)

    def test_duplicate_delivery_is_idempotent_and_conflicts_are_rejected(self):
        self.assertTrue(self.record())
        self.assertFalse(self.record())
        self.assertEqual(self.memory.estimate(self.context, self.first).evidence_count, 11)
        with self.assertRaises(ValueError):
            self.record(outcome='success')

    def test_history_preserves_budget_feasibility_and_goal_constraints(self):
        self.record(outcome='success')
        for candidate in (replace(self.first, feasible=False),
                          replace(self.first, preserves_completed_goals=False),
                          replace(self.first, preserves_constraints=False)):
            supervisor = AdaptiveRecoverySupervisor(self.memory)
            decision = supervisor.decide_with_memory(1, self.evidence, (candidate,),
                                                    context=self.context, episode_id='new')
            self.assertEqual(decision.verdict, Verdict.STOP)
        supervisor = AdaptiveRecoverySupervisor(self.memory, max_attempts=0)
        self.assertEqual(supervisor.decide_with_memory(1, self.evidence, (self.first,),
                         context=self.context, episode_id='new').verdict, Verdict.STOP)

    def test_summary_reports_persisted_candidate_outcomes(self):
        self.assertTrue(self.record(outcome='success'))
        summary = self.memory.summary()
        self.assertEqual((summary['total_outcomes'], summary['successes'], summary['failures']), (1, 1, 0))
        self.assertEqual(summary['candidates'][0]['candidate'], self.first.name)
        self.assertEqual(summary['candidates'][0]['mean_duration_s'], 4.0)


if __name__ == '__main__':
    unittest.main()
