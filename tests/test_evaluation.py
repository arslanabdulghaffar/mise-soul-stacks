import unittest
import numpy as np

from mise.evaluation import EvaluatorConfig, StablePredicate, TaskEvaluator, wilson_interval
from mise.sim import BimanualTableEnv


class EvaluationTests(unittest.TestCase):
    def test_stability_requires_continuous_observations(self):
        predicate = StablePredicate(.5, .11)
        for t in (0, .1, .2, .3, .4):
            self.assertFalse(predicate.update(t, True))
        self.assertTrue(predicate.update(.5, True))
        self.assertFalse(predicate.update(.6, False))
        self.assertFalse(predicate.update(2, True))
        self.assertFalse(predicate.update(0, True))

    def test_wilson_interval_empty_and_ten_successes(self):
        self.assertIsNone(wilson_interval(0, 0))
        low, high = wilson_interval(10, 10)
        self.assertAlmostEqual(low, .72247, places=4)
        self.assertAlmostEqual(high, 1)

    def test_evidence_counts_and_observation_times_must_be_valid(self):
        for successes, total in ((-1, 10), (11, 10), (0, -1), (True, 10), (1.5, 10)):
            with self.subTest(successes=successes, total=total), self.assertRaises(ValueError):
                wilson_interval(successes, total)
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertAlmostEqual(wilson_interval(0, 10)[1], 1 - wilson_interval(10, 10)[0])
        with self.assertRaises(ValueError):
            StablePredicate(0)
        with self.assertRaises(ValueError):
            StablePredicate(.5).update(float("nan"), True)

    def test_handoff_needs_direct_transfer_and_unsupported_receiver_retention(self):
        # Isolate the temporal hand-off contract from geometric contact detection.
        evaluator = TaskEvaluator.__new__(TaskEvaluator)
        evaluator.config = EvaluatorConfig()
        evaluator._since = {}
        evaluator._handoff_phase = "awaiting_A"
        evaluator._handoff_complete = False
        evaluator._update_handoff({"B"}, False, 0)
        evaluator._update_handoff({"B"}, False, 2)
        self.assertFalse(evaluator._handoff_complete, "Receiver possession alone is not a transfer")
        evaluator._update_handoff({"A"}, False, 3)
        evaluator._update_handoff({"A", "B"}, False, 3.1)
        evaluator._update_handoff({"B"}, False, 3.2)
        evaluator._update_handoff({"B"}, True, 3.8)
        evaluator._update_handoff({"B"}, False, 4.3)
        self.assertFalse(evaluator._handoff_complete, "Table support breaks a direct hand-off")
        evaluator._update_handoff({"A"}, False, 5)
        evaluator._update_handoff({"A", "B"}, False, 5.1)
        evaluator._update_handoff({"B"}, False, 5.2)
        evaluator._update_handoff({"B"}, False, 6.1)
        self.assertFalse(evaluator._handoff_complete)
        evaluator._update_handoff({"B"}, False, 6.2)
        self.assertTrue(evaluator._handoff_complete)

    def test_drawer_fixture_is_never_full_task_success(self):
        env = BimanualTableEnv(seed=1001)
        try:
            for name in ("spoon", "fork"):
                self.assertGreater(env.model.body(name).id, 0)
                self.assertEqual(env.model.joint(name + "_free").type[0], 0)
            evaluator = TaskEvaluator(env.model)
            env.set_drawer(.18)
            for _ in range(60):
                env.step(np.zeros(12), observe=False)
                result = evaluator.update(env.data)
            self.assertTrue(result.drawer_open)
            self.assertFalse(result.full_task_success)
            self.assertFalse(result.handoff_complete)
            self.assertFalse(result.evidence["geometry_validated"])
            self.assertEqual(result.evidence["role"], "privileged_evaluator")
            self.assertFalse(result.evidence["drawer_arm_A_contact_observed"])
            self.assertEqual(result.evidence["utensils_retrieved_from_drawer"], [])
            state_before = [array.copy() for array in (env.data.qpos, env.data.qvel, env.data.ctrl)]
            evaluator.update(env.data)
            for before, after in zip(state_before, (env.data.qpos, env.data.qvel, env.data.ctrl)):
                np.testing.assert_array_equal(before, after)
        finally:
            env.close()

    def test_control_rate_and_nonfinite_actions(self):
        env = BimanualTableEnv(seed=4)
        try:
            with self.assertRaises(ValueError):
                env.step(np.full(12, np.nan), observe=False)
            for _ in range(30):
                env.step(np.zeros(12), observe=False)
            self.assertAlmostEqual(float(env.data.time), 1.0)
        finally:
            env.close()
