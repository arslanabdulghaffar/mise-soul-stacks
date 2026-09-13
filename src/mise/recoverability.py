"""Runtime contract for the learned recoverability classifier.

The actual student will replace `HeuristicRecoveryHead`; it must keep this API and
receive only `Observation`, never simulator state.
"""

from __future__ import annotations

from .types import Observation, RecoveryDecision, RecoverabilityVerdict


class HeuristicRecoveryHead:
    """A transparent development stub used until fork-labelled weights exist."""

    def predict(self, observation: Observation, *, grasp_error: float = 0.0, alternate_arm_viable: bool = False, retry_feasible: bool = False) -> RecoveryDecision:
        if grasp_error >= 0.8:
            return RecoveryDecision(RecoverabilityVerdict.RETRY if retry_feasible else RecoverabilityVerdict.STOP, reason="A retry requires a separately tested corrective action.")
        if alternate_arm_viable and grasp_error >= 0.45:
            return RecoveryDecision(RecoverabilityVerdict.REPLAN, reason="Caller reports an alternate action; validate instruction constraints in the supervisor.")
        return RecoveryDecision(RecoverabilityVerdict.CONTINUE, reason="Development heuristic only; no visual classifier is loaded.")
