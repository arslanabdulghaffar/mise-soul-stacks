"""Empirical continuation labels; failed continuations do not establish a retry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .types import RecoveryDecision, RecoverabilityVerdict

CandidateArm = Literal["A", "B"]
Rollout = Callable[[CandidateArm, int], bool]


@dataclass(frozen=True, slots=True)
class ForkStatistics:
    current_arm: CandidateArm
    current_success_rate: float
    alternate_success_rate: float
    forks_per_arm: int


class ForkLabeler:
    """Classify a state from privileged teacher rollouts on forked simulator state."""

    def __init__(self, *, forks_per_arm: int = 24, success_threshold: float = 0.60):
        if forks_per_arm < 1:
            raise ValueError("forks_per_arm must be positive")
        if not 0 < success_threshold <= 1:
            raise ValueError("success_threshold must be in (0, 1]")
        self.forks_per_arm = forks_per_arm
        self.success_threshold = success_threshold

    def label(self, current_arm: CandidateArm, rollout: Rollout) -> tuple[RecoveryDecision, ForkStatistics]:
        if current_arm not in ("A", "B"):
            raise ValueError("current_arm must be A or B")
        alternate_arm: CandidateArm = "B" if current_arm == "A" else "A"
        current = _success_rate(rollout, current_arm, self.forks_per_arm)
        alternate = _success_rate(rollout, alternate_arm, self.forks_per_arm)
        statistics = ForkStatistics(current_arm, current, alternate, self.forks_per_arm)
        if current >= self.success_threshold:
            return RecoveryDecision(RecoverabilityVerdict.CONTINUE, reason="current-arm teacher continuations pass the empirical threshold"), statistics
        if alternate >= self.success_threshold:
            return RecoveryDecision(RecoverabilityVerdict.REPLAN, reason="alternate teacher continuation is viable; runtime constraints still require validation"), statistics
        return RecoveryDecision(RecoverabilityVerdict.STOP, reason="No tested continuation passes; no corrective regrasp was evaluated."), statistics


def _success_rate(rollout: Rollout, arm: CandidateArm, count: int) -> float:
    return sum(bool(rollout(arm, index)) for index in range(count)) / count


@dataclass(frozen=True, slots=True)
class MuJoCoSnapshot:
    """State copied before a teacher rollout mutates a simulation fork."""

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray

    @classmethod
    def capture(cls, data: Any) -> "MuJoCoSnapshot":
        return cls(float(data.time), data.qpos.copy(), data.qvel.copy(), data.act.copy(), data.ctrl.copy())

    def restore(self, data: Any) -> None:
        data.time = self.time
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        data.act[:] = self.act
        data.ctrl[:] = self.ctrl


TeacherController = Callable[[CandidateArm, Any, Any, int], None]
SuccessPredicate = Callable[[Any, Any], bool]


class MuJoCoForkRunner:
    """Execute privileged teachers from identical copies of an observed state."""

    def __init__(self, mujoco: Any, model: Any, *, horizon: int = 150):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.mujoco = mujoco
        self.model = model
        self.horizon = horizon

    def rollout_from(self, snapshot: MuJoCoSnapshot, arm: CandidateArm, fork_index: int, controller: TeacherController, success: SuccessPredicate) -> bool:
        data = self.mujoco.MjData(self.model)
        snapshot.restore(data)
        self.mujoco.mj_forward(self.model, data)
        for step in range(self.horizon):
            controller(arm, self.model, data, fork_index * self.horizon + step)
            self.mujoco.mj_step(self.model, data)
            if success(self.model, data):
                return True
        return bool(success(self.model, data))
