"""Small, serialisable types shared by planner, scheduler, and recovery code."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


Arm = Literal["A", "B", "both"]
Zone = Literal["left", "middle", "right", "drawer"]


class RecoverabilityVerdict(str, Enum):
    DONE = "done"
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class SkillStep:
    """A validated node in the planner's task graph."""

    id: int
    skill: str
    arm: Arm
    needs: tuple[int, ...] = ()
    zone: Zone = "middle"
    object: str | None = None
    target: str | None = None
    preconditions: tuple[str, ...] = ()
    timeout_s: float = 30.0
    donor: str | None = None
    receiver: str | None = None

    @property
    def goal(self) -> str | None:
        """The planner's goal; ``target`` remains the compatible serialized key."""

        return self.target

    @property
    def arms(self) -> frozenset[str]:
        return frozenset(("A", "B") if self.arm == "both" else (self.arm,))


@dataclass(slots=True)
class Plan:
    steps: list[SkillStep]
    source: str = "rule_based"


@dataclass(frozen=True, slots=True)
class Observation:
    """The only data a learned skill is allowed to receive at runtime."""

    top: object
    wrist_a: object
    wrist_b: object
    joint_positions: object


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    verdict: RecoverabilityVerdict
    # Empirical rollout success rates belong in statistics, not an uncalibrated
    # monitor confidence presented to an operator.
    confidence: float | None = None
    reason: str = ""


@dataclass(slots=True)
class EpisodeResult:
    seed: int
    success: bool
    completed_steps: list[int] = field(default_factory=list)
    retries: int = 0
    replans: int = 0
