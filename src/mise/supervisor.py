"""Deterministic, candidate-conditioned recovery with disclosed cost estimates.

Candidate estimates must come from tested continuations, eventually student
rollouts. This is a selector, not a trained predictor of recoverability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .types import RecoverabilityVerdict as Verdict, RecoveryDecision


@dataclass(frozen=True)
class RecoveryCandidate:
    name: str
    verdict: Verdict
    arm: str
    success_rate: float
    duration_s: float
    registered: bool = False
    feasible: bool = False
    preserves_constraints: bool = False
    preserves_completed_goals: bool = False
    evidence_count: int = 0
    # Optional energy/effort equivalent converted to seconds with a declared
    # application-specific weight before constructing this candidate.
    extra_cost: float = 0.0

    @property
    def expected_cost(self) -> float:
        return (self.duration_s + self.extra_cost) / self.success_rate if self.success_rate > 0 else math.inf

    def __post_init__(self):
        if self.verdict not in {Verdict.RETRY, Verdict.REPLAN}:
            raise ValueError("recovery candidates must retry or replan")
        if self.arm not in {"A", "B", "both"}:
            raise ValueError("candidate has an invalid arm")
        if not math.isfinite(self.success_rate) or not 0 <= self.success_rate <= 1:
            raise ValueError("success estimate must be between zero and one")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0 or not math.isfinite(self.extra_cost) or self.extra_cost < 0:
            raise ValueError("candidate duration must be positive and cost nonnegative")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be nonnegative")


@dataclass(frozen=True)
class MonitorEvidence:
    terminal_satisfied: bool = False
    progressing: bool = False
    failure_detected: bool = False
    ambiguous: bool = False
    fresh: bool = True
    stable_for_replan: bool = False


class RecoverySupervisor:
    def __init__(self, *, max_attempts: int = 2, episode_budget_s: float = 180, success_threshold: float = 0.6):
        if max_attempts < 0 or not math.isfinite(episode_budget_s) or episode_budget_s <= 0 or not 0 < success_threshold <= 1:
            raise ValueError("invalid supervisor limits")
        self.max_attempts, self.episode_budget_s, self.success_threshold = max_attempts, episode_budget_s, success_threshold
        self.attempts: dict[int, int] = {}
        self.selected: RecoveryCandidate | None = None
        self.selected_step: int | None = None

    def decide(self, step_id: int, evidence: MonitorEvidence, candidates: tuple[RecoveryCandidate, ...] = (),
               *, elapsed_s: float = 0, required_arm: str | None = None) -> RecoveryDecision:
        self.selected = None
        self.selected_step = None
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        if not evidence.fresh or elapsed_s >= self.episode_budget_s:
            return RecoveryDecision(Verdict.STOP, reason="Freshness or episode budget exceeded; cancel queued actions.")
        if evidence.terminal_satisfied and not evidence.ambiguous and not evidence.failure_detected:
            return RecoveryDecision(Verdict.DONE, reason="Verified terminal predicates remain satisfied.")
        if evidence.ambiguous:
            return RecoveryDecision(Verdict.STOP, reason="Evidence is ambiguous; controlled hold or assistance required.")
        if evidence.progressing and not evidence.failure_detected:
            return RecoveryDecision(Verdict.CONTINUE, reason="Observed progress remains plausible.")
        if self.attempts.get(step_id, 0) >= self.max_attempts:
            return RecoveryDecision(Verdict.STOP, reason="Per-skill recovery budget exhausted.")
        valid = [c for c in candidates if c.registered and c.feasible and c.preserves_constraints
                 and c.preserves_completed_goals and c.evidence_count > 0
                 and c.success_rate >= self.success_threshold
                 and (required_arm is None or c.arm == required_arm)
                 and (c.verdict != Verdict.REPLAN or evidence.stable_for_replan)
                 and elapsed_s + c.duration_s <= self.episode_budget_s]
        if not valid:
            return RecoveryDecision(Verdict.STOP, reason="No tested candidate satisfies feasibility, instruction and budget constraints.")
        self.selected = min(valid, key=lambda c: (c.expected_cost, -c.success_rate, c.name))
        self.selected_step = step_id
        return RecoveryDecision(self.selected.verdict, reason=f"Selected {self.selected.name}; lowest duration/success cost among viable tested candidates.")

    def begin_recovery(self, step_id: int) -> None:
        """Consume budget only when a selected correction actually begins."""
        if self.selected is None or self.selected_step != step_id:
            raise ValueError("select a valid candidate before starting recovery")
        if self.attempts.get(step_id, 0) >= self.max_attempts:
            raise ValueError("recovery budget exhausted")
        self.attempts[step_id] = self.attempts.get(step_id, 0) + 1
        self.selected = None
        self.selected_step = None
