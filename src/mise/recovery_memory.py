"""Persistent outcome memory for selecting already validated recovery actions.

This updates empirical recovery estimates, not motor-policy weights. Context
comes from the command, visual monitor and versioned skill registry. Simulator
object state and evaluator predicates are deliberately absent from this API.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .supervisor import MonitorEvidence, RecoveryCandidate, RecoverySupervisor


@dataclass(frozen=True)
class RecoveryContext:
    skill: str
    object: str
    phase: str
    failure_kind: str
    state_bucket: str
    scene_version: str
    policy_version: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in asdict(self).values()):
            raise ValueError("Recovery context requires explicit observable and version fields.")

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class RecoveryMemory:
    """SQLite history with idempotent attempt IDs and exact context matching.

    Unknown/occluded outcomes are retained for inspection but never scored.
    A small declared prior prevents one new outcome from erasing the original
    validation estimate. No history can register an untested action or relax
    its arm, feasibility, collision, goal-preservation or budget constraints.
    """

    def __init__(self, path: Path, *, prior_strength: float = 2.0):
        if not math.isfinite(prior_strength) or prior_strength <= 0:
            raise ValueError("prior_strength must be positive and finite")
        self.path, self.prior_strength = Path(path), prior_strength
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS recovery_outcomes (
                attempt_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL,
                context TEXT NOT NULL, candidate TEXT NOT NULL, arm TEXT NOT NULL,
                outcome TEXT NOT NULL CHECK(outcome IN ('success','failure','unknown')),
                duration_s REAL NOT NULL, observation_ref TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS recovery_context ON recovery_outcomes(context, candidate, arm)")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def record(self, context: RecoveryContext, candidate: RecoveryCandidate, *,
               attempt_id: str, episode_id: str, outcome: str,
               duration_s: float, observation_ref: str) -> bool:
        """Record an executed recovery's visual outcome; return False on replay.

        observation_ref identifies the saved monitor trace, not a claimed causal
        explanation. A failed grasp alone does not establish why it failed.
        """
        if not candidate.registered or candidate.evidence_count <= 0:
            raise ValueError("Only previously validated registered actions enter recovery memory.")
        if outcome not in {"success", "failure", "unknown"}:
            raise ValueError("outcome must be success, failure or unknown")
        if any(not isinstance(value, str) or not value.strip()
               for value in (attempt_id, episode_id, observation_ref)):
            raise ValueError("Attempt, episode and observation references are required.")
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Observed duration must be positive and finite.")
        row = (attempt_id, episode_id, context.key, candidate.name, candidate.arm,
               outcome, float(duration_s), observation_ref)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT * FROM recovery_outcomes WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if previous is not None:
                if previous != row:
                    raise ValueError("An attempt ID cannot be reused for a different outcome.")
                return False
            db.execute("INSERT INTO recovery_outcomes VALUES (?,?,?,?,?,?,?,?)", row)
        return True

    def estimate(self, context: RecoveryContext, candidate: RecoveryCandidate) -> RecoveryCandidate:
        with self._connect() as db:
            count, successes, duration = db.execute(
                """SELECT COUNT(*), COALESCE(SUM(outcome = 'success'), 0), AVG(duration_s)
                FROM recovery_outcomes WHERE context = ? AND candidate = ? AND arm = ?
                AND outcome != 'unknown'""", (context.key, candidate.name, candidate.arm)).fetchone()
        if count == 0 or not candidate.registered or candidate.evidence_count <= 0:
            return candidate
        weight = self.prior_strength
        return replace(candidate,
                       success_rate=(weight * candidate.success_rate + successes) / (weight + count),
                       duration_s=(weight * candidate.duration_s + count * duration) / (weight + count),
                       evidence_count=candidate.evidence_count + count)

    def failed_in_episode(self, context: RecoveryContext, episode_id: str) -> set[tuple[str, str]]:
        with self._connect() as db:
            rows = db.execute("""SELECT DISTINCT candidate, arm FROM recovery_outcomes
                WHERE context = ? AND episode_id = ? AND outcome = 'failure'""",
                (context.key, episode_id)).fetchall()
        return set(rows)

    def summary(self) -> dict:
        """Return aggregate outcomes for run evidence and the operator console."""

        with self._connect() as db:
            total, successes, failures, unknown = db.execute("""
                SELECT COUNT(*), COALESCE(SUM(outcome = 'success'), 0),
                COALESCE(SUM(outcome = 'failure'), 0), COALESCE(SUM(outcome = 'unknown'), 0)
                FROM recovery_outcomes
            """).fetchone()
            candidates = db.execute("""
                SELECT candidate, arm, COUNT(*), COALESCE(SUM(outcome = 'success'), 0),
                COALESCE(SUM(outcome = 'failure'), 0), AVG(duration_s)
                FROM recovery_outcomes GROUP BY candidate, arm ORDER BY candidate, arm
            """).fetchall()
        return {
            "total_outcomes": total, "successes": successes,
            "failures": failures, "unknown": unknown,
            "candidates": [
                {"candidate": name, "arm": arm, "outcomes": count,
                 "successes": passed, "failures": failed, "mean_duration_s": duration}
                for name, arm, count, passed, failed, duration in candidates
            ],
        }


class AdaptiveRecoverySupervisor(RecoverySupervisor):
    """Use remembered outcomes before the existing constrained cost selector.

    Runtime callers must record outcomes after executing the selected action.
    Repeating a failed candidate in an unchanged observable context within one
    episode is excluded. A genuinely changed state gets its own state bucket.
    """

    def __init__(self, memory: RecoveryMemory, **limits):
        super().__init__(**limits)
        self.memory = memory

    def decide_with_memory(self, step_id: int, evidence: MonitorEvidence,
                           candidates: tuple[RecoveryCandidate, ...], *,
                           context: RecoveryContext, episode_id: str,
                           elapsed_s: float = 0, required_arm: str | None = None):
        if not episode_id.strip():
            raise ValueError("episode_id is required")
        failed = self.memory.failed_in_episode(context, episode_id)
        estimated = tuple(self.memory.estimate(context, candidate) for candidate in candidates
                          if (candidate.name, candidate.arm) not in failed)
        return super().decide(step_id, evidence, estimated, elapsed_s=elapsed_s,
                              required_arm=required_arm)
