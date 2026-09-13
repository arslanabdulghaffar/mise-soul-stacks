"""Dependency and resource scheduling; geometry clearance is an explicit input."""

from __future__ import annotations

from .types import Plan, SkillStep
from .planner import validate_plan_graph
from collections.abc import Callable


class Scheduler:
    """Select ready steps that do not claim the same arm or table zone."""

    def __init__(self, plan: Plan, *, parallel: bool = False,
                 geometry_clear: Callable[[SkillStep, tuple[SkillStep, ...]], bool] | None = None):
        validate_plan_graph(plan)
        if parallel and geometry_clear is None:
            raise ValueError("parallel scheduling requires a validated geometry clearance callback")
        self.plan = plan
        self.parallel = parallel
        self.geometry_clear = geometry_clear
        self._by_id = {step.id: step for step in plan.steps}
        self.completed: set[int] = set()
        self.running: set[int] = set()
        self.failed: set[int] = set()

    @property
    def done(self) -> bool:
        return len(self.completed) == len(self.plan.steps)

    def ready(self) -> list[SkillStep]:
        return [
            step
            for step in self.plan.steps
            if step.id not in self.completed | self.running | self.failed
            and set(step.needs) <= self.completed
        ]

    def start_ready(self) -> list[SkillStep]:
        """Start the maximal stable set, in planner order, of non-conflicting steps."""

        selected: list[SkillStep] = []
        if self.running and not self.parallel:
            return []
        used_arms = set().union(*(self._by_id[step_id].arms for step_id in self.running)) if self.running else set()
        used_zones = {self._by_id[step_id].zone for step_id in self.running}
        used_objects = {self._by_id[step_id].object for step_id in self.running} - {None}
        for step in self.ready():
            # Dual-arm skills reserve every arm. They cannot overlap any skill.
            if step.arms & used_arms or step.zone in used_zones or (step.object and step.object in used_objects):
                continue
            others = tuple(self._by_id[i] for i in sorted(self.running)) + tuple(selected)
            if self.geometry_clear is not None and not self.geometry_clear(step, others):
                continue
            selected.append(step)
            used_arms.update(step.arms)
            used_zones.add(step.zone)
            if step.object:
                used_objects.add(step.object)
            if not self.parallel:
                break
        self.running.update(step.id for step in selected)
        return selected

    def complete(self, step_id: int) -> None:
        if step_id not in self.running:
            raise ValueError(f"step {step_id} is not running")
        self.running.remove(step_id)
        self.completed.add(step_id)

    def retry(self, step_id: int) -> None:
        if step_id not in self.running:
            raise ValueError(f"step {step_id} is not running")
        self.running.remove(step_id)

    def replan(self, plan: Plan) -> None:
        """Replace unfinished graph while retaining completed, compatible work."""

        if self.running:
            raise ValueError("replan only at a stable boundary with no running skills")
        validate_plan_graph(plan)
        incoming = {step.id: step for step in plan.steps}
        if not self._by_id.keys() <= incoming.keys():
            raise ValueError("a replan cannot discard requested or completed goals")
        for step_id, original in self._by_id.items():
            if step_id in self.completed and incoming[step_id] != original:
                raise ValueError("a replan cannot change completed work")
            if any(getattr(incoming[step_id], name) != getattr(original, name)
                   for name in ("arm", "object", "target", "donor", "receiver")):
                raise ValueError("a replan cannot change required arms or goals")
        self.plan = plan
        self._by_id = {step.id: step for step in plan.steps}
        self.running.clear()
        self.failed.clear()
