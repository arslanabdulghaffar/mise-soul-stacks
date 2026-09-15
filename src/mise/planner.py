"""Validated task graphs and a disclosed, deliberately bounded command parser.

This module does not implement a VLM. Unsupported language is rejected instead of
silently dropping objects, ordering constraints, or unsupported pouring actions.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace

from .types import Arm, Plan, SkillStep

DEFAULT_COMMAND = (
    "Open the top drawer with arm A, place the plate in the center with arm A, "
    "retrieve the spoon from the drawer with arm A, pass the spoon from arm A to arm B, "
    "place the spoon on the right with arm B, place the fork on the left with arm A, "
    "place the mug in the upper-right with arm B."
)
SUPPORTED_SKILLS = frozenset({"open_drawer", "pick", "pick_place", "handoff"})
SUPPORTED_OBJECTS = frozenset({"drawer", "plate", "mug", "spoon", "fork"})
_ARMS = frozenset({"A", "B", "both"})
_ZONES = frozenset({"left", "middle", "right", "drawer"})
_GOALS = {"plate": "table_center", "spoon": "table_right", "fork": "table_left", "mug": "table_upper_right"}


class PlanValidationError(ValueError):
    pass


class UnsupportedCommandError(PlanValidationError):
    pass


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanValidationError(f"{field} must be an integer")
    return value


def parse_plan(payload: str | Mapping[str, object]) -> Plan:
    """Validate graph syntax and the supported manipulation vocabulary."""

    try:
        raw = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"steps"}:
        raise PlanValidationError("plan must contain exactly a 'steps' field")
    records = raw["steps"]
    if not isinstance(records, list) or not records:
        raise PlanValidationError("steps must be a non-empty list")
    steps: list[SkillStep] = []
    ids: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise PlanValidationError("each step must be an object")
        allowed = {"id", "skill", "arm", "needs", "zone", "object", "to", "goal", "preconditions", "timeout_s", "donor", "receiver"}
        unknown = set(record) - allowed
        if unknown:
            raise PlanValidationError(f"unknown step fields: {sorted(unknown)}")
        step_id = _as_int(record.get("id"), "id")
        skill, arm = record.get("skill"), record.get("arm")
        needs, zone = record.get("needs", []), record.get("zone", "middle")
        if step_id in ids or step_id < 1:
            raise PlanValidationError("step ids must be unique positive integers")
        if not isinstance(skill, str) or skill not in SUPPORTED_SKILLS:
            raise PlanValidationError("unsupported skill; pouring is not implemented")
        if not isinstance(arm, str) or arm not in _ARMS or not isinstance(zone, str) or zone not in _ZONES:
            raise PlanValidationError("unknown arm or zone")
        if not isinstance(needs, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in needs):
            raise PlanValidationError("needs must be a list of integer ids")
        if step_id in needs or len(set(needs)) != len(needs):
            raise PlanValidationError("dependencies cannot include self or duplicate ids")
        obj = record.get("object")
        if obj is not None and (not isinstance(obj, str) or obj not in SUPPORTED_OBJECTS):
            raise PlanValidationError("unknown object")
        goal = record.get("goal", record.get("to"))
        if "goal" in record and "to" in record and record["goal"] != record["to"]:
            raise PlanValidationError("goal and to disagree")
        if goal is not None and (not isinstance(goal, str) or not goal.strip()):
            raise PlanValidationError("goal must be a non-empty string")
        conditions = record.get("preconditions", [])
        if not isinstance(conditions, list) or any(not isinstance(x, str) or not x.strip() for x in conditions):
            raise PlanValidationError("preconditions must be a list of non-empty strings")
        timeout = record.get("timeout_s", 30.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise PlanValidationError("timeout_s must be a finite positive number")
        donor, receiver = record.get("donor"), record.get("receiver")
        if skill == "handoff":
            if arm != "both" or donor not in ("A", "B") or receiver not in ("A", "B") or donor == receiver or obj is None:
                raise PlanValidationError("handoff requires both arms, an object, and distinct donor/receiver arms")
        elif donor is not None or receiver is not None:
            raise PlanValidationError("donor and receiver are only valid for handoff")
        elif arm == "both":
            raise PlanValidationError("only handoff currently supports both arms")
        ids.add(step_id)
        steps.append(SkillStep(step_id, skill, arm, tuple(needs), zone, obj, goal, tuple(conditions), float(timeout), donor, receiver))
    validate_plan_graph(Plan(steps))
    return Plan(steps=steps, source="vlm")


def validate_plan_graph(plan: Plan) -> None:
    """Reject duplicate ids, missing dependencies and cycles, including Python plans."""

    ids = {step.id for step in plan.steps}
    if len(ids) != len(plan.steps) or any(isinstance(step.id, bool) or not isinstance(step.id, int) or step.id < 1 for step in plan.steps):
        raise PlanValidationError("step ids must be unique positive integers")
    if any(set(step.needs) - ids for step in plan.steps):
        raise PlanValidationError("plan depends on a missing step")
    dependencies = {step.id: set(step.needs) for step in plan.steps}
    resolved: set[int] = set()
    while dependencies:
        ready = [step_id for step_id, needs in dependencies.items() if needs <= resolved]
        if not ready:
            raise PlanValidationError("dependency graph contains a cycle")
        for step_id in ready:
            resolved.add(step_id)
            del dependencies[step_id]


class RuleBasedPlanner:
    """Parse sequential drawer, pick, place and handoff clauses only.

    Commas, semicolons, ``then`` and ``and`` between actions preserve textual
    order. More complex before/after, negation and conditional language is
    unsupported. Scene state can skip an implicit drawer prerequisite, but an
    explicit open instruction remains in the graph.
    """

    def plan(self, command: str, *, drawer_open: bool = False) -> Plan:
        if not isinstance(command, str) or not command.strip() or len(command) > 2000:
            raise UnsupportedCommandError("command must contain 1 to 2000 characters")
        text = command.lower().strip().rstrip(".! ")
        if re.search(r"\b(pour\w*|fill|water|liquid)\b", text):
            raise UnsupportedCommandError("Pouring is unsupported; the core task uses a direct spoon hand-off.")
        unordered_task = text in {"set the table", "set a dinner table", "set the dinner table"}
        if unordered_task:
            text = DEFAULT_COMMAND.lower().rstrip(".")
        if re.search(r"\b(before|after|unless|until|except|without|not|don't|never|only|simultaneously)\b", text):
            raise UnsupportedCommandError("Use supported actions in execution order, separated by commas or 'then'.")
        clauses = re.split(r"\s*(?:[,;]|\bthen\b|\band\b(?=\s+(?:open|pick|retrieve|take|place|put|position|pass|hand|set)\b))\s*", text)
        steps: list[SkillStep] = []
        held: dict[str, str] = {}
        last_object: str | None = None
        drawer_ready = drawer_open

        def add(skill: str, arm: str, obj: str, goal: str | None = None, *, donor: str | None = None, receiver: str | None = None) -> None:
            nonlocal drawer_ready
            conditions = ("drawer_open",) if obj in {"spoon", "fork"} and skill in {"pick", "pick_place"} and obj not in held else ()
            if conditions and not drawer_ready:
                add("open_drawer", "A", "drawer", "drawer_open")
            if skill == "handoff":
                conditions = (f"{donor}_holds_{obj}", "shared_pose_reachable")
            zone = "drawer" if obj == "drawer" else "left" if goal == "table_left" else "right" if goal in {"table_right", "table_upper_right"} else "middle"
            # Reserve time for the registered bounded corrections; the global
            # 180-second episode limit and explicit model-plan deadlines remain.
            timeout = 55.0 if skill == "pick_place" and obj == "fork" else 45.0 if skill == "pick_place" and obj == "spoon" else 30.0
            steps.append(SkillStep(len(steps) + 1, skill, arm, (steps[-1].id,) if steps else (), zone, obj, goal, conditions, timeout, donor, receiver))
            if skill == "open_drawer":
                drawer_ready = True

        for clause in clauses:
            clause = re.sub(r"^(?:and\s+|please\s+)", "", clause.strip())
            if not clause:
                continue
            opener = re.fullmatch(r"open (?:the )?(?:top )?drawer(?: (?:with|using) (?:arm )?([ab]))?", clause)
            if opener:
                add("open_drawer", (opener.group(1) or "a").upper(), "drawer", "drawer_open")
                continue
            handoff = re.fullmatch(r"(?:pass|hand(?: off)?|handoff) (?:the )?(spoon|fork|plate|mug|it)(?: from (?:arm )?([ab]))? to (?:arm )?([ab])", clause)
            if handoff:
                obj = last_object if handoff.group(1) == "it" else handoff.group(1)
                if obj is None:
                    raise UnsupportedCommandError("A hand-off pronoun needs a preceding object.")
                donor = (handoff.group(2) or held.get(obj, "A")).upper()
                receiver = handoff.group(3).upper()
                if donor == receiver or (obj in held and held[obj] != donor):
                    raise UnsupportedCommandError("Hand-off must transfer from the holding arm to the other arm.")
                if obj not in held:
                    add("pick", donor, obj, f"held_by_{donor}")
                add("handoff", "both", obj, f"held_by_{receiver}", donor=donor, receiver=receiver)
                held[obj], last_object = receiver, obj
                continue
            action = re.fullmatch(r"(pick up|pick|retrieve|take|place|put|position|set) (?:the )?(plate|mug|spoon|fork|it)(.*)", clause)
            if not action:
                raise UnsupportedCommandError(f"Unsupported action clause: {clause!r}")
            verb, object_word, tail = action.groups()
            obj = last_object if object_word == "it" else object_word
            if obj is None:
                raise UnsupportedCommandError("A pronoun needs a preceding object.")
            assigned = re.findall(r" (?:with|using) (?:arm )?([ab])\b", tail)
            if len(assigned) > 1:
                raise UnsupportedCommandError("Each action needs at most one arm assignment.")
            arm = assigned[0].upper() if assigned else held.get(obj, "B" if obj in {"mug", "spoon"} and verb in {"place", "put", "position", "set"} else "A")
            tail = re.sub(r" (?:with|using) (?:arm )?[ab]\b", "", tail).strip()
            tail = re.sub(r"^from (?:the )?(?:top )?drawer\s*", "", tail)
            location = re.fullmatch(r"(?:(?:in|on|at|to) (?:the )?)?(center|centre|table center|table_center|left|right|upper-right|upper right|table)?", tail)
            if not location:
                raise UnsupportedCommandError(f"Unsupported goal or command constraint: {tail!r}")
            place = verb in {"place", "put", "position", "set"}
            if obj in held and held[obj] != arm:
                raise UnsupportedCommandError("The requested arm does not hold the object; include an explicit hand-off.")
            goal = _GOALS[obj]
            word = location.group(1)
            if word and word != "table":
                goal = {"center": "table_center", "centre": "table_center", "table center": "table_center", "table_center": "table_center", "left": "table_left", "right": "table_right", "upper-right": "table_upper_right", "upper right": "table_upper_right"}[word]
            if not place and word:
                raise UnsupportedCommandError("Use a separate place clause after a pick action.")
            if place and goal != _GOALS[obj]:
                raise UnsupportedCommandError(f"The registered {obj} placement supports only {_GOALS[obj]}.")
            add("pick_place" if place else "pick", arm, obj, goal if place else f"held_by_{arm}")
            if place:
                held.pop(obj, None)
            else:
                held[obj] = arm
            last_object = obj
        if not steps:
            raise UnsupportedCommandError("No supported action was found.")
        if unordered_task:
            # A goal-only request does not impose the canonical example's textual
            # order. Keep physical prerequisites while exposing independent work.
            drawer_id = next(s.id for s in steps if s.skill == "open_drawer")
            last_by_object: dict[str, int] = {}
            graph = []
            for step in steps:
                dependencies = ()
                if step.object in last_by_object:
                    dependencies = (last_by_object[step.object],)
                elif step.object in {"spoon", "fork"}:
                    dependencies = (drawer_id,)
                graph.append(replace(step, needs=dependencies))
                if step.object is not None:
                    last_by_object[step.object] = step.id
            steps = graph
        return Plan(steps, source="rule_based")


def planner_output_or_fallback(output: str, command: str, *, drawer_open: bool) -> Plan:
    """Accept output only when it preserves the supported command's full contract.

    A model repair loop is not implemented. This is the disclosed deterministic
    fallback, and unsupported commands fail even when the model emits valid JSON.
    """

    expected = RuleBasedPlanner().plan(command, drawer_open=drawer_open)
    try:
        candidate = parse_plan(output)
        # Comparison in graph order conservatively preserves explicit action order,
        # arm assignments and goals. A model may use different positive step ids.
        if len(candidate.steps) != len(expected.steps):
            raise PlanValidationError("model changed requested actions")
        for index, (actual, required) in enumerate(zip(candidate.steps, expected.steps)):
            fields = ("skill", "arm", "object", "target", "donor", "receiver")
            if any(getattr(actual, field) != getattr(required, field) for field in fields):
                raise PlanValidationError("model changed an instruction constraint")
            expected_indices = {step.id: position for position, step in enumerate(expected.steps)}
            required_dependencies = {candidate.steps[expected_indices[dependency]].id for dependency in required.needs}
            if not required_dependencies <= set(actual.needs):
                raise PlanValidationError("model removed required action ordering")
            candidate.steps[index] = replace(actual, preconditions=required.preconditions, timeout_s=min(actual.timeout_s, required.timeout_s))
        return candidate
    except PlanValidationError:
        return expected
