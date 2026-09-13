"""Dispatch the registered and independently validated physical expert skills."""
from .drawer_skill import DRAWER_COMMAND
from .planner import DEFAULT_COMMAND, validate_plan_graph
from .telemetry import CONTACT_COMMAND

FULL_COMMAND = "Set the table."
SUPPORTED_CONTACT_COMMANDS = (FULL_COMMAND, DEFAULT_COMMAND, CONTACT_COMMAND, DRAWER_COMMAND)


def is_drawer_plan(plan):
    return len(plan.steps) == 1 and (plan.steps[0].skill, plan.steps[0].arm,
            plan.steps[0].object, plan.steps[0].target) == ('open_drawer', 'A', 'drawer', 'drawer_open')


def validate_contact_plan(plan):
    validate_plan_graph(plan)
    from .full_task import is_full_plan, validate_full_plan
    if is_full_plan(plan):
        validate_full_plan(plan)
        return
    if is_drawer_plan(plan):
        return
    try:
        from .manipulation import validate_contact_plan as validate_mug_plan
        validate_mug_plan(plan)
    except ValueError as exc:
        raise ValueError(str(exc) + ' Available physical commands: ' + ' '.join(SUPPORTED_CONTACT_COMMANDS)) from exc


def create_skill_env(plan, seed):
    validate_contact_plan(plan)
    from .full_task import is_full_plan
    if is_full_plan(plan):
        from .full_task import create_full_env
        return create_full_env(seed)
    if is_drawer_plan(plan):
        from .drawer_skill import create_drawer_env
        return create_drawer_env(seed)
    from .manipulation import create_contact_env
    return create_contact_env(seed)


def create_skill_controller(env, plan, *, recovery_mode="adaptive", recovery_memory=None,
                            episode_id="local-episode", memory_write=True):
    from .full_task import is_full_plan
    if is_full_plan(plan):
        from .full_task import FullTaskController
        return FullTaskController(env, plan, recovery_mode=recovery_mode,
                                  recovery_memory=recovery_memory,
                                  episode_id=episode_id, memory_write=memory_write)
    if is_drawer_plan(plan):
        from .drawer_skill import DrawerContactController
        return DrawerContactController(env, plan)
    from .manipulation import ManipulationController
    return ManipulationController(env, plan)


def create_skill_evaluator(env, plan):
    from .full_task import is_full_plan
    if is_full_plan(plan):
        from .full_task import FullTaskEvaluator
        return FullTaskEvaluator(env)
    if is_drawer_plan(plan):
        from .drawer_evaluation import DrawerContactEvaluator
        return DrawerContactEvaluator(env.model)
    from .contact_evaluation import ContactSkillEvaluator
    return ContactSkillEvaluator(env.model)
