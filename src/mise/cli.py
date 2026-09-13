"""Local robotics, console and evidence entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import yaml

from .collection import collect_drawer_episodes
from .planner_teacher import generate_planner_records
from .planner import RuleBasedPlanner, DEFAULT_COMMAND
from .recovery_dataset import collect_drawer_fork_labels
from .scheduler import Scheduler
from .scripted import DrawerExpert
from .sim import BimanualTableEnv, build_scene


EXAMPLE_COMMAND = DEFAULT_COMMAND


def run_demo(video: Path) -> None:
    from .telemetry import FIXTURE_NOTE
    print(FIXTURE_NOTE)
    env = BimanualTableEnv(seed=0)
    try:
        frames = DrawerExpert(env).run(record_hz=5)
        video.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(video, frames, fps=5)
        print(f"wrote {video} ({len(frames)} frames)")
    finally:
        env.close()


def check_plan(seed_file: Path) -> None:
    seeds = yaml.safe_load(seed_file.read_text())
    if not isinstance(seeds, dict) or not isinstance(seeds.get("required"), list):
        raise ValueError("seed file must contain a required list")
    planner = RuleBasedPlanner()
    for seed in seeds["required"]:
        plan = planner.plan(EXAMPLE_COMMAND)
        scheduler = Scheduler(plan)
        # This checks the entire scheduling path while learned skills are pending.
        while not scheduler.done:
            for step in scheduler.start_ready():
                scheduler.complete(step.id)
    print(f"Scheduling checks passed for {len(seeds['required'])} seeds. No physics was evaluated; this is not task success.")


def run_recorded(seed: int, output: Path, controller: str = 'contact_expert', preset: str = 'nominal',
                 command: str | None = None, recovery_mode: str = "adaptive") -> bool:
    """Run one isolated worker and retain its complete evidence on success or failure."""
    import time
    from .server import RunManager, RunRequest
    from .telemetry import CONTACT_COMMAND
    manager = RunManager(output)
    try:
        command = command or (CONTACT_COMMAND if controller == 'contact_expert' else 'Open the drawer with arm A.')
        run = manager.create(RunRequest(seed=seed, controller=controller, command=command,
                                        preset=preset, recovery_mode=recovery_mode))
        print(f"Started {run['id']}: {command}", flush=True)
        while manager.get(run['id'])['status'] not in {'completed', 'stopped', 'failed'}:
            time.sleep(.2)
        if manager.process is not None:
            manager.process.join(timeout=5)
        record = manager.get(run['id'])
        outcome = ('full_task_success' if record['summary'].get('scope') == 'full_task'
                   else 'contact_skill_success' if controller == 'contact_expert' else 'fixture_drawer_open')
        passed = record['status'] == 'completed' and record['summary'].get(outcome) is True
        print(f"seed={seed} run={run['id']} status={record['status']} {outcome}={record['summary'].get(outcome)}")
        print(f"Evidence: {output.resolve() / run['id']}")
        print(record['summary']['reason'])
        return passed
    finally:
        manager.close()


def run_eval(seed_file: Path, output: Path, controller: str = 'contact_expert') -> None:
    """Retain all fixed-seed attempts and score only their declared scope."""
    seeds = yaml.safe_load(seed_file.read_text())
    if not isinstance(seeds, dict) or not isinstance(seeds.get("required"), list):
        raise ValueError("seed file must contain a required list")
    command = "Set the table." if controller == 'contact_expert' else None
    outcomes = [run_recorded(int(seed), output, controller, command=command) for seed in seeds['required']]
    scope = "full task" if controller == 'contact_expert' else "drawer fixture"
    print(f"{scope}: {sum(outcomes)}/{len(outcomes)} passed. All attempts retained.")


def collect(episodes: int, output: Path, record_hz: int) -> None:
    summaries = collect_drawer_episodes(output, episodes=episodes, record_hz=record_hz)
    print(f"wrote {len(summaries)} aligned drawer episodes to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mise")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scene")
    demo = commands.add_parser("demo")
    demo.add_argument("--video", type=Path, default=Path("artifacts/day1_drawer_expert.mp4"))
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--seeds", type=Path, default=Path("eval/seeds.yaml"))
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    evaluate.add_argument('--controller', choices=('contact_expert', 'scripted_drawer'), default='contact_expert')
    contact = commands.add_parser('contact')
    contact.add_argument('--seed', type=int, default=1001)
    contact.add_argument('--output', type=Path, default=Path('artifacts/contact_runs'))
    contact.add_argument('--preset', choices=('nominal', 'low_friction', 'displaced_objects'), default='nominal')
    full = commands.add_parser('full')
    full.add_argument('--seed', type=int, default=1001)
    full.add_argument('--output', type=Path, default=Path('artifacts/full_runs'))
    full.add_argument('--recovery-mode', choices=('none', 'blind_retry', 'adaptive'), default='adaptive')
    plan_check = commands.add_parser("check-plan")
    plan_check.add_argument("--seeds", type=Path, default=Path("eval/seeds.yaml"))
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    serve.add_argument("--read-only", action="store_true")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--episodes", type=int, default=20)
    collect_parser.add_argument("--output", type=Path, default=Path("data/raw_drawer"))
    collect_parser.add_argument("--record-hz", type=int, default=10)
    planner_data = commands.add_parser("planner-data")
    planner_data.add_argument("--count", type=int, default=10000)
    planner_data.add_argument("--output", type=Path, default=Path("data/planner_teacher"))
    planner_data.add_argument("--no-images", action="store_true")
    label_data = commands.add_parser("label-data")
    label_data.add_argument("--samples", type=int, default=1000)
    label_data.add_argument("--output", type=Path, default=Path("data/fork_labels"))
    label_data.add_argument("--with-images", action="store_true")
    args = parser.parse_args()
    if args.command == "scene":
        print(build_scene())
    elif args.command == "demo":
        run_demo(args.video)
    elif args.command == "eval":
        run_eval(args.seeds, args.output, args.controller)
    elif args.command == 'contact':
        if not run_recorded(args.seed, args.output, preset=args.preset):
            raise SystemExit(1)
    elif args.command == 'full':
        if not run_recorded(args.seed, args.output, command="Set the table.",
                            recovery_mode=args.recovery_mode):
            raise SystemExit(1)
    elif args.command == "check-plan":
        check_plan(args.seeds)
    elif args.command == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(runs_dir=args.runs_dir, read_only=args.read_only), host=args.host, port=args.port)
    elif args.command == "collect":
        collect(args.episodes, args.output, args.record_hz)
    elif args.command == "planner-data":
        print(generate_planner_records(args.output, count=args.count, with_images=not args.no_images))
    elif args.command == "label-data":
        print(collect_drawer_fork_labels(args.output, samples=args.samples, with_images=args.with_images))


if __name__ == "__main__":
    main()
