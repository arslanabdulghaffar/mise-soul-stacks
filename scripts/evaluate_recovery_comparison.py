"""Compare recovery policies on identical predeclared full-task seeds."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from mise.evaluation import wilson_interval
from mise.telemetry import atomic_json, file_hash, utc_now
from evaluate_full_task import evaluate


VARIANTS = (
    ("none", "No recovery"),
    ("blind_retry", "Bounded blind retry"),
    ("adaptive", "Monitored cost-selected recovery"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=Path("eval/seeds.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/evaluations/recovery_matched_seeds.json"))
    args = parser.parse_args()
    seeds = [int(seed) for seed in yaml.safe_load(args.seeds.read_text())["required"]]
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for mode, label in VARIANTS:
        results = []
        for seed in seeds:
            result = evaluate(seed, recovery_mode=mode)
            result["variant_label"] = label
            results.append(result)
            rows.append(result)
            print(f"variant={mode} seed={seed} success={result['success']} "
                  f"attempts={result['recovery_attempts']}", flush=True)
        successes = sum(bool(result["success"]) for result in results)
        summaries[mode] = {
            "label": label,
            "successes": successes,
            "total": len(results),
            "wilson95": wilson_interval(successes, len(results)),
            "recovery_attempts": sum(int(result["recovery_attempts"]) for result in results),
            "total_simulation_seconds": sum(float(result["simulation_seconds"]) for result in results),
            "results": results,
        }
    adaptive = summaries["adaptive"]["successes"]
    retry = summaries["blind_retry"]["successes"]
    report = {
        "schema_version": "mise.recovery-comparison.v1",
        "created_at": utc_now(),
        "scope": "matched_seed_full_task_recovery_comparison",
        "command": "Set the table.",
        "seeds": seeds,
        "fixed_conditions": ["scene", "seed", "task", "time budget", "goal verifier"],
        "variants": summaries,
        "paired_success_gain_over_blind_retry": adaptive - retry,
        "source_sha256": {
            "full_task": file_hash(Path(__file__).resolve().parents[1] / "src/mise/full_task.py"),
            "supervisor": file_hash(Path(__file__).resolve().parents[1] / "src/mise/supervisor.py"),
            "evaluator": file_hash(Path(__file__).resolve().parents[1] / "scripts/evaluate_full_task.py"),
        },
        "limitations": (
            "The nominal suite contains only the naturally occurring failures for these fixed seeds; "
            "the comparison does not establish broad generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    csv_path = args.output.with_suffix(".csv")
    fields = ("recovery_mode", "variant_label", "seed", "success", "completed_steps",
              "recovery_attempts", "simulation_seconds", "wall_seconds", "direct_object_write", "failure")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    print(f"wrote {args.output} and {csv_path}; adaptive gain over blind retry: {adaptive - retry}")
    if adaptive != len(seeds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
