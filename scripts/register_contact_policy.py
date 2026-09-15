"""Register a passing frozen ACT evaluation for local console execution."""
import argparse
import json
from pathlib import Path

from mise.sim import ROOT
from mise.telemetry import atomic_json, file_hash
from scripts.validate_intel_pair import read_evaluation

RUNTIME_SOURCES = (
    "src/mise/learned_controller.py", "src/mise/learning/runtime.py",
    "src/mise/learning/preprocessing.py", "src/mise/learning/openvino_config.py",
    "src/mise/contact_vision.py", "src/mise/contact_evaluation.py",
    "src/mise/manipulation.py", "src/mise/sim.py", "src/mise/kinematics.py",
)


def register(model: Path, evaluation: Path, scene: Path) -> dict:
    manifest, results = read_evaluation(evaluation)
    if manifest["backend"] != "openvino" or not results["complete"] or results["total"] < 10 or results["successes"] != results["total"]:
        raise ValueError("Registration requires at least ten completed, successful OpenVINO task attempts")
    if manifest.get("execution_steps", 15) != 15 or manifest.get("temporal_ensemble", False):
        raise ValueError("Console currently executes the validated 15-step non-ensemble configuration")
    files = {}
    for name in ("contact_act.xml", "contact_act.bin", "model_manifest.json"):
        files[name] = file_hash(model / "openvino" / name)
        if files[name] != manifest["files_sha256"][name]:
            raise ValueError(f"Evaluation used a different model file: {name}")
    exported = json.loads((model / "openvino/model_manifest.json").read_text())
    if not exported.get("numerical_parity_passed") or manifest["scene_sha256"] != file_hash(scene):
        raise ValueError("Scene or export parity does not match the evaluated policy")
    sources = {name: file_hash(ROOT / name) for name in RUNTIME_SOURCES}
    if any(value != manifest["files_sha256"][name] for name, value in sources.items()):
        raise ValueError("Learned controller dependencies changed after evaluation")
    record = dict(schema_version="mise.accepted-contact-policy.v1", scope="learned_mug_contact_skill",
                  successes=results["successes"], total=results["total"], seeds=manifest["seeds"],
                  model_files_sha256=files, checkpoint_sha256=manifest["checkpoint_sha256"],
                  scene_path=str(scene.resolve().relative_to(ROOT)), scene_sha256=file_hash(scene),
                  runtime_sources_sha256=sources, runtime_config=manifest["runtime_config"],
                  evaluation_manifest_sha256=file_hash(evaluation / "manifest.json"),
                  evaluation_results_sha256=file_hash(evaluation / "results.json"),
                  full_task_success=None, intel_target_verified=False)
    atomic_json(model / "accepted_policy.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=Path("data/contact/frozen/scene.xml"))
    args = parser.parse_args()
    print(json.dumps(register(args.model, args.evaluation, args.scene), indent=2))
