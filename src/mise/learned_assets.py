"""Resolve a locally installed policy with retained, passing closed-loop evidence."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from .sim import ROOT
from .telemetry import file_hash

LEARNED_NOTE = "Learned ACT mug pick/place using RGB, robot joints and action history; independently scored contact skill, not full table-setting success."


def model_directory() -> Path:
    return Path(os.environ.get("MISE_CONTACT_ACT_MODEL", str(ROOT / "artifacts/models/contact_act_context"))).resolve()


def policy_installed() -> bool:
    return (model_directory() / "accepted_policy.json").is_file() and importlib.util.find_spec("openvino") is not None


def accepted_policy(directory: Path | None = None) -> dict:
    directory = (directory or model_directory()).resolve()
    registration = directory / "accepted_policy.json"
    try:
        accepted = json.loads(registration.read_text())
        if accepted.get("schema_version") != "mise.accepted-contact-policy.v1":
            raise ValueError("Unsupported policy registration")
        if accepted.get("successes", 0) < 10 or accepted["successes"] != accepted.get("total"):
            raise ValueError("Policy registration requires every seed in a completed ten-or-more-seed evaluation to pass")
        for name, expected in accepted["model_files_sha256"].items():
            if name not in {"contact_act.xml", "contact_act.bin", "model_manifest.json"}:
                raise ValueError("Unknown registered policy file")
            if file_hash(directory / "openvino" / name) != expected:
                raise ValueError(f"Registered policy file changed: {name}")
        if set(accepted["model_files_sha256"]) != {"contact_act.xml", "contact_act.bin", "model_manifest.json"}:
            raise ValueError("Incomplete registered model")
        scene = ROOT / accepted["scene_path"]
        if not scene.resolve().is_relative_to(ROOT) or file_hash(scene) != accepted["scene_sha256"]:
            raise ValueError("Registered policy scene changed")
        for name, expected in accepted["runtime_sources_sha256"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or file_hash(path) != expected:
                raise ValueError(f"Validated policy dependency changed: {name}")
        return {**accepted, "model_directory": str(directory), "scene_path": str(scene),
                "acceptance_sha256": file_hash(registration)}
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Install a validated ACT bundle before selecting the learned controller") from exc
