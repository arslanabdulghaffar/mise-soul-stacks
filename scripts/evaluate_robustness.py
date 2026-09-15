"""Predeclared full-task stress matrix with compiled, frozen scene variants."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from runpy import run_path
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LP_NUM_THREADS", "1")

import numpy as np

from mise.sim import ROOT
from mise.telemetry import atomic_json, file_hash, hardware_identity, utc_now
from scripts.evaluate_full_task import evaluate

FAMILIES = ("lighting", "background", "shape", "physical", "combined")


def variant(base: Path, destination: Path, seed: int, family: str) -> dict:
    if family not in FAMILIES:
        raise ValueError("Unknown robustness family")
    tree = ET.parse(base)
    root = tree.getroot()
    rng = np.random.default_rng(seed + 81000)
    changes = {}
    if family in ("lighting", "combined"):
        intensity = float(rng.uniform(.75, 1.25))
        angle = float(rng.uniform(-.4, .4))
        for light in root.findall(".//light"):
            diffuse = np.fromstring(light.get("diffuse", ".7 .7 .7"), sep=" ") * intensity
            light.set("diffuse", " ".join(map(str, diffuse)))
            light.set("dir", f"{np.sin(angle)} .2 {-np.cos(angle)}")
        changes["lighting"] = dict(intensity_scale=intensity, angle_radians=angle)
    if family in ("background", "combined"):
        choices = ((.18, .18, .18, 1), (.35, .32, .30, 1), (.24, .28, .27, 1))
        color = choices[int(rng.integers(len(choices)))]
        root.find(".//geom[@name='table_top']").set("rgba", " ".join(map(str, color)))
        changes["table_rgba"] = color
    if family in ("shape", "combined"):
        # Compile alternate collision primitives, not just different colors or scales.
        root.find(".//geom[@name='mug']").attrib.update(type="box", size=".021 .021 .035")
        root.find(".//geom[@name='spoon_bowl']").attrib.update(type="box", size=".017 .024 .0035")
        changes["shape"] = {"mug": "cuboid replaces cylinder", "spoon_bowl": "box replaces ellipsoid"}
    if family in ("physical", "combined"):
        mass, friction = float(rng.uniform(.8, 1.2)), float(rng.uniform(.8, 1.2))
        for name in ("plate", "mug", "spoon", "fork"):
            for geom in root.find(f".//body[@name='{name}']").findall("geom"):
                geom.set("mass", str(float(geom.get("mass", ".001")) * mass))
                values = np.fromstring(geom.get("friction", "1 .005 .0001"), sep=" ") * friction
                geom.set("friction", " ".join(map(str, values)))
        changes["physical"] = dict(mass_scale=mass, friction_scale=friction)
    # Mesh references must retain their source directory when the XML is frozen elsewhere.
    compiler = root.find("compiler")
    compiler.set("meshdir", str((base.parent / compiler.get("meshdir", "")).resolve()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.seeds)) != len(args.seeds) or len(set(args.families)) != len(args.families):
        parser.error("Use a fresh output directory and distinct seeds/families")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative")
    args.output.mkdir(parents=True)
    base = run_path(str(ROOT / "scripts/build_full_scene.py"))["build_full_scene"]()
    sources = [*sorted((ROOT / "src/mise").rglob("*.py")), Path(__file__).resolve(),
               ROOT / "scripts/evaluate_full_task.py", ROOT / "configs/full_evaluator.yaml"]
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        saved = args.output / "source" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(source.read_bytes())
        hashes[str(relative)] = file_hash(saved)
    cases = []
    for family in args.families:
        for seed in args.seeds:
            scene = args.output / "scenes" / f"{family}-{seed}.xml"
            changes = variant(base, scene, seed, family)
            cases.append(dict(family=family, seed=seed, scene=str(scene), changes=changes, scene_sha256=file_hash(scene)))
    manifest = dict(schema_version="mise.robustness.v1", created_at=utc_now(), cases=cases,
                    source_sha256=hashes, hardware=hardware_identity(),
                    scope="deterministic_full_task_stress_matrix", learned_policy=False,
                    limits="Bounded primitive shape, lighting, flat background, and physical variations; fixed fiducial colors; no real-world or arbitrary-shape claim.")
    atomic_json(args.output / "manifest.json", manifest)
    results = []
    for case in cases:
        started = time.monotonic()
        try:
            result = evaluate(case["seed"], scene_path=Path(case["scene"]))
        except Exception as exc:
            result = dict(seed=case["seed"], success=False, failure=f"{type(exc).__name__}: {exc}",
                          wall_seconds=time.monotonic()-started)
        result.update(family=case["family"], scene_sha256=case["scene_sha256"])
        results.append(result)
        atomic_json(args.output / f'{case["family"]}-{case["seed"]}.json', result)
        report = dict(manifest_sha256=file_hash(args.output / "manifest.json"), results=results,
                      successes=sum(row["success"] for row in results), total=len(results), predeclared=len(cases),
                      complete=len(results)==len(cases),
                      by_family={name: dict(successes=sum(row["success"] for row in results if row["family"]==name),
                                            total=sum(row["family"]==name for row in results)) for name in args.families})
        atomic_json(args.output / "results.json", report)
        print(f'{case["family"]} seed={case["seed"]} success={result["success"]} failure={result.get("failure")}', flush=True)
    if not all(row["success"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
