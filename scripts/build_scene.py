"""Compose two namespace-safe SO-101 arms and a physical table-setting scene.

MuJoCo includes cannot be instantiated twice without duplicate names. This builder
copies the official robot's XML tree and prefixes all name-bearing entities and
references per arm, while retaining the original mesh files through a stable
submodule path.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from mise.scene_io import write_scene

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.xml"
OUTPUT = ROOT / "assets" / "generated" / "mise_bimanual.xml"


def _prefix_arm(source: ET.Element, arm: str, position: str) -> tuple[ET.Element, list[ET.Element], list[ET.Element]]:
    """Return namespaced body, assets, and actuators from the official model."""

    prefix = f"arm_{arm}_"
    source_assets = source.find("asset")
    source_world = source.find("worldbody")
    source_actuators = source.find("actuator")
    assert source_assets is not None and source_world is not None and source_actuators is not None
    # The official file lets MuJoCo derive mesh names from file names.  We make
    # those names explicit before namespacing so each arm can own an asset entry.
    source_assets = deepcopy(source_assets)
    for item in source_assets.findall("mesh"):
        if item.get("name") is None and item.get("file"):
            item.set("name", Path(item.attrib["file"]).stem)
    body = deepcopy(next(iter(source_world)))
    assets = [deepcopy(item) for item in source_assets]
    actuators = [deepcopy(item) for item in source_actuators]
    names: set[str] = set()
    for root in [body, *assets, *actuators]:
        for element in root.iter():
            if "name" in element.attrib:
                names.add(element.attrib["name"])
    # `class` values are intentionally shared: their defaults are defined once.
    for root in [body, *assets, *actuators]:
        for element in root.iter():
            for attribute in ("name", "joint", "mesh", "material", "site", "tendon"):
                value = element.get(attribute)
                if value in names:
                    element.set(attribute, prefix + value)
    body.set("name", prefix + "base")
    body.set("pos", position)
    return body, assets, actuators


def build_scene() -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError("SO-101 submodule is missing; run `git submodule update --init --recursive`.")
    source = ET.parse(SOURCE).getroot()
    root = ET.Element("mujoco", {"model": "mise_bimanual"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "../../vendor/SO-ARM100/Simulation/SO101/assets", "autolimits": "true"})
    option = ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81", "integrator": "implicitfast"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "256", "offheight": "256"})
    default = ET.SubElement(root, "default")
    arm_default = ET.SubElement(default, "default", {"class": "so101_new_calib"})
    ET.SubElement(arm_default, "joint", {"damping": "1", "frictionloss": "0.1", "armature": "0.005"})
    ET.SubElement(arm_default, "position", {"kp": "50"})
    visual_default = ET.SubElement(arm_default, "default", {"class": "visual"})
    ET.SubElement(visual_default, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"})
    collision_default = ET.SubElement(arm_default, "default", {"class": "collision"})
    ET.SubElement(collision_default, "geom", {"group": "3"})
    servo_default = ET.SubElement(default, "default", {"class": "sts3215"})
    ET.SubElement(servo_default, "joint", {"damping": "0.60", "frictionloss": "0.052", "armature": "0.028"})
    ET.SubElement(servo_default, "position", {"kp": "998.22", "kv": "2.731", "forcerange": "-2.94 2.94"})

    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    actuator = ET.SubElement(root, "actuator")
    for arm, position in (("a", "0.34 0.33 0.78"), ("b", "0.34 -0.33 0.78")):
        body, assets, actuators = _prefix_arm(source, arm, position)
        # Mirror the second base so its nominal reach points into the workspace.
        # Both contact geometry and shared reach still require expert validation.
        if arm == "b":
            body.set("quat", "0 0 0 1")
        for material in assets:
            if material.tag == "material":
                rgba = [float(v) for v in material.get("rgba", "0 0 0 1").split()]
                if rgba[0] > 0.8 and rgba[1] > 0.6 and rgba[2] < 0.3:
                    material.set("rgba", "0.45 0.83 0.89 1" if arm == "a" else "0.64 0.58 0.94 1")
        # Upstream removes base collisions. This explicitly named primitive is
        # a conservative approximation; collision calibration remains pending.
        ET.SubElement(body, "geom", {"name": f"arm_{arm}_base_collision", "type": "box", "pos": "0.005 0 0.025", "size": "0.035 0.035 0.025", "rgba": "0.2 0.2 0.2 0.2", "group": "3"})
        # Wrist cameras are mounted on the gripper body from the official kinematic tree.
        for candidate in body.iter("body"):
            if candidate.get("name") == f"arm_{arm}_gripper":
                ET.SubElement(candidate, "camera", {"name": f"wrist_{arm}", "pos": "0 0 -0.11", "xyaxes": "0 -1 0 1 0 0", "fovy": "70"})
                break
        world.append(body)
        for item in assets:
            asset.append(item)
        for item in actuators:
            actuator.append(item)

    ET.SubElement(world, "light", {"name": "key", "pos": "0 0 2.6", "dir": "0 0 -1", "directional": "true", "diffuse": "0.9 0.9 0.9"})
    ET.SubElement(world, "camera", {"name": "top", "pos": "0 0 2.25", "xyaxes": "1 0 0 0 1 0", "fovy": "58"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "3 3 0.1", "rgba": "0.13 0.15 0.18 1"})
    table = ET.SubElement(world, "body", {"name": "table", "pos": "0 0 0.72"})
    ET.SubElement(table, "geom", {"name": "table_top", "type": "box", "size": "0.62 0.52 0.06", "rgba": "0.40 0.23 0.12 1", "friction": "1.2 0.005 0.0001"})
    for x in (-0.53, 0.53):
        for y in (-0.43, 0.43):
            ET.SubElement(table, "geom", {"type": "box", "pos": f"{x} {y} -0.40", "size": "0.04 0.04 0.38", "rgba": "0.20 0.12 0.07 1"})

    drawer_case = ET.SubElement(world, "body", {"name": "drawer_case", "pos": "-0.34 0.27 0.81"})
    # Separate walls leave a real cavity and clearance for the sliding tray.
    for name, pos, size in (
        ("back", "-0.155 0 0.04", "0.008 0.145 0.07"),
        ("side_left", "0 0.14 0.04", "0.155 0.008 0.07"),
        ("side_right", "0 -0.14 0.04", "0.155 0.008 0.07"),
        ("lid", "0 0 0.11", "0.155 0.145 0.008"),
    ):
        ET.SubElement(drawer_case, "geom", {"name": f"drawer_case_{name}", "type": "box", "pos": pos, "size": size, "rgba": "0.22 0.25 0.30 1"})
    drawer = ET.SubElement(drawer_case, "body", {"name": "drawer"})
    ET.SubElement(drawer, "joint", {"name": "drawer_joint", "type": "slide", "axis": "1 0 0", "range": "0 0.18", "damping": "10"})
    for name, pos, size in (
        ("bottom", "0 0 0", "0.14 0.125 0.006"),
        ("front", "0.14 0 0.032", "0.006 0.125 0.038"),
        ("back", "-0.14 0 0.025", "0.006 0.125 0.025"),
        ("left", "0 0.125 0.025", "0.14 0.006 0.025"),
        ("right", "0 -0.125 0.025", "0.14 0.006 0.025"),
    ):
        ET.SubElement(drawer, "geom", {"name": f"drawer_{name}", "type": "box", "pos": pos, "size": size, "rgba": "0.50 0.34 0.19 1", "mass": "0.12"})
    ET.SubElement(drawer, "geom", {"name": "drawer_handle", "type": "box", "pos": "0.165 0 0.04", "size": "0.012 0.05 0.012", "rgba": "0.08 0.08 0.08 1", "mass": "0.05"})
    # Fast enough for a short-horizon privileged drawer teacher while remaining
    # a position-controlled physical joint rather than a state teleport.
    ET.SubElement(actuator, "position", {"name": "drawer_slide", "joint": "drawer_joint", "kp": "900", "ctrlrange": "0 0.18"})

    _add_object(world, "plate", "cylinder", "0.13 0.018", "-0.025 -0.23 0.798", "0.88 0.88 0.83 1")
    _add_object(world, "mug", "cylinder", "0.045 0.06", "0.23 -0.23 0.84", "0.18 0.42 0.82 1")
    _add_object(world, "bottle", "cylinder", "0.032 0.13", "-0.30 -0.29 0.91", "0.18 0.70 0.42 1")
    _add_utensil(world, "spoon", "-0.395 0.25 0.821")
    _add_utensil(world, "fork", "-0.285 0.25 0.821")
    # Sites mark the evaluator's configured center-error circles; they have no
    # collision or attachment behavior. Keep XML and evaluator goals in sync.
    config = yaml.safe_load((ROOT / "configs" / "evaluator.yaml").read_text())
    for name, region in config["goal_regions"].items():
        x, y = region["center_xy"]
        ET.SubElement(world, "site", {"name": f"goal_{region['name']}", "type": "cylinder", "pos": f"{x} {y} 0.781", "size": f"{region['radius_m']} 0.001", "rgba": "0.25 0.85 0.65 0.45"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    return write_scene(tree, OUTPUT)


def _add_object(world: ET.Element, name: str, shape: str, size: str, pos: str, rgba: str) -> None:
    body = ET.SubElement(world, "body", {"name": name, "pos": pos})
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    ET.SubElement(body, "geom", {"name": name, "type": shape, "size": size, "mass": "0.12", "rgba": rgba, "friction": "1.0 0.005 0.0001"})


def _add_utensil(world: ET.Element, name: str, pos: str) -> None:
    """Free rigid utensils with 6 mm handles; simplified geometry is explicit."""
    body = ET.SubElement(world, "body", {"name": name, "pos": pos})
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    common = {"rgba": "0.72 0.77 0.80 1", "friction": "1.0 0.005 0.0001"}
    ET.SubElement(body, "geom", {**common, "name": name, "type": "box", "size": "0.005 0.060 0.003", "mass": "0.016"})
    if name == "spoon":
        ET.SubElement(body, "geom", {**common, "name": "spoon_bowl", "type": "ellipsoid", "pos": "0 0.073 0", "size": "0.017 0.024 0.0035", "mass": "0.008"})
    else:
        ET.SubElement(body, "geom", {**common, "name": "fork_head", "type": "box", "pos": "0 0.061 0", "size": "0.016 0.010 0.003", "mass": "0.006"})
        for index, x in enumerate((-0.012, -0.004, 0.004, 0.012)):
            ET.SubElement(body, "geom", {**common, "name": f"fork_tine_{index}", "type": "box", "pos": f"{x} 0.084 0", "size": "0.002 0.016 0.003", "mass": "0.001"})


if __name__ == "__main__":
    print(build_scene())
