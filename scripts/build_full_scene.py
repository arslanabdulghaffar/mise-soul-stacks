"""Build the shared passive scene for the complete bimanual table-setting task.

The geometry is intentionally simple and visible. Nothing attaches an object to
a gripper and only the twelve SO-101 joints are actuated.
"""
from pathlib import Path
from runpy import run_path
from xml.etree import ElementTree as ET

import numpy as np

from mise.scene_io import write_scene
from mise.sim import ROOT

OUTPUT = ROOT / "assets/generated/mise_full_task.xml"


def _set_vector(element: ET.Element, attribute: str, index: int, value: float) -> None:
    vector = element.get(attribute, "").split()
    vector[index] = str(value)
    element.set(attribute, " ".join(vector))


def build_full_scene() -> Path:
    contact_path = run_path(str(ROOT / "scripts/build_contact_scene.py"))["build_contact_scene"]()
    root = ET.parse(contact_path).getroot()
    root.set("model", "mise_complete_table_setting")
    world = root.find("worldbody")

    root.find(".//body[@name='arm_a_base']").attrib.update(pos="-.20 .14 .88", quat="1 0 0 0")
    root.find(".//geom[@name='pedestal_a']").set("pos", "-.20 .14 .828")
    root.find(".//body[@name='arm_b_base']").attrib.update(pos=".40 .10 .88", quat="0 0 0 1")
    root.find(".//geom[@name='pedestal_b']").set("pos", ".40 .10 .828")

    drawer_case = root.find(".//body[@name='drawer_case']")
    drawer_case.attrib.update(pos=".18 .325 .81", quat="0 0 0 1")
    drawer_joint = root.find(".//joint[@name='drawer_joint']")
    drawer_joint.attrib.update(damping="1", frictionloss=".05")
    for name in ("drawer_case_back", "drawer_case_lid"):
        _set_vector(root.find(f".//geom[@name='{name}']"), "size", 1, .09)
    for name, sign in (("drawer_case_side_left", 1), ("drawer_case_side_right", -1)):
        geom = root.find(f".//geom[@name='{name}']")
        _set_vector(geom, "pos", 1, sign * .084)
        _set_vector(geom, "size", 1, .006)
    lid = root.find(".//geom[@name='drawer_case_lid']")
    lid.set("pos", "0 0 .086")
    for name in ("drawer_case_back", "drawer_case_side_left", "drawer_case_side_right"):
        geom = root.find(f".//geom[@name='{name}']")
        _set_vector(geom, "pos", 2, .042)
        _set_vector(geom, "size", 2, .042)
    for name in ("drawer_bottom", "drawer_front", "drawer_back"):
        _set_vector(root.find(f".//geom[@name='{name}']"), "size", 1, .070)
    for name, sign in (("drawer_left", 1), ("drawer_right", -1)):
        geom = root.find(f".//geom[@name='{name}']")
        _set_vector(geom, "pos", 1, sign * .070)
        _set_vector(geom, "size", 1, .004)
    root.find(".//geom[@name='drawer_front']").attrib.update(pos=".14 0 .018", size=".006 .070 .024")
    handle = root.find(".//geom[@name='drawer_handle']")
    handle.attrib.update(type="cylinder", pos=".23 0 .06", size=".022 .020",
                         friction="2 .02 .002", condim="4", rgba=".95 .72 .03 1")
    ET.SubElement(root.find(".//body[@name='drawer']"), "geom",
                  name="drawer_handle_stem", type="box", pos=".185 0 .042",
                  size=".045 .007 .006", mass=".01", rgba=".08 .08 .08 1")

    for arm in "ab":
        root.find(f".//geom[@name='arm_{arm}_moving_pad']").set("quat", "1 0 0 0")
        ET.SubElement(root.find(f".//body[@name='arm_{arm}_gripper']"), "site",
                      name=f"arm_{arm}_narrow_tcp", pos=".004 0 -.088", size=".002", group="3")

    for name, y, color in (("spoon", .35, ".92 .08 .04 1"),
                           ("fork", .30, ".05 .82 .13 1")):
        body = root.find(f".//body[@name='{name}']")
        body.attrib.update(pos=f".16 {y} .841", quat="-.7071067812 0 0 .7071067812")
        body.find(f"geom[@name='{name}']").attrib.update(size=".008 .060 .006", rgba=color)
        # Keep heads metallic so the vivid handle centroid is a stable RGB grasp target.
        for geom in body.findall("geom"):
            if geom.get("name") != name:
                geom.set("rgba", ".48 .52 .56 1")
        lane = -.025 if name == "spoon" else .025
        for index, x in enumerate((.075, -.065)):
            ET.SubElement(root.find(".//body[@name='drawer']"), "geom",
                          name=f"{name}_rack_{index}", type="box", pos=f"{x} {lane} .015",
                          size=".006 .012 .009", mass=".002", rgba=".4 .28 .15 1",
                          friction="1 .005 .0001")

    mug = root.find(".//body[@name='mug']")
    mug.set("pos", ".22 -.05 .816")
    plate = root.find(".//body[@name='plate']")
    plate.set("pos", "-.04 -.12 .789")
    plate.find("geom").attrib.update(size=".08 .004", mass=".040", rgba=".94 .94 .90 1")
    for index in range(24):
        theta = 2 * np.pi * index / 24
        ET.SubElement(plate, "geom", name=f"plate_rim_{index}", type="box",
                      pos=f"{.076 * np.cos(theta)} {.076 * np.sin(theta)} .015",
                      size=".011 .004 .015", euler=f"0 0 {theta + np.pi / 2}",
                      mass=".0005", rgba=".94 .94 .90 1", friction="1.5 .01 .001")

    # Replace legacy goal sites with scene-specific heights and distinct colors.
    for site in list(world.findall("site")):
        if site.get("name", "").startswith("goal_"):
            world.remove(site)
    # Markers avoid the four fiducial colors so perception cannot mistake a
    # desired location for an object that never arrived there.
    for name, xy in (
        ("plate_center", (.05, 0)),
        ("fork_left", (-.15, 0)),
        ("spoon_right", (.25, 0)),
        ("mug_upper_right", (.25, .20)),
    ):
        ET.SubElement(world, "site", name=f"goal_{name}", type="cylinder",
                      pos=f"{xy[0]} {xy[1]} .781", size=".022 .0005", rgba=".95 .72 .03 .5")

    ET.indent(root)
    return write_scene(ET.ElementTree(root), OUTPUT)


if __name__ == "__main__":
    print(build_full_scene())
