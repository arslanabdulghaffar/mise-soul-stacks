"""Passive drawer with a graspable knob for the physical arm-A skill.

The handle, tray and case are simplified rigid geometry. Only the twelve robot
joints are actuated; this is a separate scene from the archived mug dataset.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
from runpy import run_path

from mise.sim import ROOT
from mise.scene_io import write_scene


def build_drawer_scene() -> Path:
    contact = run_path(str(ROOT / 'scripts/build_contact_scene.py'))['build_contact_scene']()
    root = ET.parse(contact).getroot()
    root.set('model', 'mise_physical_drawer')
    root.find(".//body[@name='arm_a_base']").set('pos', '-.15 .18 .88')
    root.find(".//body[@name='arm_a_base']").set('quat', '.9659258263 0 0 .2588190451')
    root.find(".//geom[@name='pedestal_a']").set('pos', '-.15 .18 .828')
    case = root.find(".//body[@name='drawer_case']")
    case.set('pos', '.18 .325 .81')
    case.set('quat', '0 0 0 1')
    joint = root.find(".//joint[@name='drawer_joint']")
    joint.set('damping', '1')
    joint.set('frictionloss', '.05')
    for name in ('drawer_case_back', 'drawer_case_lid'):
        geom = root.find(f".//geom[@name='{name}']")
        size = geom.get('size').split(); size[1] = '.09'
        geom.set('size', ' '.join(size))
    for name, sign in [('drawer_case_side_left', 1), ('drawer_case_side_right', -1)]:
        geom = root.find(f".//geom[@name='{name}']")
        pos = geom.get('pos').split(); pos[1] = str(sign * .084)
        size = geom.get('size').split(); size[1] = '.006'
        geom.set('pos', ' '.join(pos)); geom.set('size', ' '.join(size))
    for name in ('drawer_bottom', 'drawer_front', 'drawer_back'):
        geom = root.find(f".//geom[@name='{name}']")
        size = geom.get('size').split(); size[1] = '.070'
        geom.set('size', ' '.join(size))
    for name, sign in [('drawer_left', 1), ('drawer_right', -1)]:
        geom = root.find(f".//geom[@name='{name}']")
        pos = geom.get('pos').split(); pos[1] = str(sign * .070)
        size = geom.get('size').split(); size[1] = '.004'
        geom.set('pos', ' '.join(pos)); geom.set('size', ' '.join(size))
    handle = root.find(".//geom[@name='drawer_handle']")
    handle.attrib.update(type='cylinder', pos='.23 0 .06', size='.022 .020',
                         friction='2 .02 .002', condim='4', rgba='.95 .75 .03 1')
    ET.SubElement(root.find(".//body[@name='drawer']"), 'geom',
                  name='drawer_handle_stem', type='box', pos='.185 0 .042',
                  size='.045 .007 .006', mass='.01', rgba='.08 .08 .08 1')
    for name, y in [('spoon', .35), ('fork', .30)]:
        body = root.find(f".//body[@name='{name}']")
        body.set('pos', f'.20 {y} .8205')
        body.set('quat', '-.7071067812 0 0 .7071067812')
    output = ROOT / 'assets/generated/mise_drawer_contact.xml'
    ET.indent(root)
    return write_scene(ET.ElementTree(root), output)


if __name__ == '__main__':
    print(build_drawer_scene())
