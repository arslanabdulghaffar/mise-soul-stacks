"""Reachable contact-baseline scene with an explicit primitive finger-pad model.

The upstream finger meshes are concave; a single convex hull fills part of the
jaw opening. Keep their visual meshes and replace only those collision hulls
with finite frictional pads. No welds, object actuators or grasp constraints.
This is a simulation approximation, not calibrated real-hardware geometry.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from mise.sim import build_scene, ROOT
from mise.scene_io import write_scene

OUTPUT = ROOT / "assets/generated/mise_contact.xml"


def build_contact_scene() -> Path:
    # Regenerate from tracked source so a stale legacy XML cannot enter a run.
    root = ET.parse(build_scene()).getroot()
    root.set('model', 'mise_contact_baseline')
    world = root.find('worldbody')
    for arm, pos in [('a', '-0.40 0.32 0.88'), ('b', '0.28 0.03 0.88')]:
        base = root.find(f".//body[@name='arm_{arm}_base']")
        base.set('pos', pos)
        mount = root.find(f".//geom[@name='arm_{arm}_base_collision']")
        # The old conservative base box intersected its rotating shoulder.
        mount.set('pos', '0 0 .001')
        mount.set('size', '.03 .03 .004')
        xyz = [float(x) for x in pos.split()]
        ET.SubElement(world, 'geom', name=f'pedestal_{arm}', type='box',
                      pos=f'{xyz[0]} {xyz[1]} .828', size='.035 .035 .048', rgba='.18 .22 .28 1')
        gripper = root.find(f".//body[@name='arm_{arm}_gripper']")
        jaw = root.find(f".//body[@name='arm_{arm}_moving_jaw_so101_v1']")
        for body in (gripper, jaw):
            for geom in list(body.findall('geom')):
                if geom.get('class') == 'collision' and any(part in geom.get('mesh', '') for part in ('wrist_roll_follower', 'moving_jaw')):
                    body.remove(geom)
        common = dict(type='box', rgba='.08 .08 .08 1', friction='2.0 .02 .002', condim='4', solref='.004 1', priority='1', solimp='.99 .99 .001')
        ET.SubElement(gripper, 'geom', **common, name=f'arm_{arm}_fixed_pad', pos='-.014 0 -.086', size='.006 .009 .018')
        ET.SubElement(jaw, 'geom', **common, name=f'arm_{arm}_moving_pad', pos='-.0042 -.0656 .0188', size='.008 .018 .009', quat='.978031 0 0 -.208460')
        ET.SubElement(gripper, 'site', name=f'arm_{arm}_tcp', pos='.019 0 -.088', size='.002', group='3')
        root.find(f".//actuator/position[@name='arm_{arm}_gripper']").set('forcerange', '-.5 .5')
        # View the fingertips from above and to the side. The legacy camera was
        # below the grasp point and could look through the mug during placement.
        camera = root.find(f".//camera[@name='wrist_{arm}']")
        camera.set('pos', '.04 -.04 .015')
        z = np.array([.021, -.04, .103])
        z /= np.linalg.norm(z)
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        camera.set('xyaxes', ' '.join(str(float(v)) for v in np.r_[x, y]))
        camera.set('fovy', '80')
    for name, pos in [('mug', '.10 .02 .816'), ('plate', '-.10 -.25 .798'), ('bottle', '-.4 -.3 .91')]:
        root.find(f".//body[@name='{name}']").set('pos', pos)
    mug = root.find(".//geom[@name='mug']")
    mug.attrib.update(size='.022 .035', mass='.06', friction='1.5 .01 .001', condim='4', rgba='.04 .15 .95 1')
    # An overhead calibrated camera keeps the entire tabletop in view.
    root.find(".//camera[@name='top']").set('pos', '0 0 1.75')
    # Passive marker below the mug goal. Sites have no collision geometry.
    ET.SubElement(world, 'site', name='mug_goal_contact', type='cylinder', pos='.25 .20 .781',
                  size='.027 .0005', rgba='.95 .7 .15 .6')
    # A contact run has no independent drawer drive. Preserve its passive joint
    # and other objects for context and the separate full-task evaluator.
    actuator = root.find('actuator')
    actuator.remove(actuator.find("position[@name='drawer_slide']"))
    ET.indent(root)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    return write_scene(ET.ElementTree(root), OUTPUT)


if __name__ == '__main__':
    print(build_contact_scene())
