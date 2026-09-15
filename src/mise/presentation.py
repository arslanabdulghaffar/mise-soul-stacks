"""Display-only cameras; never used by a controller or its calibrated observer."""

from __future__ import annotations

import numpy as np

CAMERAS = ("top", "wrist_a", "wrist_b", "side_a", "side_b")
PROFILES = {
    "economy": {"size": 256, "inset_size": 160, "hz": 2, "wall_guard_s": 300},
    "balanced": {"size": 384, "inset_size": 256, "hz": 3, "wall_guard_s": 900},
    "detail": {"size": 720, "inset_size": 384, "hz": 2, "wall_guard_s": 900},
}


def capture_settings(profile: str, control_hz: int = 30) -> dict:
    selected = PROFILES[profile]
    intervals = {name: max(1, round(control_hz / hz))
                 for name, hz in ((camera, selected["hz"]) for camera in CAMERAS)}
    return {"profile": profile, "width": selected["size"], "height": selected["size"],
            "capture_every": intervals,
            "camera_dimensions": {camera: {"width": selected["size"] if camera == "top" else selected["inset_size"],
                                            "height": selected["size"] if camera == "top" else selected["inset_size"]} for camera in CAMERAS},
            "synchronized_capture": True,
            "camera_capture_hz": {name: control_hz / interval for name, interval in intervals.items()},
            "wall_guard_s": selected["wall_guard_s"], "interpolated": False,
            "timebase": "simulation_seconds", "controller_camera_size": 256}


class PresentationRenderer:
    def __init__(self, env, size: int, *, camera_dimensions: dict | None = None):
        self.env = env
        # Offscreen buffer capacity is a visual setting, not camera calibration.
        env.model.vis.global_.offwidth = max(size, env.model.vis.global_.offwidth)
        env.model.vis.global_.offheight = max(size, env.model.vis.global_.offheight)
        self.size = size
        self.dimensions = camera_dimensions or {camera: {"width": size, "height": size} for camera in CAMERAS}
        from mujoco.gl_context import GLContext
        self.context = GLContext(size, size)
        self.context.make_current()
        samples = env.model.vis.quality.offsamples
        try:
            env.model.vis.quality.offsamples = 0
            self.render_context = env._mj.MjrContext(env.model, env._mj.mjtFontScale.mjFONTSCALE_150.value)
        finally:
            env.model.vis.quality.offsamples = samples
        env._mj.mjr_setBuffer(env._mj.mjtFramebuffer.mjFB_OFFSCREEN, self.render_context)
        self.scene = env._mj.MjvScene(env.model, maxgeom=10000)
        self.scene.flags[env._mj.mjtRndFlag.mjRND_SHADOW] = False
        self.sides = {}
        for name, x, azimuth in (("side_a", -.16, 135), ("side_b", .16, 45)):
            camera = env._mj.MjvCamera()
            camera.type = env._mj.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [x, 0, .86]
            camera.distance = .95
            camera.azimuth = azimuth
            camera.elevation = -32
            self.sides[name] = camera

    def render(self, camera: str):
        if camera not in CAMERAS:
            raise ValueError(f"Unknown display camera: {camera}")
        dimensions = self.dimensions[camera]
        selected = self.sides.get(camera)
        if selected is None:
            selected = self.env._mj.MjvCamera()
            selected.type = self.env._mj.mjtCamera.mjCAMERA_FIXED
            selected.fixedcamid = self.env.model.camera(camera).id
        self.context.make_current()
        self.env._mj.mjv_updateScene(self.env.model, self.env.data, self.env._render_options,
                                    None, selected, self.env._mj.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = self.env._mj.MjrRect(0, 0, dimensions["width"], dimensions["height"])
        self.env._mj.mjr_render(viewport, self.scene, self.render_context)
        frame = np.empty((dimensions["height"], dimensions["width"], 3), dtype=np.uint8)
        self.env._mj.mjr_readPixels(frame, None, viewport, self.render_context)
        return np.flipud(frame).copy()

    def close(self):
        self.context.make_current()
        self.render_context.free()
        self.context.free()
