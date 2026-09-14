"""Display-only cameras; never used by a controller or its calibrated observer."""

from __future__ import annotations

CAMERAS = ("top", "wrist_a", "wrist_b", "side_a", "side_b")
PROFILES = {
    "economy": {"size": 256, "hz": (2, .25, .25, .5, .5), "wall_guard_s": 300},
    "balanced": {"size": 384, "hz": (6, 2, 2, 3, 3), "wall_guard_s": 900},
    "detail": {"size": 720, "hz": (3, 1, 1, 1, 1), "wall_guard_s": 900},
}


def capture_settings(profile: str, control_hz: int = 30) -> dict:
    selected = PROFILES[profile]
    intervals = {name: max(1, round(control_hz / hz))
                 for name, hz in zip(CAMERAS, selected["hz"])}
    return {"profile": profile, "width": selected["size"], "height": selected["size"],
            "capture_every": intervals,
            "camera_capture_hz": {name: control_hz / interval for name, interval in intervals.items()},
            "wall_guard_s": selected["wall_guard_s"], "interpolated": False,
            "timebase": "simulation_seconds", "controller_camera_size": 256}


class PresentationRenderer:
    def __init__(self, env, size: int):
        self.env = env
        # Offscreen buffer capacity is a visual setting, not camera calibration.
        env.model.vis.global_.offwidth = max(size, env.model.vis.global_.offwidth)
        env.model.vis.global_.offheight = max(size, env.model.vis.global_.offheight)
        self.renderer = env._mj.Renderer(env.model, height=size, width=size)
        # Shadow maps are expensive on software GL and obscure small utensils.
        # Only presentation uses this flag; controller rendering stays unchanged.
        self.renderer.scene.flags[env._mj.mjtRndFlag.mjRND_SHADOW] = False
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
        self.renderer.update_scene(self.env.data, camera=self.sides.get(camera, camera),
                                   scene_option=self.env._render_options)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()
