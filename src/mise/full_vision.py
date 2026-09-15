"""Calibrated RGB localization for the simplified full-task scene.

These functions consume pixels only. Colors are deliberate visual fiducials for
the hackathon baseline, not claims of general object recognition.
"""
from dataclasses import dataclass

import numpy as np

from .contact_vision import Detection


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def _largest_component(mask: np.ndarray) -> list[tuple[int, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    biggest: list[tuple[int, int]] = []
    height, width = mask.shape
    for row, column in zip(*np.where(mask)):
        if visited[row, column]:
            continue
        stack, component = [(int(row), int(column))], []
        visited[row, column] = True
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        if len(component) > len(biggest):
            biggest = component
    return biggest


def locate_color(rgb: np.ndarray, object_name: str, *, object_top: float,
                 bounds: Bounds | None = None, minimum_pixels: int = 20,
                 camera_height: float = 1.75, fovy: float = 58.0,
                 measure_axis: bool = False) -> Detection:
    image = np.asarray(rgb, dtype=float)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB observation must have shape H x W x 3.")
    red, green, blue = (image[:, :, index] for index in range(3))
    if object_name == "spoon":
        mask = (red > 105) & (red > 1.65 * green) & (red > 1.65 * blue)
    elif object_name == "fork":
        mask = (green > 90) & (green > 1.55 * red) & (green > 1.55 * blue)
    elif object_name == "plate":
        mask = (red > 175) & (green > 175) & (blue > 165)
        mask &= np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue)) < 35
    elif object_name == "mug":
        mask = (blue > 80) & (blue > 1.5 * red) & (blue > 1.5 * green)
    elif object_name == "drawer_handle":
        mask = (red > 100) & (green > .75 * red) & (green < 1.05 * red) & (blue < .4 * red)
    else:
        raise ValueError(f"Unknown visual fiducial: {object_name}")
    height, width = mask.shape
    scale = 2 * (camera_height - object_top) * np.tan(np.deg2rad(fovy / 2)) / height
    if bounds is not None:
        rows, columns = np.indices((height, width))
        world_x = (columns - (width - 1) / 2) * scale
        world_y = ((height - 1) / 2 - rows) * scale
        mask &= ((world_x >= bounds.xmin) & (world_x <= bounds.xmax)
                 & (world_y >= bounds.ymin) & (world_y <= bounds.ymax))
    component = _largest_component(mask)
    # A tabletop highlight can join the white plate into an implausibly large
    # region. Tighten brightness only in that case, preserving the calibrated
    # centroid under nominal lighting. The bound allows a maximum 20 cm plate extent,
    # projected with camera calibration; no simulator object state is read.
    if object_name == "plate" and len(component) * scale**2 > .20**2:
        component = _largest_component(mask & (red > 220) & (green > 220) & (blue > 210))
    if len(component) < minimum_pixels:
        raise ValueError(f"The {object_name.replace('_', ' ')} is not sufficiently visible in RGB.")
    row, column = np.mean(component, axis=0)
    xy = ((float(column) - (width - 1) / 2) * scale,
          ((height - 1) / 2 - float(row)) * scale)
    axis_error = None
    if measure_axis:
        centered = np.asarray(component, dtype=float) - np.asarray([row, column])
        covariance = centered.T @ centered
        axis = np.linalg.eigh(covariance)[1][:, -1]
        # An axis has no sign: both handle-up and handle-down are vertical.
        axis_error = float(np.rad2deg(np.arctan2(abs(axis[1]), abs(axis[0]))))
    return Detection(xy, len(component), (float(column), float(row)), axis_error)


def locate_handle(rgb: np.ndarray, *, opened: bool = False) -> Detection:
    xbounds = (-.27, -.17) if opened else (-.08, -.02)
    return locate_color(rgb, "drawer_handle", object_top=.89,
                        bounds=Bounds(*xbounds, .28, .37), minimum_pixels=25)


def locate_plate(rgb: np.ndarray) -> Detection:
    return locate_color(rgb, "plate", object_top=.82,
                        bounds=Bounds(-.14, .14, -.21, .08), minimum_pixels=90)


def locate_utensil(rgb: np.ndarray, name: str, *, in_drawer: bool) -> Detection:
    bounds = Bounds(-.08, .06, .26, .38) if in_drawer else Bounds(-.23, .34, -.06, .23)
    return locate_color(rgb, name, object_top=.85 if in_drawer else .80,
                        bounds=bounds, minimum_pixels=12, measure_axis=True)


def locate_full_mug(rgb: np.ndarray) -> Detection:
    return locate_color(rgb, "mug", object_top=.85,
                        bounds=Bounds(.14, .31, -.12, .27), minimum_pixels=30)
