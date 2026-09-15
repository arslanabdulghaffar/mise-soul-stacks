"""Bounded RGB color localization with a calibrated overhead camera.

No object poses, segmentation IDs, depth buffers or simulator body positions are
used. This intentionally supports one known blue cylinder on this tabletop;
it is not general object recognition or a learned perception model.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Detection:
    xy: tuple[float, float]
    pixels: int
    centroid: tuple[float, float]
    axis_error_degrees: float | None = None
    heading_radians: float | None = None


def locate_mug(rgb: np.ndarray, *, camera_height: float = 1.75, fovy: float = 58., object_top: float = .85) -> Detection:
    image = np.asarray(rgb, dtype=float)
    mask = (image[:, :, 2] > 1.5 * image[:, :, 0]) & (image[:, :, 2] > 1.5 * image[:, :, 1]) & (image[:, :, 2] > 80)
    # Largest connected component rejects blue fringes on the violet robot.
    visited = np.zeros(mask.shape, dtype=bool)
    biggest = []
    height, width = mask.shape
    for row, column in zip(*np.where(mask)):
        if visited[row, column]:
            continue
        stack, component = [(row, column)], []
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
    if len(biggest) < 35:
        raise ValueError('The blue mug is not sufficiently visible in the overhead camera.')
    y, x = np.mean(biggest, axis=0)
    scale = 2 * (camera_height - object_top) * np.tan(np.deg2rad(fovy / 2)) / height
    xy = ((float(x) - (width - 1) / 2) * scale, ((height - 1) / 2 - float(y)) * scale)
    return Detection(xy, len(biggest), (float(x), float(y)))
