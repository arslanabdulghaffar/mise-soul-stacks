"""The single RGB/proprioception preprocessing path used by train and runtime."""
from __future__ import annotations
import numpy as np
from PIL import Image

CAMERA_KEYS = ('observation.images.top', 'observation.images.wrist_a', 'observation.images.wrist_b')
JOINT_ORDER = tuple(f'{arm}.{joint}' for arm in ('A', 'B') for joint in ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'))
RGB_MEAN = np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None]
RGB_STD = np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]


def preprocess_images(images, image_size: int):
    views = []
    for image in images:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError('Camera frames must be HWC RGB uint8 arrays')
        resized = np.asarray(Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32)
        views.append(resized.transpose(2, 0, 1) / 255.)
    return (np.stack(views) - RGB_MEAN) / RGB_STD


def normalize_state(state, statistics):
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != 12 or not np.isfinite(state).all():
        raise ValueError('Robot state requires twelve finite joint angles in radians')
    return (state - np.asarray(statistics['state_mean'], dtype=np.float32)) / np.asarray(statistics['state_std'], dtype=np.float32)


def denormalize_actions(actions, statistics):
    return np.asarray(actions, dtype=np.float32) * np.asarray(statistics['action_std'], dtype=np.float32) + np.asarray(statistics['action_mean'], dtype=np.float32)
