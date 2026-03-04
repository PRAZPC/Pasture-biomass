from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def extract_image_features(image_path: Path) -> np.ndarray:
    """
    Handcrafted image features that work without deep-model dependencies.
    Returns shape (n_features,).
    """
    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize((256, 256))
        gray = img.convert("L").resize((256, 256))

    rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0
    gray_arr = np.asarray(gray, dtype=np.float32) / 255.0

    features: list[float] = []

    # RGB channel statistics
    for channel_idx in range(3):
        channel = rgb_arr[:, :, channel_idx]
        features.extend(
            [
                float(channel.mean()),
                float(channel.std()),
                float(channel.min()),
                float(channel.max()),
                float(np.percentile(channel, 25)),
                float(np.percentile(channel, 50)),
                float(np.percentile(channel, 75)),
            ]
        )

    # Grayscale and texture-like features
    features.extend(
        [
            float(gray_arr.mean()),
            float(gray_arr.std()),
            float(np.percentile(gray_arr, 10)),
            float(np.percentile(gray_arr, 90)),
        ]
    )

    # Gradient-based roughness features
    grad_x = np.abs(np.diff(gray_arr, axis=1))
    grad_y = np.abs(np.diff(gray_arr, axis=0))
    features.extend(
        [
            float(grad_x.mean()),
            float(grad_x.std()),
            float(grad_y.mean()),
            float(grad_y.std()),
        ]
    )

    # Color index approximations
    r = rgb_arr[:, :, 0]
    g = rgb_arr[:, :, 1]
    b = rgb_arr[:, :, 2]
    eps = 1e-6
    exg = 2.0 * g - r - b
    ngrdi = (g - r) / (g + r + eps)
    features.extend(
        [
            float(exg.mean()),
            float(exg.std()),
            float(ngrdi.mean()),
            float(ngrdi.std()),
        ]
    )

    return np.asarray(features, dtype=np.float32)
