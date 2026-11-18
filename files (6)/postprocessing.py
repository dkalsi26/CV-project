

import cv2
import numpy as np
from PIL import Image


def gray_world_white_balance(image_bgr):
    """Remove a global colour cast with gray-world channel gains."""
    image = image_bgr.astype(np.float32)
    channel_means = image.mean(axis=(0, 1))
    gray_mean = float(channel_means.mean())
    gains = gray_mean / (channel_means + 1e-6)
    balanced = image * gains.reshape(1, 1, 3)
    return np.rint(np.clip(balanced, 0, 255)).astype(np.uint8)


def gamma_correction(image_bgr, gamma=1.2):
    """Apply LUT gamma correction; values above 1 brighten the image."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    inverse_gamma = 1.0 / gamma
    table = np.rint(
        ((np.arange(256, dtype=np.float32) / 255.0) ** inverse_gamma) * 255.0
    ).astype(np.uint8)
    return cv2.LUT(image_bgr, table)


def clahe_luminance(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE only to LAB lightness, preserving the colour channels."""
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_grid_size),
    )
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def post_process_bgr(image_bgr, gamma=1.2, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Run the complete fixed post-processing chain on a BGR image."""
    balanced = gray_world_white_balance(image_bgr)
    gamma_corrected = gamma_correction(balanced, gamma=gamma)
    return clahe_luminance(
        gamma_corrected,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )


def post_process_pil(image, gamma=1.2, clip_limit=2.0, tile_grid_size=(8, 8)):
    """PIL/RGB adapter used by the comparison pipeline."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result_bgr = post_process_bgr(
        bgr,
        gamma=gamma,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb, mode="RGB")
