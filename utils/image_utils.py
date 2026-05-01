"""Image I/O utility functions."""

import os
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load image, returns numpy array in BGR format.

    Args:
        path: Image file path

    Returns:
        np.ndarray in BGR format

    Raises:
        FileNotFoundError: File does not exist
        ValueError: Image read failed
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file does not exist: {path}")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Save image to file.

    Args:
        image: numpy array in BGR format
        path: Save path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Cannot save image to: {path}")


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Scale image proportionally so that the longest side does not exceed max_size.

    If the image is already smaller than max_size, no processing is done.

    Args:
        image: Input image
        max_size: Maximum pixel count for the longest side

    Returns:
        Scaled image
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return image
    scale = max_size / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB.

    Args:
        image: Image in BGR format

    Returns:
        Image in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR.

    Args:
        image: Image in RGB format

    Returns:
        Image in BGR format
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
