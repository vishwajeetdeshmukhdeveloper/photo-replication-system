"""
Image utility helpers for loading, saving, encoding, and resizing images.
"""

import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Path to the image file.

    Returns:
        BGR image as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """
    Save an image to disk.

    Args:
        image: Image as a NumPy array (BGR or grayscale).
        path: Destination file path.

    Returns:
        The resolved Path where the image was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def encode_image_base64(image: np.ndarray, fmt: str = ".png") -> str:
    """
    Encode an image to a base64 string.

    Args:
        image: Image as a NumPy array.
        fmt: Output format extension (e.g. '.png', '.jpg').

    Returns:
        Base64-encoded string of the image.
    """
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise ValueError(f"Failed to encode image to {fmt}")
    return base64.b64encode(buffer).decode("utf-8")


def decode_image_base64(data: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to a NumPy array.

    Args:
        data: Base64 string representing the image.

    Returns:
        BGR image as a NumPy array.
    """
    buffer = base64.b64decode(data)
    arr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image data")
    return img


def resize_preserve_aspect(
    image: np.ndarray,
    target_width: int,
    max_height: Optional[int] = None,
) -> np.ndarray:
    """
    Resize an image to a target width while preserving the aspect ratio.

    Args:
        image: Input image.
        target_width: Desired width in pixels.
        max_height: Optional maximum height; if the computed height exceeds
                     this, the image is further scaled down.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)

    if max_height and new_h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = max_height

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def bytes_to_cv2(data: bytes) -> np.ndarray:
    """
    Convert raw file bytes to an OpenCV image.

    Args:
        data: Raw bytes of an image file.

    Returns:
        BGR image as a NumPy array.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode the provided image bytes")
    return img
