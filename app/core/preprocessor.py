"""
Photo Preprocessor
──────────────────────
Converts a raw Photo image into a clean binary mask suitable for
feature extraction.  Pipeline:

    1. Grayscale conversion
    2. Noise reduction   (Gaussian blur + Non-Local Means denoising)
    3. Adaptive thresholding  (binary inversion)
    4. Morphological cleanup  (remove small artifacts)
    5. Size normalization      (resize to standard width, preserve aspect ratio)
"""

import cv2
import numpy as np
from typing import Dict

from app.config import (
    GAUSSIAN_KERNEL,
    DENOISE_H,
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    MORPH_KERNEL_SIZE,
    NORMALIZED_WIDTH,
)
from app.utils.image_utils import resize_preserve_aspect


class SignaturePreprocessor:
    """Transforms a raw signature image into a clean binary mask."""

    def __init__(
        self,
        gaussian_kernel: tuple = GAUSSIAN_KERNEL,
        denoise_h: int = DENOISE_H,
        adaptive_block: int = ADAPTIVE_BLOCK_SIZE,
        adaptive_c: int = ADAPTIVE_C,
        morph_kernel: tuple = MORPH_KERNEL_SIZE,
        target_width: int = NORMALIZED_WIDTH,
    ):
        self.gaussian_kernel = gaussian_kernel
        self.denoise_h = denoise_h
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.morph_kernel = morph_kernel
        self.target_width = target_width

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run the full preprocessing pipeline.

        Args:
            image: BGR input image.

        Returns:
            Dictionary with intermediate and final results:
                - 'original'   : the input image
                - 'grayscale'  : after grayscale conversion
                - 'denoised'   : after noise reduction
                - 'binary'     : after adaptive thresholding
                - 'cleaned'    : after morphological cleanup
                - 'normalized' : after size normalization (final output)
        """
        steps: Dict[str, np.ndarray] = {"original": image.copy()}

        # 1. Grayscale
        gray = self._to_grayscale(image)
        steps["grayscale"] = gray

        # 2. Noise reduction
        denoised = self._denoise(gray)
        steps["denoised"] = denoised

        # 3. Adaptive thresholding → binary mask (white ink on black bg)
        binary = self._threshold(denoised)
        steps["binary"] = binary

        # 4. Morphological cleanup
        cleaned = self._morphological_clean(binary)
        steps["cleaned"] = cleaned

        # 5. Normalize size
        normalized = self._normalize_size(cleaned)
        steps["normalized"] = normalized

        return steps

    # ── Private Helpers ────────────────────────────────────────────────────

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        # Gaussian blur for initial smoothing
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, 0)
        # Non-local means denoising for preserving edges
        denoised = cv2.fastNlMeansDenoising(blurred, None, self.denoise_h, 7, 21)
        return denoised

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        # Adaptive threshold produces white foreground (signature) on black background
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block,
            self.adaptive_c,
        )
        return binary

    def _morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel)
        # Close small gaps in strokes
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Remove tiny isolated noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        return opened

    def _normalize_size(self, binary: np.ndarray) -> np.ndarray:
        return resize_preserve_aspect(binary, self.target_width)
