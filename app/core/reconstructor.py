"""
Signature Reconstructor
───────────────────────
Rebuilds a signature on a clean canvas from the extracted features.

Key principle: we never copy pixels from the original image.
Instead, we draw the signature from extracted contours, skeleton,
and measured stroke widths.

Two reconstruction methods are provided:
    1. Contour-based  – fills or draws the smoothed contours
    2. Skeleton-based – draws the thinned centerline with measured thickness

Both methods produce a signature that is structurally equivalent to the
original but genuinely reconstructed from features.
"""

import cv2
import numpy as np
from typing import Dict

from app.config import (
    CANVAS_PADDING,
    DEFAULT_STROKE_THICKNESS,
    ANTI_ALIAS,
    POST_BLUR_KERNEL,
    INK_COLOR,
)
from app.core.extractor import SignatureFeatures


class SignatureReconstructor:
    """Reconstructs a signature from extracted features onto a clean canvas."""

    def __init__(
        self,
        padding: int = CANVAS_PADDING,
        default_thickness: int = DEFAULT_STROKE_THICKNESS,
        anti_alias: bool = ANTI_ALIAS,
        post_blur: tuple = POST_BLUR_KERNEL,
        ink_color: tuple = INK_COLOR,
    ):
        self.padding = padding
        self.default_thickness = default_thickness
        self.line_type = cv2.LINE_AA if anti_alias else cv2.LINE_8
        self.post_blur = post_blur
        self.ink_color = ink_color

    # ── Public API ─────────────────────────────────────────────────────────

    def reconstruct(self, features: SignatureFeatures) -> Dict[str, np.ndarray]:
        """
        Run all reconstruction methods and return results.

        Args:
            features: Extracted SignatureFeatures.

        Returns:
            Dictionary with:
                - 'contour_filled'    : filled-contour reconstruction
                - 'contour_outline'   : outline-contour reconstruction
                - 'skeleton_based'    : skeleton + stroke-width reconstruction
                - 'final'             : best combined result
        """
        results: Dict[str, np.ndarray] = {}

        # Method 1: Filled contours
        results["contour_filled"] = self._reconstruct_contour_filled(features)

        # Method 2: Contour outlines with estimated stroke width
        results["contour_outline"] = self._reconstruct_contour_outline(features)

        # Method 3: Skeleton-based with variable thickness
        results["skeleton_based"] = self._reconstruct_skeleton(features)

        # Final: combine contour fill + light post-processing
        results["final"] = self._produce_final(features)

        return results

    # ── Reconstruction Methods ─────────────────────────────────────────────

    def _reconstruct_contour_filled(self, features: SignatureFeatures) -> np.ndarray:
        """
        Draw filled contours to produce solid strokes.
        """
        canvas, offset = self._create_canvas(features)
        shifted = self._shift_contours(features.smoothed_contours, offset)

        # Fill the contours with ink color
        cv2.drawContours(canvas, shifted, -1, self.ink_color, cv2.FILLED, self.line_type)

        return self._post_process(canvas)

    def _reconstruct_contour_outline(self, features: SignatureFeatures) -> np.ndarray:
        """
        Draw contour outlines with the estimated average stroke thickness.
        """
        canvas, offset = self._create_canvas(features)
        shifted = self._shift_contours(features.smoothed_contours, offset)

        thickness = max(1, int(round(features.stroke_width_mean)))
        cv2.drawContours(canvas, shifted, -1, self.ink_color, thickness, self.line_type)

        return self._post_process(canvas)

    def _reconstruct_skeleton(self, features: SignatureFeatures) -> np.ndarray:
        """
        Draw the skeleton with variable stroke width derived from the
        distance transform, producing natural thickness variation.
        """
        canvas, offset = self._create_canvas(features)
        x_off, y_off = offset

        # Get skeleton points and their corresponding stroke widths
        skel = features.skeleton
        dist_map = features.stroke_width_map

        # Find all skeleton pixels
        ys, xs = np.where(skel > 0)

        if len(xs) == 0:
            # Fallback to contour method if skeleton is empty
            return self._reconstruct_contour_filled(features)

        # Draw circles at each skeleton point with radius from distance transform
        for px, py in zip(xs, ys):
            radius = max(1, int(dist_map[py, px]))
            cv2.circle(
                canvas,
                (px + x_off, py + y_off),
                radius,
                self.ink_color,
                -1,
                self.line_type,
            )

        return self._post_process(canvas)

    def _produce_final(self, features: SignatureFeatures) -> np.ndarray:
        """
        Produce the best final reconstruction by combining methods.
        Uses filled contours as the primary output with enhanced smoothing.
        """
        canvas, offset = self._create_canvas(features)
        shifted = self._shift_contours(features.smoothed_contours, offset)

        # Draw filled contours
        cv2.drawContours(canvas, shifted, -1, self.ink_color, cv2.FILLED, self.line_type)

        # Also draw outlines for crisp edges
        thickness = max(1, int(round(features.stroke_width_mean * 0.3)))
        cv2.drawContours(canvas, shifted, -1, self.ink_color, thickness, self.line_type)

        # Enhanced post-processing
        result = self._post_process(canvas)

        return result

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _create_canvas(self, features: SignatureFeatures):
        """
        Create a clean white canvas sized to the signature bounding box
        plus padding.  Returns the canvas and the (x, y) offset.
        """
        bx, by, bw, bh = features.bounding_rect
        canvas_w = bw + 2 * self.padding
        canvas_h = bh + 2 * self.padding

        # White canvas (BGR)
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        # Offset to shift contours so they start at (padding, padding)
        x_off = self.padding - bx
        y_off = self.padding - by

        return canvas, (x_off, y_off)

    @staticmethod
    def _shift_contours(contours, offset):
        """Translate all contour points by (x_off, y_off)."""
        x_off, y_off = offset
        return [cnt + np.array([x_off, y_off]) for cnt in contours]

    def _post_process(self, canvas: np.ndarray) -> np.ndarray:
        """Apply subtle Gaussian blur for ink-like softness."""
        if self.post_blur:
            canvas = cv2.GaussianBlur(canvas, self.post_blur, 0)
        return canvas
