"""
Feature Extractor
─────────────────
Extracts structural features from a preprocessed binary signature mask:

    • Contours (external + internal hierarchy)
    • Bounding box of the signature region
    • Stroke width estimation via distance transform
    • Skeleton (single-pixel centerline via Zhang-Suen thinning)

All extracted data is returned as a structured dictionary so that the
Reconstructor can rebuild the signature from features alone.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import MIN_CONTOUR_AREA, APPROX_EPSILON_FACTOR


@dataclass
class SignatureFeatures:
    """Structured representation of extracted signature features."""

    contours: List[np.ndarray]
    hierarchy: Optional[np.ndarray]
    bounding_rect: Tuple[int, int, int, int]  # (x, y, w, h)
    stroke_width_mean: float
    stroke_width_map: np.ndarray              # distance-transform map
    skeleton: np.ndarray                       # single-pixel thinned skeleton
    smoothed_contours: List[np.ndarray]        # simplified / smoothed contours
    image_shape: Tuple[int, int]               # (height, width) of the input


class FeatureExtractor:
    """Extracts structural features from a binary signature mask."""

    def __init__(
        self,
        min_contour_area: int = MIN_CONTOUR_AREA,
        approx_epsilon: float = APPROX_EPSILON_FACTOR,
    ):
        self.min_contour_area = min_contour_area
        self.approx_epsilon = approx_epsilon

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, binary: np.ndarray) -> SignatureFeatures:
        """
        Extract all signature features from a binary mask.

        Args:
            binary: Preprocessed binary image (white strokes on black bg).

        Returns:
            SignatureFeatures dataclass with all extracted data.
        """
        # Contours
        contours, hierarchy = self._find_contours(binary)
        filtered_contours = self._filter_contours(contours)
        smoothed = self._smooth_contours(filtered_contours)

        # Bounding box of all signature content
        bounding_rect = self._compute_bounding_rect(filtered_contours, binary.shape)

        # Stroke width via distance transform
        dist_map = self._distance_transform(binary)
        stroke_width = self._estimate_stroke_width(binary, dist_map)

        # Skeleton (thinning)
        skeleton = self._skeletonize(binary)

        return SignatureFeatures(
            contours=filtered_contours,
            hierarchy=hierarchy,
            bounding_rect=bounding_rect,
            stroke_width_mean=stroke_width,
            stroke_width_map=dist_map,
            skeleton=skeleton,
            smoothed_contours=smoothed,
            image_shape=binary.shape[:2],
        )

    def get_debug_images(
        self, binary: np.ndarray, features: SignatureFeatures
    ) -> Dict[str, np.ndarray]:
        """
        Generate visualization images for debugging.

        Returns:
            Dictionary of labelled debug images.
        """
        debug: Dict[str, np.ndarray] = {}

        # Contours drawn on blank canvas
        canvas = np.zeros((*features.image_shape, 3), dtype=np.uint8)
        cv2.drawContours(canvas, features.contours, -1, (0, 255, 0), 1)
        debug["contours"] = canvas

        # Distance transform heatmap
        dist_norm = cv2.normalize(
            features.stroke_width_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        debug["distance_transform"] = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)

        # Skeleton
        skel_vis = (features.skeleton * 255).astype(np.uint8)
        debug["skeleton"] = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2BGR)

        # Smoothed contours
        smooth_canvas = np.zeros((*features.image_shape, 3), dtype=np.uint8)
        cv2.drawContours(smooth_canvas, features.smoothed_contours, -1, (0, 200, 255), 1)
        debug["smoothed_contours"] = smooth_canvas

        return debug

    # ── Private Helpers ────────────────────────────────────────────────────

    def _find_contours(self, binary: np.ndarray):
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours, hierarchy

    def _filter_contours(self, contours) -> List[np.ndarray]:
        """Remove noise contours below the area threshold."""
        return [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]

    def _smooth_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Approximate contours with fewer points for smoother strokes."""
        smoothed = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            epsilon = self.approx_epsilon * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            smoothed.append(approx)
        return smoothed

    def _compute_bounding_rect(
        self, contours: List[np.ndarray], img_shape: Tuple[int, ...]
    ) -> Tuple[int, int, int, int]:
        """Compute the bounding rectangle enclosing all contours."""
        if not contours:
            return (0, 0, img_shape[1], img_shape[0])

        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        return (x, y, w, h)

    def _distance_transform(self, binary: np.ndarray) -> np.ndarray:
        """Compute the distance transform (distance to nearest background pixel)."""
        return cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    def _estimate_stroke_width(
        self, binary: np.ndarray, dist_map: np.ndarray
    ) -> float:
        """
        Estimate mean stroke width from the distance transform.
        The average distance value on the foreground approximates half-width,
        so we double it.
        """
        foreground_mask = binary > 0
        if not np.any(foreground_mask):
            return 2.0
        mean_half_width = float(np.mean(dist_map[foreground_mask]))
        return max(2.0, mean_half_width * 2.0)

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Extract the skeleton using morphological thinning.
        Falls back to a manual iterative approach if ximgproc is unavailable.
        """
        try:
            skeleton = cv2.ximgproc.thinning(
                binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
        except AttributeError:
            # Fallback: iterative morphological skeleton
            skeleton = self._morphological_skeleton(binary)

        return skeleton

    @staticmethod
    def _morphological_skeleton(binary: np.ndarray) -> np.ndarray:
        """Manual skeletonization via iterative erosion and opening."""
        img = binary.copy()
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(img, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break

        return skel
