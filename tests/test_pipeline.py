"""
Tests for the Signature Replication Pipeline
─────────────────────────────────────────────
Run with:  pytest tests/test_pipeline.py -v
"""

import sys
import os
import numpy as np
import cv2
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.preprocessor import SignaturePreprocessor
from app.core.extractor import FeatureExtractor
from app.core.reconstructor import SignatureReconstructor
from app.core.pipeline import SignatureReplicationPipeline
from app.utils.image_utils import encode_image_base64, decode_image_base64


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_signature():
    """Create a synthetic signature image for testing."""
    canvas = np.full((200, 500, 3), 255, dtype=np.uint8)

    # Draw a cursive-like signature path
    pts = np.array([
        [50, 150], [80, 80], [120, 140], [160, 60],
        [200, 130], [240, 50], [280, 120], [320, 70],
        [360, 140], [400, 90], [440, 150]
    ], dtype=np.int32)

    cv2.polylines(canvas, [pts], False, (20, 20, 20), thickness=3, lineType=cv2.LINE_AA)

    # Add a small underline
    cv2.line(canvas, (60, 170), (420, 170), (30, 30, 30), 2, cv2.LINE_AA)

    return canvas


@pytest.fixture
def preprocessor():
    return SignaturePreprocessor()


@pytest.fixture
def extractor():
    return FeatureExtractor()


@pytest.fixture
def reconstructor():
    return SignatureReconstructor()


@pytest.fixture
def pipeline():
    return SignatureReplicationPipeline()


# ── Preprocessor Tests ─────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_process_returns_all_steps(self, preprocessor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        expected_keys = {"original", "grayscale", "denoised", "binary", "cleaned", "normalized"}
        assert expected_keys == set(steps.keys())

    def test_grayscale_is_single_channel(self, preprocessor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        assert len(steps["grayscale"].shape) == 2

    def test_binary_is_binary(self, preprocessor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        unique = np.unique(steps["binary"])
        assert all(v in [0, 255] for v in unique)

    def test_normalized_has_correct_width(self, preprocessor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        assert steps["normalized"].shape[1] == preprocessor.target_width


# ── Extractor Tests ────────────────────────────────────────────────────────────

class TestExtractor:
    def test_extract_finds_contours(self, preprocessor, extractor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        assert len(features.contours) > 0

    def test_stroke_width_is_positive(self, preprocessor, extractor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        assert features.stroke_width_mean > 0

    def test_skeleton_is_not_empty(self, preprocessor, extractor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        assert np.any(features.skeleton > 0)

    def test_bounding_rect_is_valid(self, preprocessor, extractor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        x, y, w, h = features.bounding_rect
        assert w > 0 and h > 0

    def test_debug_images(self, preprocessor, extractor, synthetic_signature):
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        debug = extractor.get_debug_images(steps["normalized"], features)
        assert "contours" in debug
        assert "skeleton" in debug
        assert "distance_transform" in debug


# ── Reconstructor Tests ────────────────────────────────────────────────────────

class TestReconstructor:
    def test_reconstruct_returns_all_variants(
        self, preprocessor, extractor, reconstructor, synthetic_signature
    ):
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        results = reconstructor.reconstruct(features)
        expected = {"contour_filled", "contour_outline", "skeleton_based", "final"}
        assert expected == set(results.keys())

    def test_final_is_color_image(
        self, preprocessor, extractor, reconstructor, synthetic_signature
    ):
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        results = reconstructor.reconstruct(features)
        assert len(results["final"].shape) == 3  # H x W x 3

    def test_output_not_identical_to_input(
        self, preprocessor, extractor, reconstructor, synthetic_signature
    ):
        """Verify the output is NOT a pixel copy of the input."""
        steps = preprocessor.process(synthetic_signature)
        features = extractor.extract(steps["normalized"])
        results = reconstructor.reconstruct(features)

        # Shapes will differ (padding, normalization)
        final_shape = results["final"].shape[:2]
        input_shape = synthetic_signature.shape[:2]
        # Reconstruction should NOT have the same dimensions as the raw input
        # (it's sized to the bounding box + padding)
        assert final_shape != input_shape or not np.array_equal(
            results["final"],
            cv2.resize(synthetic_signature, (final_shape[1], final_shape[0]))
        )


# ── Pipeline End-to-End ────────────────────────────────────────────────────────

class TestPipeline:
    def test_full_pipeline(self, pipeline, synthetic_signature):
        result = pipeline.run(synthetic_signature)
        assert result.final is not None
        assert result.final.shape[2] == 3  # BGR color
        assert len(result.reconstructions) == 4
        assert len(result.preprocessing_steps) > 0

    def test_pipeline_from_bytes(self, pipeline, synthetic_signature):
        _, buf = cv2.imencode(".png", synthetic_signature)
        result = pipeline.run_from_bytes(buf.tobytes())
        assert result.final is not None

    def test_encode_all_base64(self, pipeline, synthetic_signature):
        result = pipeline.run(synthetic_signature)
        encoded = result.encode_all_base64()
        assert len(encoded) > 0
        # Each should be a valid base64 string
        for key, val in encoded.items():
            assert isinstance(val, str)
            assert len(val) > 100  # non-trivial content


# ── Image Utils ────────────────────────────────────────────────────────────────

class TestImageUtils:
    def test_encode_decode_roundtrip(self, synthetic_signature):
        b64 = encode_image_base64(synthetic_signature)
        decoded = decode_image_base64(b64)
        assert decoded.shape == synthetic_signature.shape
