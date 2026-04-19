"""
Application configuration settings.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
SAMPLE_INPUT_DIR = BASE_DIR / "samples" / "input"
SAMPLE_OUTPUT_DIR = BASE_DIR / "samples" / "output"
FRONTEND_DIR = BASE_DIR / "frontend"

# ── Processing Defaults ───────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NORMALIZED_WIDTH = 800           # pixels – target canvas width for normalization
CANVAS_PADDING = 40              # pixels – padding around the reconstructed signature

# ── Preprocessing ─────────────────────────────────────────────────────────────
GAUSSIAN_KERNEL = (5, 5)         # kernel size for Gaussian blur
DENOISE_H = 10                   # filter strength for Non-Local Means denoising
ADAPTIVE_BLOCK_SIZE = 15         # block size for adaptive thresholding (must be odd)
ADAPTIVE_C = 10                  # constant subtracted from mean in adaptive threshold
MORPH_KERNEL_SIZE = (3, 3)       # kernel for morphological operations

# ── Feature Extraction ────────────────────────────────────────────────────────
MIN_CONTOUR_AREA = 25            # minimum contour area to keep (filters noise)
APPROX_EPSILON_FACTOR = 0.001   # factor for contour approximation precision (smaller = more detail)

# ── Reconstruction ────────────────────────────────────────────────────────────
DEFAULT_STROKE_THICKNESS = 2     # fallback stroke thickness if detection fails
ANTI_ALIAS = True                # use anti-aliased line drawing
POST_BLUR_KERNEL = (3, 3)       # optional smoothing on final output (ink-like effect)
INK_COLOR = (15, 15, 15)        # near-black ink color (BGR)


def ensure_directories():
    """Create required runtime directories if they don't exist."""
    for directory in [UPLOAD_DIR, OUTPUT_DIR, SAMPLE_INPUT_DIR, SAMPLE_OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
