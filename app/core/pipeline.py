"""
Photo Replication Pipeline
──────────────────────────────
End-to-end orchestrator that chains:

    Preprocessor → Extractor → Reconstructor

Supports processing from file paths, raw bytes, or NumPy arrays.
Returns structured results including all intermediate steps.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass, field

from app.core.preprocessor import SignaturePreprocessor
from app.core.extractor import FeatureExtractor, SignatureFeatures
from app.core.reconstructor import SignatureReconstructor
from app.utils.image_utils import load_image, save_image, encode_image_base64, bytes_to_cv2


@dataclass
class ReplicationResult:
    """Container for the full pipeline output."""

    # Final reconstructed signature
    final: np.ndarray

    # All reconstruction variants
    reconstructions: Dict[str, np.ndarray]

    # Preprocessing intermediate steps
    preprocessing_steps: Dict[str, np.ndarray]

    # Extracted features
    features: SignatureFeatures

    # Debug images from extractor
    debug_images: Dict[str, np.ndarray]

    def get_all_steps(self) -> Dict[str, np.ndarray]:
        """Return all images in pipeline order for visualization."""
        steps = {}

        # Preprocessing
        for name in ["original", "grayscale", "denoised", "binary", "cleaned", "normalized"]:
            if name in self.preprocessing_steps:
                img = self.preprocessing_steps[name]
                # Convert single-channel to 3-channel for consistent display
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                steps[f"1_preprocess_{name}"] = img

        # Feature extraction debug
        for name, img in self.debug_images.items():
            steps[f"2_extract_{name}"] = img

        # Reconstructions
        for name, img in self.reconstructions.items():
            steps[f"3_reconstruct_{name}"] = img

        return steps

    def encode_all_base64(self) -> Dict[str, str]:
        """Encode all step images to base64 for API response."""
        return {
            name: encode_image_base64(img)
            for name, img in self.get_all_steps().items()
        }


class SignatureReplicationPipeline:
    """
    End-to-end pipeline for signature replication.

    Usage:
        pipeline = SignatureReplicationPipeline()
        result = pipeline.run(image)
        cv2.imwrite("output.png", result.final)
    """

    def __init__(
        self,
        preprocessor: Optional[SignaturePreprocessor] = None,
        extractor: Optional[FeatureExtractor] = None,
        reconstructor: Optional[SignatureReconstructor] = None,
    ):
        self.preprocessor = preprocessor or SignaturePreprocessor()
        self.extractor = extractor or FeatureExtractor()
        self.reconstructor = reconstructor or SignatureReconstructor()

    def run(self, image: np.ndarray) -> ReplicationResult:
        """
        Run the full replication pipeline on a BGR image.

        Args:
            image: Input BGR image (NumPy array).

        Returns:
            ReplicationResult with all outputs and intermediates.
        """
        # Step 1: Preprocess
        preprocess_steps = self.preprocessor.process(image)
        binary_mask = preprocess_steps["normalized"]

        # Step 2: Extract features
        features = self.extractor.extract(binary_mask)
        debug_images = self.extractor.get_debug_images(binary_mask, features)

        # Step 3: Reconstruct
        reconstructions = self.reconstructor.reconstruct(features)

        return ReplicationResult(
            final=reconstructions["final"],
            reconstructions=reconstructions,
            preprocessing_steps=preprocess_steps,
            features=features,
            debug_images=debug_images,
        )

    def run_from_bytes(self, data: bytes) -> ReplicationResult:
        """Run pipeline from raw image bytes (e.g. from file upload)."""
        image = bytes_to_cv2(data)
        return self.run(image)

    def run_from_file(self, path: Union[str, Path]) -> ReplicationResult:
        """Run pipeline from an image file path."""
        image = load_image(path)
        return self.run(image)

    def run_and_save(
        self,
        image: np.ndarray,
        output_path: Union[str, Path],
        save_steps: bool = False,
        steps_dir: Optional[Union[str, Path]] = None,
    ) -> ReplicationResult:
        """
        Run pipeline and save the final output (and optionally all steps).

        Args:
            image: Input BGR image.
            output_path: Where to save the final reconstructed signature.
            save_steps: If True, save all intermediate step images.
            steps_dir: Directory for step images (defaults to output_path's parent).

        Returns:
            ReplicationResult.
        """
        result = self.run(image)

        # Save final
        save_image(result.final, output_path)

        # Optionally save all steps
        if save_steps:
            steps_dir = Path(steps_dir) if steps_dir else Path(output_path).parent / "steps"
            steps_dir.mkdir(parents=True, exist_ok=True)
            for name, img in result.get_all_steps().items():
                save_image(img, steps_dir / f"{name}.png")

        return result
