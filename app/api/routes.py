"""
API Routes for Signature Replication System
────────────────────────────────────────────
Defines all HTTP endpoints for the application.
"""

import uuid
import time
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, UPLOAD_DIR, OUTPUT_DIR
from app.core.pipeline import SignatureReplicationPipeline
from app.utils.image_utils import encode_image_base64, save_image

router = APIRouter()

# Shared pipeline instance (stateless, safe to reuse)
pipeline = SignatureReplicationPipeline()


@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Signature Replication System"}


@router.post("/api/replicate")
async def replicate_signature(file: UploadFile = File(...)):
    """
    Accept a signature image and return the reconstructed version.

    Returns:
        JSON with base64-encoded final reconstruction and metadata.
    """
    _validate_upload(file)

    contents = await file.read()
    request_id = uuid.uuid4().hex[:12]

    try:
        start_time = time.time()
        result = pipeline.run_from_bytes(contents)
        elapsed = round(time.time() - start_time, 3)

        # Save to disk
        output_path = OUTPUT_DIR / f"{request_id}_reconstructed.png"
        save_image(result.final, output_path)

        return JSONResponse({
            "success": True,
            "request_id": request_id,
            "processing_time_seconds": elapsed,
            "reconstructed_image": encode_image_base64(result.final),
            "metadata": {
                "stroke_width_mean": round(result.features.stroke_width_mean, 2),
                "contour_count": len(result.features.contours),
                "bounding_rect": result.features.bounding_rect,
                "output_shape": list(result.final.shape[:2]),
            },
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/api/replicate-steps")
async def replicate_with_steps(file: UploadFile = File(...)):
    """
    Accept a signature image and return all processing steps.
    Useful for debugging and visualizing the pipeline.

    Returns:
        JSON with base64 images for every intermediate step.
    """
    _validate_upload(file)

    contents = await file.read()
    request_id = uuid.uuid4().hex[:12]

    try:
        start_time = time.time()
        result = pipeline.run_from_bytes(contents)
        elapsed = round(time.time() - start_time, 3)

        # Encode all intermediate steps
        all_steps = result.encode_all_base64()

        return JSONResponse({
            "success": True,
            "request_id": request_id,
            "processing_time_seconds": elapsed,
            "steps": all_steps,
            "final_image": encode_image_base64(result.final),
            "metadata": {
                "stroke_width_mean": round(result.features.stroke_width_mean, 2),
                "contour_count": len(result.features.contours),
                "bounding_rect": result.features.bounding_rect,
                "output_shape": list(result.final.shape[:2]),
            },
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


def _validate_upload(file: UploadFile):
    """Validate uploaded file type and size."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
