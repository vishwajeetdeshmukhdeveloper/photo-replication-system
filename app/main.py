"""
Photo Replication System — FastAPI Application
───────────────────────────────────────────────────
Main entry point.  Run with:

    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import ensure_directories, FRONTEND_DIR
from app.api.routes import router


# ── App Initialization ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Photo Replication System",
    description=(
        "A computer vision application that extracts stroke patterns from "
        "handwritten signatures and reconstructs them on a clean canvas."
    ),
    version="1.0.0",
)

# CORS – allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files – serve the frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Include API routes
app.include_router(router)


# ── Events ─────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    ensure_directories()


# ── Root – serve the frontend ──────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
