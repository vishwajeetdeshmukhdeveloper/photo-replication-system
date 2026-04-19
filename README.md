# Signature Replication System

A computer vision application that **extracts and reconstructs handwritten signatures** using feature-based analysis — not pixel copying. Upload a signature image and get a clean, reconstructed version built from extracted stroke patterns.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Technical Deep Dive](#technical-deep-dive)
- [Future Improvements](#future-improvements)

---

## How It Works

The system follows a three-stage pipeline:

```
Input Image → Preprocessing → Feature Extraction → Reconstruction → Output Image
```

### 1. Preprocessing
- **Grayscale conversion** — Reduces the image to a single luminance channel, eliminating color noise.
- **Noise reduction** — Gaussian blur smooths high-frequency noise, then Non-Local Means denoising further cleans the image while preserving edge detail.
- **Adaptive thresholding** — Converts the grayscale image to binary (black and white). Adaptive thresholding handles varying lighting conditions across the image by computing thresholds locally.
- **Morphological cleanup** — `MORPH_CLOSE` fills small gaps in strokes; `MORPH_OPEN` removes tiny isolated noise dots.
- **Size normalization** — Resizes to a standard canvas width while preserving the aspect ratio.

### 2. Feature Extraction
- **Contour detection** — `cv2.findContours(RETR_TREE)` extracts all boundary contours, including nested ones (e.g., loops in letters like 'e' or 'o').
- **Contour filtering** — Removes contours below a minimum area threshold to eliminate noise artifacts.
- **Contour smoothing** — `cv2.approxPolyDP` reduces the number of contour points for smoother strokes.
- **Distance transform** — `cv2.distanceTransform` computes the distance of every foreground pixel to the nearest background pixel. This gives a stroke-width map.
- **Skeleton extraction** — Zhang-Suen thinning (via `cv2.ximgproc.thinning` or a fallback implementation) produces single-pixel-wide centerlines of each stroke.

### 3. Reconstruction
Three methods are implemented, and the best is selected as the final output:

| Method | Approach | Best For |
|--------|----------|----------|
| **Contour Filled** | Draws filled contours on a clean canvas | Thick, solid signatures |
| **Contour Outline** | Draws contour outlines with measured thickness | Consistent-width strokes |
| **Skeleton Based** | Draws circles along the skeleton with variable radius from the distance transform | Natural thickness variation |

The **final output** combines filled contours with outline enhancement and anti-aliased post-processing.

---

## Copying vs. Reconstruction — What's the Difference?

| Aspect | Pixel Copying | Feature-Based Reconstruction |
|--------|--------------|------------------------------|
| **Method** | Copies raw pixel values | Extracts contours, skeleton, stroke width |
| **Output** | Identical to input | Structurally equivalent but independently drawn |
| **Background** | Carries over noise, artifacts, paper texture | Clean white canvas |
| **Adaptability** | None — it's just a copy | Can adjust thickness, smoothing, scale |
| **Proof of analysis** | No understanding of the signature | Demonstrates structural understanding |

---

## Project Structure

```
model1/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration & defaults
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessor.py     # Image preprocessing pipeline
│   │   ├── extractor.py        # Contour & feature extraction
│   │   ├── reconstructor.py    # Signature reconstruction
│   │   └── pipeline.py         # End-to-end orchestrator
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py      # I/O helpers
├── frontend/
│   ├── index.html              # Web UI
│   ├── style.css               # Styles
│   └── script.js               # Client logic
├── samples/
│   ├── input/                  # Sample input images
│   └── output/                 # Sample output images
├── uploads/                    # Runtime uploads
├── outputs/                    # Runtime outputs
├── tests/
│   └── test_pipeline.py        # Test suite
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Install

```bash
# Clone or navigate to the project directory
cd model1

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

### Run Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Usage

### Web Interface
1. Open http://localhost:8000
2. Drag and drop a signature image (JPG/PNG) onto the upload zone
3. Click **"Replicate Signature"**
4. View the original vs. reconstructed comparison
5. Toggle **"Show processing steps"** to see all intermediate stages
6. Click **Download** to save the result

### Python API

```python
from app.core.pipeline import SignatureReplicationPipeline
from app.utils.image_utils import load_image, save_image

pipeline = SignatureReplicationPipeline()

image = load_image("samples/input/vishwajeet_signature.png")
result = pipeline.run(image)

save_image(result.final, "samples/output/reconstructed.png")
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Serve the web frontend |
| `GET /api/health` | GET | Health check |
| `POST /api/replicate` | POST | Upload image → get reconstructed signature |
| `POST /api/replicate-steps` | POST | Upload image → get all intermediate steps |

### `POST /api/replicate`

**Request:** Multipart form data with a `file` field (JPG/PNG image).

**Response:**
```json
{
  "success": true,
  "request_id": "abc123def456",
  "processing_time_seconds": 0.342,
  "reconstructed_image": "<base64 PNG>",
  "metadata": {
    "stroke_width_mean": 4.5,
    "contour_count": 23,
    "bounding_rect": [12, 8, 760, 180],
    "output_shape": [260, 840]
  }
}
```

Interactive API docs are available at **http://localhost:8000/docs** (Swagger UI).

---

## Technical Deep Dive

### Image Preprocessing

The preprocessing pipeline converts a raw photograph of a signature into a clean binary mask.

**Grayscale** reduces dimensionality from 3 channels to 1, making subsequent operations faster and eliminating color-based noise.

**Noise reduction** uses two passes: Gaussian blur (`cv2.GaussianBlur`) for broad smoothing, followed by Non-Local Means denoising (`cv2.fastNlMeansDenoising`) which compares patches across the image to remove noise while preserving fine details like thin strokes.

**Adaptive thresholding** (`cv2.adaptiveThreshold`) is critical — unlike global thresholding (`cv2.threshold`), it computes a different threshold for each small region of the image. This handles uneven lighting, shadows, and paper discoloration that would cause global methods to fail.

### Contour Detection

`cv2.findContours` with `RETR_TREE` mode extracts the complete contour hierarchy, including nested contours (useful for letter loops). Each contour is a sequence of (x, y) coordinates tracing the boundary of a connected region.

Contours smaller than a configurable area threshold are discarded as noise. The remaining contours are smoothed using `cv2.approxPolyDP`, which applies the Douglas-Peucker algorithm to reduce the number of points while preserving the overall shape.

### Distance Transform & Skeleton

The **distance transform** computes, for every foreground pixel, its distance to the nearest background pixel. Pixels at the center of thick strokes have high distance values, while pixels near stroke edges have low values. This gives us a stroke-width map.

The **skeleton** (Zhang-Suen thinning) reduces every stroke to a single-pixel-wide centerline. Combined with the distance transform, we can reconstruct strokes by drawing circles of the appropriate radius at each skeleton point — producing natural thickness variation.

---

## Future Improvements

### Deep Learning Approaches

1. **Convolutional Autoencoder**
   - Train an encoder-decoder network on signature images
   - The encoder learns a compact latent representation of the signature
   - The decoder reconstructs the signature from this representation
   - Would handle complex styles and degraded inputs better than classical CV

2. **Generative Adversarial Network (GAN)**
   - Use a Pix2Pix or CycleGAN architecture
   - Generator: produces reconstructed signatures
   - Discriminator: distinguishes real from generated signatures
   - Could learn to replicate ink texture, pressure variation, and writing style

3. **Style Transfer**
   - Separate content (stroke structure) from style (ink texture, pressure)
   - Apply consistent styling to the reconstructed skeleton

### Other Improvements
- **Multi-signature detection** — Automatically segment multiple signatures from a single document
- **Stroke order prediction** — Infer the temporal order of strokes using graph analysis
- **Pressure simulation** — Model pen pressure from stroke width variation
- **Writer identification** — Build a feature vector for writer authentication
- **PDF support** — Extract signature regions from PDF documents

---

## License

This project is for educational and research purposes.
