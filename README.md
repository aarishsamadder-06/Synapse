# SYNAPSE — AI Image Similarity Analyzer

SYNAPSE is a full-stack academic computer-vision application for comparing two images with a genuine MobileNetV2 feature-extraction pipeline. It reports raw cosine similarity as the primary result and also shows a clearly labelled normalized display index.

## Features

- JPG, JPEG, PNG and WEBP drag-and-drop uploads with browser and server validation.
- MobileNetV2 (`include_top=False`, average pooling, ImageNet weights) embeddings.
- Raw cosine similarity plus a normalized 0–100 display index (not a probability).
- Raw cosine score, classification, metadata and beginner-friendly explanation.
- FastAPI health and compare endpoints, CORS configuration and explicit errors.
- Responsive dark SaaS-style UI, loading lifecycle, reset and result visualization.

## Run locally (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Swagger documentation is available at http://localhost:8000/docs.
The first backend startup downloads MobileNetV2 ImageNet weights. Set `VITE_API_URL` to change the backend URL.

## Tests and build

```powershell
cd backend
pytest
cd ..\frontend
npm run build
```

## Architecture

The backend validates and decodes each upload, resizes it to 224×224 RGB, applies TensorFlow's MobileNetV2 preprocessing, extracts embeddings once per request from a model loaded at application startup, and computes cosine similarity. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/MACHINE_LEARNING.md](docs/MACHINE_LEARNING.md), and [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md).

## Important limitation

The raw cosine score is the primary output. The normalized index `((cosine + 1) / 2) * 100` is only a display transformation; it is not a calibrated probability, identity guarantee, pixel-equality measure, or percentage of identical content. Classification bands are provisional and have not been validated on a representative dataset. A representative labelled validation set is required before claiming scientifically meaningful thresholds.
