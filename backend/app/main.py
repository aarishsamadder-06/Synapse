from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import API_VERSION, MODEL_NAME
from .feature_extractor import FeatureExtractor
from .image_processor import process_image
from .schemas import CompareResponse, HealthResponse
from .similarity import classify_similarity, cosine_similarity, normalized_similarity_index

extractor: FeatureExtractor | None = None
model_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global extractor, model_error
    try:
        extractor = FeatureExtractor()
        model_error = None
    except RuntimeError as error:
        extractor = None
        model_error = str(error)
    yield
    extractor = None


app = FastAPI(title="SYNAPSE API", version=API_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://synapse-fs95.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if extractor else "degraded",
        model_ready=extractor is not None,
        model_name=MODEL_NAME,
        api_version=API_VERSION,
    )


@app.post("/api/compare", response_model=CompareResponse)
async def compare(image1: UploadFile = File(...), image2: UploadFile = File(...)) -> CompareResponse:
    if extractor is None:
        raise HTTPException(status_code=503, detail=model_error or "The model is not ready.")
    try:
        first = process_image(await image1.read(), image1.filename or "", image1.content_type)
        second = process_image(await image2.read(), image2.filename or "", image2.content_type)
        first_embedding = extractor.extract(first.tensor)
        second_embedding = extractor.extract(second.tensor)
        cosine = cosine_similarity(first_embedding, second_embedding)
        index = normalized_similarity_index(cosine)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Image analysis failed. Please try again.") from error
    return CompareResponse(
        success=True,
        normalized_similarity_index=round(index, 2),
        cosine_similarity=round(cosine, 6),
        classification=classify_similarity(cosine),
        model_name=MODEL_NAME,
        algorithm_name="Cosine Similarity",
        image1=first.metadata,
        image2=second.metadata,
        explanation=(
            "MobileNetV2 extracted feature representations from both images and cosine "
            "similarity compared their alignment. The raw cosine is the primary result; the "
            "normalized index is only a display transformation and is not pixel equality, "
            "a calibrated probability, or a percentage of identical content."
        ),
        score_interpretation=(
            "Classification uses provisional raw-cosine bands. Thresholds have not been "
            "validated on a representative dataset and should not be treated as universal."
        ),
    )
