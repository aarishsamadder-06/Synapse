from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    filename: str
    width: int
    height: int
    format: str
    file_size: int = Field(alias="file_size")

    model_config = {"populate_by_name": True}


class CompareResponse(BaseModel):
    success: bool
    normalized_similarity_index: float
    cosine_similarity: float
    classification: str
    model_name: str
    algorithm_name: str
    image1: ImageMetadata
    image2: ImageMetadata
    explanation: str
    score_interpretation: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_name: str
    api_version: str
