# API documentation

## `GET /api/health`

Returns `status`, `model_ready`, `model_name`, and `api_version`. A failed TensorFlow initialization is reported as `degraded` and `model_ready: false`.

## `POST /api/compare`

Send `multipart/form-data` with required `image1` and `image2` fields. Successful responses contain `success`, `cosine_similarity` (the primary raw metric), `normalized_similarity_index` (display-only 0–100 transformation), `classification`, model/algorithm names, metadata for each image, and an explanation. The derived index is not a probability or percentage of identical content. Invalid uploads return HTTP 400; unavailable model state returns 503; unexpected processing failures return 500.

Uploads must use JPG/JPEG, PNG or WEBP, be no larger than 10 MB, and be at least 32×32 pixels.
