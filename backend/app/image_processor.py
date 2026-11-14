from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_UPLOAD_SIZE, MIN_IMAGE_DIMENSION


@dataclass
class ProcessedImage:
    image: Image.Image
    metadata: dict
    tensor: np.ndarray


def process_image(content: bytes, filename: str, content_type: str | None) -> ProcessedImage:
    if len(content) > MAX_UPLOAD_SIZE:
        raise ValueError(f"Image exceeds the {MAX_UPLOAD_SIZE // (1024 * 1024)} MB limit.")
    extension = PurePath(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file extension. Use JPG, PNG, or WEBP.")
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        raise ValueError("Unsupported image MIME type.")
    try:
        with Image.open(BytesIO(content)) as source:
            source.verify()
        with Image.open(BytesIO(content)) as source:
            image = source.convert("RGB")
            width, height = image.size
            image_format = source.format or extension.lstrip(".").upper()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("The uploaded file is not a valid, readable image.") from error
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ValueError(f"Images must be at least {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels.")
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    tensor = np.asarray(resized, dtype=np.float32)
    return ProcessedImage(
        image=image,
        metadata={
            "filename": filename,
            "width": width,
            "height": height,
            "format": image_format,
            "file_size": len(content),
        },
        tensor=tensor,
    )
