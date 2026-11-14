import os
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.feature_extractor import FeatureExtractor
from app.image_processor import process_image
from app.similarity import cosine_similarity


def image_bytes(kind: str, image_format: str = "PNG") -> bytes:
    is_flower = kind.startswith("flower")
    image = Image.new("RGB", (160, 160), (245, 220, 80) if is_flower else (24, 35, 60))
    draw = ImageDraw.Draw(image)
    if is_flower:
        draw.ellipse((45, 45, 115, 115), fill=(120, 70, 25))
        petal_color = (240, 90, 150) if kind == "flower" else (220, 50, 80)
        for box in ((20, 45, 70, 95), (90, 45, 140, 95), (45, 15, 95, 65), (45, 95, 95, 145)):
            draw.ellipse(box, fill=petal_color)
        draw.rectangle((75, 110, 85, 160), fill=(35, 130, 55))
    elif kind == "face":
        draw.ellipse((35, 20, 125, 140), fill=(194, 139, 94))
        draw.ellipse((58, 65, 68, 75), fill=(20, 20, 20))
        draw.ellipse((92, 65, 102, 75), fill=(20, 20, 20))
        draw.arc((60, 75, 100, 115), 0, 180, fill=(70, 25, 25), width=3)
    else:
        draw.rectangle((20, 20, 140, 140), fill=(190, 30, 35))
        draw.line((20, 20, 140, 140), fill=(20, 220, 230), width=12)
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_all_supported_images_share_the_same_rgb_model_input_shape():
    for kind in ("flower", "face", "unrelated"):
        processed = process_image(image_bytes(kind), f"{kind}.png", "image/png")
        assert processed.image.mode == "RGB"
        assert processed.tensor.shape == (224, 224, 3)
        assert processed.tensor.dtype == np.float32


@pytest.mark.skipif(
    os.getenv("SYNAPSE_RUN_MODEL_TESTS") != "1",
    reason="Set SYNAPSE_RUN_MODEL_TESTS=1 to download/use ImageNet weights.",
)
class TestMobileNetV2ImageCases:
    def embedding(self, extractor, kind: str, image_format: str = "PNG"):
        processed = process_image(image_bytes(kind, image_format), f"{kind}.{image_format.lower()}", f"image/{image_format.lower()}")
        return extractor.extract(processed.tensor)

    def test_identical_images(self, extractor):
        assert cosine_similarity(self.embedding(extractor, "flower"), self.embedding(extractor, "flower")) > 0.999

    def test_same_image_in_different_formats(self, extractor):
        cosine = cosine_similarity(
            self.embedding(extractor, "flower", "PNG"),
            self.embedding(extractor, "flower", "WEBP"),
        )
        assert np.isfinite(cosine)

    def test_two_different_flower_images(self, extractor):
        first = self.embedding(extractor, "flower")
        second = self.embedding(extractor, "flower2")
        assert np.isfinite(cosine_similarity(first, second))

    def test_flower_and_human_face(self, extractor):
        assert np.isfinite(cosine_similarity(self.embedding(extractor, "flower"), self.embedding(extractor, "face")))

    def test_two_unrelated_images(self, extractor):
        assert np.isfinite(cosine_similarity(self.embedding(extractor, "face"), self.embedding(extractor, "unrelated")))
