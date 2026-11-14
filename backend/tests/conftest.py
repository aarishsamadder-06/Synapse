import pytest

from app.feature_extractor import FeatureExtractor


@pytest.fixture(scope="session")
def extractor():
    return FeatureExtractor()
