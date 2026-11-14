import numpy as np
import pytest

from app.feature_extractor import l2_normalize_embedding
from app.similarity import classify_similarity, cosine_similarity, normalized_similarity_index


def test_identical_embeddings_have_raw_cosine_one_and_index_one_hundred():
    cosine = cosine_similarity(np.array([1, 2]), np.array([1, 2]))
    assert cosine == 1
    assert normalized_similarity_index(cosine) == 100


def test_embedding_is_l2_normalized():
    normalized = l2_normalize_embedding(np.array([3, 4]))
    assert np.allclose(normalized, [0.6, 0.8])


def test_zero_embedding_is_rejected():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(2), np.ones(2))


def test_classification_thresholds():
    assert classify_similarity(0.9) == "Very High Similarity"
    assert classify_similarity(0.7) == "High Similarity"
    assert classify_similarity(0.5) == "Moderate Similarity"
    assert classify_similarity(0.2) == "Low Similarity"
