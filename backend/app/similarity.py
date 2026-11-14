import numpy as np


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity, rejecting invalid or zero-length embeddings."""
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    if first.size == 0 or second.size == 0 or first.size != second.size:
        raise ValueError("Embeddings must be non-empty and have the same size.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Embeddings contain invalid numerical values.")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        raise ValueError("Cannot compare a zero-length embedding.")
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def normalized_similarity_index(value: float) -> float:
    """Map cosine to a display index; this is not a probability or calibrated percentage."""
    if not np.isfinite(value):
        raise ValueError("Similarity must be finite.")
    return float(np.clip(((value + 1.0) / 2.0) * 100.0, 0.0, 100.0))


def classify_similarity(cosine: float) -> str:
    """Use transparent, provisional raw-cosine bands, not a validated classifier."""
    if cosine >= 0.8:
        return "Very High Similarity"
    if cosine >= 0.6:
        return "High Similarity"
    if cosine >= 0.3:
        return "Moderate Similarity"
    return "Low Similarity"
