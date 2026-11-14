import numpy as np

from .config import MODEL_NAME, MODEL_WEIGHTS


def l2_normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("The model returned a zero or invalid feature vector.")
    normalized = embedding / norm
    if not np.isfinite(normalized).all():
        raise ValueError("The model returned invalid normalized feature values.")
    return normalized


class FeatureExtractor:
    def __init__(self) -> None:
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
        except ImportError as error:
            raise RuntimeError("TensorFlow is required to initialize the feature extractor.") from error
        self._tf = tf
        self.model = MobileNetV2(
            weights=MODEL_WEIGHTS, include_top=False, pooling="avg", input_shape=(224, 224, 3)
        )

    def extract(self, image_tensor: np.ndarray) -> np.ndarray:
        if image_tensor.shape != (224, 224, 3):
            raise ValueError("Feature extraction requires a 224x224 RGB image tensor.")
        batch = np.expand_dims(image_tensor, axis=0)
        prepared = self._tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        embedding = self.model.predict(prepared, verbose=0)[0]
        return l2_normalize_embedding(embedding)

    @property
    def name(self) -> str:
        return MODEL_NAME
