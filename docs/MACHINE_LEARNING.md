# Machine learning methodology

Each image is verified with Pillow, converted to RGB, resized to 224×224, converted to a NumPy tensor, and passed through TensorFlow's `mobilenet_v2.preprocess_input` after adding a single batch dimension `(1, 224, 224, 3)`. Both inputs use this identical path. MobileNetV2 is loaded once with `weights="imagenet"`, no classification head, and global average pooling. The resulting vectors are L2-normalized before comparison; cosine similarity also validates dimensions, finiteness and non-zero norms.

For vectors A and B:

`cosine(A, B) = dot(A, B) / (norm(A) * norm(B))`

The UI normalized index is `((cosine + 1) / 2) * 100`, clamped to 0–100. It is a display transformation rather than a probability, calibrated semantic percentage, or measure of identical content. A cosine of `0.3237` therefore produces `66.185`, but that arithmetic does not mean “66.2% similar”; it only reflects the chosen mapping of the raw vector angle.

The current classification bands (`<0.3`, `0.3–0.6`, `0.6–0.8`, `>=0.8`) are transparent provisional heuristics over the raw cosine. They are not scientifically justified universal thresholds. A production-quality classifier would require a representative, labelled validation dataset, calibration, and evaluation for the intended domain.
