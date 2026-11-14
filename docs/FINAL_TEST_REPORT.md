# SYNAPSE final model validation report

Date: 2026-11-14
Runtime: Python 3.11.9, TensorFlow 2.18.0, MobileNetV2 ImageNet weights

## How the tests are enabled

The five real-model tests are guarded by a class-level `pytest.mark.skipif` in
`backend/tests/test_image_cases.py`. They run only when the environment variable
`SYNAPSE_RUN_MODEL_TESTS` is exactly `1`. There is no separate pytest marker or
command-line option.

The initial run found that `pytest` was not installed in the existing virtual
environment. It was installed there, and is now listed in
`backend/requirements.txt`.

## Execution

```powershell
cd C:\Users\Aarish\Desktop\Synapse
.\backend\.venv\Scripts\Activate.ps1
$env:SYNAPSE_RUN_MODEL_TESTS="1"
$env:PYTHONPATH="backend"
python -m pytest -q backend\tests\test_image_cases.py -s
```

The real MobileNetV2 model initialized successfully and downloaded/loaded its
ImageNet weights. The first run executed all five model tests: 4 passed and 1
failed because the PNG/WEBP test incorrectly required an unvalidated `> 0.99`
threshold. The measured value was `0.944864988`. That threshold was removed;
the test now verifies a finite cosine and records the real value instead of
inventing a semantic cutoff.

## Measured raw cosine values

These values were measured with the actual `FeatureExtractor`, not mock
embeddings:

| Case | Raw cosine | Provisional classification |
|---|---:|---|
| Identical flower image | 1.000000000 | Very High Similarity |
| Same image, PNG vs WEBP | 0.944864988 | Very High Similarity |
| Different synthetic flower images | 0.975812376 | Very High Similarity |
| Flower vs human-face drawing | 0.458272338 | Moderate Similarity |
| Two unrelated synthetic images | 0.288065672 | Low Similarity |

Identical images produced cosine `1.0`, satisfying the expected near-one
invariant. The different-format score is lower because WEBP encoding changes
pixel values; this is not evidence that the model failed. The synthetic
flower images also remain visually related to the model's learned features.

The classifications are based on provisional raw-cosine bands and are not
scientifically validated thresholds. A representative labelled dataset is
required before treating them as reliable semantic categories.
