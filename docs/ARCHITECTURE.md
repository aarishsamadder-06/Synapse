# Architecture

`frontend/src/main.jsx` contains the composed React experience and reusable upload/result sections; `styles.css` provides the responsive design system. API communication uses `fetch` and `VITE_API_URL`.

The FastAPI application in `backend/app/main.py` owns lifecycle-managed model initialization and the two public routes. `image_processor.py` validates and decodes uploads, `feature_extractor.py` wraps MobileNetV2, and `similarity.py` contains pure numerical operations. Uploaded bytes are processed in memory and are not persisted.
