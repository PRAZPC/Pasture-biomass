from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .config import MODEL_FILE, PREPROCESSOR_FILE
from .feature_extraction import extract_image_features


class ModelNotReadyError(RuntimeError):
    pass


def _load_artifact(path: Path) -> Any:
    if not path.exists():
        raise ModelNotReadyError(
            f"Missing model artifact: {path}. Train your model once and save it here."
        )
    return joblib.load(path)


def extract_features_from_image(image_path: Path, preprocessor: Any | None = None) -> np.ndarray:
    features = extract_image_features(image_path)
    features_2d = features.reshape(1, -1)

    if preprocessor is not None:
        if hasattr(preprocessor, "transform"):
            return preprocessor.transform(features_2d)
        raise ValueError("Loaded preprocessor does not expose a .transform() method.")

    return features_2d


def predict_biomass(image_path: Path) -> float:
    loaded = _load_artifact(MODEL_FILE)
    preprocessor = _load_artifact(PREPROCESSOR_FILE) if PREPROCESSOR_FILE.exists() else None

    model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded

    features = extract_features_from_image(image_path=image_path, preprocessor=preprocessor)
    prediction = model.predict(features)

    value = float(prediction[0])
    if np.isnan(value) or np.isinf(value):
        raise ValueError("Model prediction returned an invalid numeric value.")

    return value
