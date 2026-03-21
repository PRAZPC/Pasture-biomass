from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18, resnet50

from .config import LEGACY_REQUIRED_FILES, METADATA_FILE, MODEL_DIR
from .inference_features import CONTEXT_FEATURE_COLUMNS, build_context_feature_vector
from .multimodal_model import BiomassMultimodalNet, TARGET_COLUMNS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelNotReadyError(RuntimeError):
    pass


_LEGACY_ENCODER: torch.nn.Module | None = None
_BOOSTED_ENCODERS: dict[str, torch.nn.Module] = {}


def _require_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ModelNotReadyError(f"Missing model metadata: {path}. Run training/train.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _artifact_path(metadata: dict[str, Any], key: str) -> Path:
    artifacts = metadata.get("artifacts", {})
    name = artifacts.get(key)
    if not name:
        raise ModelNotReadyError(f"Model metadata is missing artifact '{key}'.")
    path = MODEL_DIR / Path(name).name
    if not path.exists():
        raise ModelNotReadyError(f"Missing model artifact: {path}.")
    return path


def _legacy_artifacts_available() -> bool:
    return all(path.exists() for path in LEGACY_REQUIRED_FILES)


def _get_legacy_encoder() -> torch.nn.Module:
    global _LEGACY_ENCODER
    if _LEGACY_ENCODER is None:
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        encoder.fc = torch.nn.Identity()
        encoder.to(DEVICE)
        encoder.eval()
        _LEGACY_ENCODER = encoder
    return _LEGACY_ENCODER


def _extract_legacy_image_features(image_path: Path) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ]
    )
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    encoder = _get_legacy_encoder()
    with torch.no_grad():
        features = encoder(tensor)
    return features.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _predict_with_legacy_artifacts(image_path: Path, height_cm: float, sampling_date: date) -> float:
    if not _legacy_artifacts_available():
        raise ModelNotReadyError(
            "Missing legacy model artifacts. Expected xgb_model.pkl, lgb_model.pkl, pca.pkl, and scaler.pkl."
        )

    xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    lgb_model = joblib.load(MODEL_DIR / "lgb_model.pkl")
    pca = joblib.load(MODEL_DIR / "pca.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    image_features = _extract_legacy_image_features(image_path)
    pca_features = pca.transform(image_features.reshape(1, -1))

    month = float(sampling_date.month)
    day_of_year = float(sampling_date.timetuple().tm_yday)
    ndvi = 0.0
    species = 0.0
    state = 0.0
    base_tabular = np.array(
        [
            ndvi,
            float(height_cm),
            ndvi * float(height_cm),
            float(height_cm) ** 2,
            ndvi**2,
            species,
            state,
            month,
            day_of_year,
        ],
        dtype=np.float32,
    )
    expected_total_features = int(getattr(scaler, "n_features_in_", pca_features.shape[1] + base_tabular.shape[0]))
    expected_tabular_features = expected_total_features - pca_features.shape[1]
    if expected_tabular_features < 1 or expected_tabular_features > base_tabular.shape[0]:
        raise ModelNotReadyError(
            f"Legacy scaler expects {expected_total_features} total features, which is unsupported."
        )
    legacy_tabular = base_tabular[:expected_tabular_features].reshape(1, -1)

    features = np.concatenate([pca_features, legacy_tabular], axis=1)
    scaled = scaler.transform(features)
    pred_xgb = float(xgb_model.predict(scaled)[0])
    pred_lgb = float(lgb_model.predict(scaled)[0])
    value = float(np.expm1((pred_xgb + pred_lgb) / 2.0))
    if np.isnan(value) or np.isinf(value):
        raise ValueError("Legacy model prediction returned an invalid numeric value.")
    return value


def _load_boosted_encoder(metadata: dict[str, Any]) -> torch.nn.Module:
    encoder_path = _artifact_path(metadata, "encoder")
    cache_key = str(encoder_path)
    cached = _BOOSTED_ENCODERS.get(cache_key)
    if cached is not None:
        return cached

    backbone = metadata.get("image_backbone", "resnet50")
    if backbone == "resnet50":
        encoder = resnet50(weights=None)
    elif backbone == "resnet18":
        encoder = resnet18(weights=None)
    else:
        raise ModelNotReadyError(f"Unsupported image backbone in metadata: {backbone}")

    encoder.fc = torch.nn.Identity()
    state = torch.load(encoder_path, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(state)
    encoder.to(DEVICE)
    encoder.eval()
    _BOOSTED_ENCODERS[cache_key] = encoder
    return encoder


def _extract_boosted_image_features(
    image_path: Path,
    metadata: dict[str, Any],
) -> np.ndarray:
    image_size = int(metadata.get("image_size", 224))
    transform = _build_transform(image_size=image_size)
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    encoder = _load_boosted_encoder(metadata)
    with torch.no_grad():
        features = encoder(tensor)
    return features.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _predict_with_boosted_metadata(
    metadata: dict[str, Any],
    image_path: Path,
    height_cm: float,
    sampling_date: date,
) -> float:
    xgb_model = joblib.load(_artifact_path(metadata, "xgb_model"))
    lgb_model = joblib.load(_artifact_path(metadata, "lgb_model"))
    pca = joblib.load(_artifact_path(metadata, "pca"))
    scaler = joblib.load(_artifact_path(metadata, "scaler"))

    image_features = _extract_boosted_image_features(image_path=image_path, metadata=metadata)
    image_reduced = pca.transform(image_features.reshape(1, -1)).astype(np.float32)
    context_features = build_context_feature_vector(
        height_cm=height_cm,
        sampling_date=sampling_date,
    ).reshape(1, -1)

    features = np.concatenate([image_reduced, context_features], axis=1)
    features = scaler.transform(features)
    pred_xgb = float(xgb_model.predict(features)[0])
    pred_lgb = float(lgb_model.predict(features)[0])
    value = float(np.expm1((pred_xgb + pred_lgb) / 2.0))
    if np.isnan(value) or np.isinf(value):
        raise ValueError("Boosted model prediction returned an invalid numeric value.")
    return value


def _load_models() -> tuple[list[BiomassMultimodalNet], dict[str, Any]]:
    metadata = _require_json(METADATA_FILE)
    checkpoints = metadata.get("fold_checkpoints", [])
    if not checkpoints:
        raise ModelNotReadyError("No fold checkpoints were listed in model metadata.")

    context_columns = metadata.get("context_feature_columns", CONTEXT_FEATURE_COLUMNS)
    context_dim = len(context_columns)
    hidden_dim = int(metadata.get("hidden_dim", 256))
    dropout = float(metadata.get("dropout", 0.25))

    models: list[BiomassMultimodalNet] = []
    for rel_path in checkpoints:
        checkpoint_path = MODEL_DIR / Path(rel_path).name
        if not checkpoint_path.exists():
            raise ModelNotReadyError(
                f"Missing fold checkpoint: {checkpoint_path}. Re-run training/train.py."
            )
        model = BiomassMultimodalNet(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        models.append(model)

    return models, metadata


def _prepare_inputs(
    image_path: Path,
    height_cm: float,
    sampling_date: date,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = _build_transform(image_size=image_size)
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)

    context = build_context_feature_vector(
        height_cm=height_cm,
        sampling_date=sampling_date,
    )
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return tensor, context_tensor


def predict_biomass(image_path: Path, height_cm: float, sampling_date: date) -> float:
    if not METADATA_FILE.exists():
        return _predict_with_legacy_artifacts(image_path, height_cm, sampling_date)

    metadata = _require_json(METADATA_FILE)
    model_type = metadata.get("model_type", "")
    if model_type == "deployable_resnet50_boosted_ensemble":
        return _predict_with_boosted_metadata(
            metadata=metadata,
            image_path=image_path,
            height_cm=height_cm,
            sampling_date=sampling_date,
        )
    if model_type == "legacy_resnet18_tabular_ensemble":
        return _predict_with_legacy_artifacts(image_path, height_cm, sampling_date)

    metadata_models, metadata = _load_models()
    image_size = int(metadata.get("image_size", 224))
    image_tensor, context_tensor = _prepare_inputs(
        image_path=image_path,
        height_cm=height_cm,
        sampling_date=sampling_date,
        image_size=image_size,
    )

    target_name = metadata.get("primary_target", TARGET_COLUMNS[0])
    if target_name not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported primary target in metadata: {target_name}")

    preds: list[float] = []
    with torch.no_grad():
        for model in metadata_models:
            outputs = model(image_tensor, context_tensor)
            preds.append(float(torch.expm1(outputs[target_name]).cpu().item()))

    value = float(np.mean(preds))
    if np.isnan(value) or np.isinf(value):
        raise ValueError("Model prediction returned an invalid numeric value.")
    return value
