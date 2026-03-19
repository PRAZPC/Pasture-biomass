from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from PIL import Image

from .config import (
    XGB_MODEL_FILE,
    LGB_MODEL_FILE,
    PCA_FILE,
    SCALER_FILE,
    SSL_ENCODER_FILE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelNotReadyError(RuntimeError):
    pass


def _require(path: Path) -> Any:
    """Load a joblib artifact, raising ModelNotReadyError if missing."""
    if not path.exists():
        raise ModelNotReadyError(
            f"Missing model artifact: {path}. Run the training script first."
        )
    return joblib.load(path)


def _load_encoder() -> tuple[nn.Module, int]:
    """
    Load the SSL-trained encoder. Auto-detects ResNet-18 or ResNet-50 based on saved weights.
    Returns (encoder, image_size) tuple.
    """
    if not SSL_ENCODER_FILE.exists():
        raise ModelNotReadyError(
            f"Missing SSL encoder: {SSL_ENCODER_FILE}. Run training/train.py first."
        )
    
    state = torch.load(SSL_ENCODER_FILE, map_location=DEVICE, weights_only=True)
    
    # Detect model type by checking layer dimensions
    # ResNet-18: layer4.1.conv2 has 512 channels
    # ResNet-50: layer4.2.conv3 has 2048 channels
    if 'layer4.2.conv3.weight' in state:
        # ResNet-50
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        img_size = 224
    else:
        # ResNet-18
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        img_size = 160
    
    encoder.fc = nn.Identity()
    encoder.load_state_dict(state)
    encoder.to(DEVICE)
    encoder.eval()
    return encoder, img_size


def _extract_image_features(image_path: Path, encoder: nn.Module, img_size: int) -> np.ndarray:
    """Run the image through the SSL encoder."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = encoder(tensor)
    return feat.cpu().numpy().flatten()


def predict_biomass(image_path: Path) -> float:
    """
    Image-only inference pipeline:
      1. SSL encoder image features
      2. PCA dimensionality reduction
      3. StandardScaler/RobustScaler
      4. XGBRegressor + LGBMRegressor ensemble (average)
      5. expm1 to undo log1p target transform
    """
    # --- load all artifacts ---
    xgb = _require(XGB_MODEL_FILE)
    lgb = _require(LGB_MODEL_FILE)
    pca = _require(PCA_FILE)
    scaler = _require(SCALER_FILE)
    encoder, img_size = _load_encoder()

    # --- image features ---
    img_feat = _extract_image_features(image_path, encoder, img_size)
    img_feat_pca = pca.transform(img_feat.reshape(1, -1))

    # --- scale ---
    X = scaler.transform(img_feat_pca)

    # --- ensemble predict (log-space) then invert ---
    pred_xgb = xgb.predict(X)[0]
    pred_lgb = lgb.predict(X)[0]
    pred_log = (pred_xgb + pred_lgb) / 2.0
    value = float(np.expm1(pred_log))

    if np.isnan(value) or np.isinf(value):
        raise ValueError("Model prediction returned an invalid numeric value.")

    return value
