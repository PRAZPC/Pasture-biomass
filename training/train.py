from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference_features import CONTEXT_FEATURE_COLUMNS, build_context_feature_frame

warnings.filterwarnings("ignore")

DATASET_DIR = PROJECT_ROOT / "Dataset"
TRAIN_CSV = DATASET_DIR / "train.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

PRIMARY_TARGET = "Dry_Total_g"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainArtifacts:
    metadata_path: Path
    encoder_path: Path
    xgb_model_path: Path
    lgb_model_path: Path
    pca_path: Path
    scaler_path: Path


class ImageEmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = DATASET_DIR / row["image_path"]
        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))
        return image, idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deployable biomass estimator from image, height, and sampling date."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frame(max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV)
    df = df.pivot_table(
        index=[
            "sample_id",
            "image_path",
            "Sampling_Date",
            "State",
            "Species",
            "Pre_GSHH_NDVI",
            "Height_Ave_cm",
        ],
        columns="target_name",
        values="target",
    ).reset_index()

    df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"], errors="coerce")
    df = df.dropna(subset=["Sampling_Date", "Height_Ave_cm", PRIMARY_TARGET]).copy()

    valid_rows = []
    for idx, row in df.iterrows():
        if (DATASET_DIR / row["image_path"]).exists():
            valid_rows.append(idx)

    df = df.loc[valid_rows].reset_index(drop=True)
    if max_samples is not None:
        df = df.head(max_samples).copy()
    return df.reset_index(drop=True)


def build_feature_extractor() -> torch.nn.Module:
    encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.to(DEVICE)
    encoder.eval()
    return encoder


def extract_image_embeddings(
    df: pd.DataFrame,
    encoder: torch.nn.Module,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = ImageEmbeddingDataset(df=df, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings = np.zeros((len(dataset), 2048), dtype=np.float32)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(DEVICE, non_blocking=True)
            batch_embeddings = encoder(images).detach().cpu().numpy().astype(np.float32)
            embeddings[indices.numpy()] = batch_embeddings
    return embeddings


def build_models(seed: int) -> tuple[XGBRegressor, LGBMRegressor]:
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    lgb = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )
    return xgb, lgb


def evaluate_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def cross_validate(X: np.ndarray, y_log: np.ndarray, folds: int, seed: int) -> tuple[list[dict[str, float]], dict[str, float]]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred_log = np.zeros(len(y_log), dtype=np.float32)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        xgb, lgb = build_models(seed=seed + fold)
        xgb.fit(X[train_idx], y_log[train_idx])
        lgb.fit(X[train_idx], y_log[train_idx])

        pred_log = (xgb.predict(X[val_idx]) + lgb.predict(X[val_idx])) / 2.0
        oof_pred_log[val_idx] = pred_log
        metrics = evaluate_predictions(y_log[val_idx], pred_log)
        metrics["fold"] = float(fold)
        fold_metrics.append(metrics)
        print(
            f"Fold {fold} R2={metrics['r2']:.4f} "
            f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}"
        )

    return fold_metrics, evaluate_predictions(y_log, oof_pred_log)


def save_artifacts(
    args: argparse.Namespace,
    encoder: torch.nn.Module,
    pca: PCA,
    scaler: RobustScaler,
    xgb: XGBRegressor,
    lgb: LGBMRegressor,
    fold_metrics: list[dict[str, float]],
    final_metrics: dict[str, float],
    image_feature_dim: int,
) -> TrainArtifacts:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    encoder_path = SAVED_MODELS_DIR / "image_encoder.pth"
    xgb_model_path = SAVED_MODELS_DIR / "xgb_model.pkl"
    lgb_model_path = SAVED_MODELS_DIR / "lgb_model.pkl"
    pca_path = SAVED_MODELS_DIR / "pca.pkl"
    scaler_path = SAVED_MODELS_DIR / "scaler.pkl"
    metadata_path = SAVED_MODELS_DIR / "model_metadata.json"

    torch.save(encoder.state_dict(), encoder_path)
    joblib.dump(xgb, xgb_model_path)
    joblib.dump(lgb, lgb_model_path)
    joblib.dump(pca, pca_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": "deployable_resnet50_boosted_ensemble",
        "primary_target": PRIMARY_TARGET,
        "targets": [PRIMARY_TARGET],
        "image_backbone": "resnet50",
        "image_size": args.image_size,
        "image_feature_dim": image_feature_dim,
        "pca_components": int(getattr(pca, "n_components_", args.pca_components)),
        "context_feature_columns": CONTEXT_FEATURE_COLUMNS,
        "artifacts": {
            "encoder": encoder_path.name,
            "xgb_model": xgb_model_path.name,
            "lgb_model": lgb_model_path.name,
            "pca": pca_path.name,
            "scaler": scaler_path.name,
        },
        "folds": args.folds,
        "fold_metrics": fold_metrics,
        "mean_r2": float(np.mean([m["r2"] for m in fold_metrics])),
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "final_r2": final_metrics["r2"],
        "final_rmse": final_metrics["rmse"],
        "final_mae": final_metrics["mae"],
        "deployable_inputs": ["image", "height_cm", "sampling_date"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainArtifacts(
        metadata_path=metadata_path,
        encoder_path=encoder_path,
        xgb_model_path=xgb_model_path,
        lgb_model_path=lgb_model_path,
        pca_path=pca_path,
        scaler_path=scaler_path,
    )


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 72)
    print("Biomass Trainer: Deployable ResNet50 + Boosted Ensemble")
    print("=" * 72)

    print("\n[1/4] Loading data...")
    df = load_frame(max_samples=args.max_samples)
    if len(df) < args.folds:
        raise ValueError(f"Need at least {args.folds} valid samples, found {len(df)}.")
    print(f"Samples after pivot: {len(df)}")

    print("\n[2/4] Extracting fixed image embeddings...")
    encoder = build_feature_extractor()
    image_embeddings = extract_image_embeddings(
        df=df,
        encoder=encoder,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Image embedding shape: {image_embeddings.shape}")

    print("\n[3/4] Building deployable feature matrix...")
    context_df = build_context_feature_frame(df)
    print(f"Context feature shape: {context_df.shape}")

    pca_components = min(args.pca_components, image_embeddings.shape[0] - 1, image_embeddings.shape[1])
    if pca_components < 8:
        raise ValueError(f"PCA requires at least 8 usable components, got {pca_components}.")
    pca = PCA(n_components=pca_components, random_state=args.seed)
    image_reduced = pca.fit_transform(image_embeddings).astype(np.float32)

    combined = np.concatenate(
        [image_reduced, context_df.to_numpy(dtype=np.float32)],
        axis=1,
    )
    scaler = RobustScaler()
    X = scaler.fit_transform(combined).astype(np.float32)
    y_log = np.log1p(df[PRIMARY_TARGET].to_numpy(dtype=np.float32))

    print(f"Final feature shape: {X.shape}")

    print("\n[4/4] Cross-validating and fitting final models...")
    fold_metrics, final_metrics = cross_validate(
        X=X,
        y_log=y_log,
        folds=args.folds,
        seed=args.seed,
    )
    print(f"OOF R2:   {final_metrics['r2']:.4f}")
    print(f"OOF RMSE: {final_metrics['rmse']:.4f}")
    print(f"OOF MAE:  {final_metrics['mae']:.4f}")

    final_xgb, final_lgb = build_models(seed=args.seed)
    final_xgb.fit(X, y_log)
    final_lgb.fit(X, y_log)

    artifacts = save_artifacts(
        args=args,
        encoder=encoder.cpu(),
        pca=pca,
        scaler=scaler,
        xgb=final_xgb,
        lgb=final_lgb,
        fold_metrics=fold_metrics,
        final_metrics=final_metrics,
        image_feature_dim=image_embeddings.shape[1],
    )

    print("\nArtifacts saved:")
    for key, value in asdict(artifacts).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    train()
