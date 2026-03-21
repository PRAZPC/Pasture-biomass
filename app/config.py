from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"

# Persist trained artifacts here so training is one-time and inference can reuse them.
MODEL_DIR = PROJECT_ROOT / "saved_models"
MODEL_FILE = MODEL_DIR / "model_metadata.json"
PREPROCESSOR_FILE = MODEL_DIR / "fold_1.pth"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
PREDICTIONS_FILE = MODEL_DIR / "predictions_history.json"

FOLD_MODEL_GLOB = "fold_*.pth"
LEGACY_MODEL_FILE = MODEL_DIR / "xgb_model.pkl"
LEGACY_PREPROCESSOR_FILE = MODEL_DIR / "scaler.pkl"
LEGACY_REQUIRED_FILES = (
    MODEL_DIR / "xgb_model.pkl",
    MODEL_DIR / "lgb_model.pkl",
    MODEL_DIR / "pca.pkl",
    MODEL_DIR / "scaler.pkl",
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
