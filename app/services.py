from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from .config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    MODEL_FILE,
    PREDICTIONS_FILE,
    PREPROCESSOR_FILE,
    UPLOAD_DIR,
)
from .model_inference import ModelNotReadyError, predict_biomass
from .schemas import ModelStatusResponse, PredictionRecord, PredictionResponse


def _utc_from_timestamp(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def model_status() -> ModelStatusResponse:
    model_exists = MODEL_FILE.exists()
    pre_exists = PREPROCESSOR_FILE.exists()
    return ModelStatusResponse(
        model_ready=model_exists,
        model_path=str(MODEL_FILE),
        preprocessor_path=str(PREPROCESSOR_FILE),
        model_last_modified_utc=_utc_from_timestamp(MODEL_FILE.stat().st_mtime) if model_exists else None,
        preprocessor_last_modified_utc=_utc_from_timestamp(PREPROCESSOR_FILE.stat().st_mtime)
        if pre_exists
        else None,
        model_size_bytes=MODEL_FILE.stat().st_size if model_exists else None,
        preprocessor_size_bytes=PREPROCESSOR_FILE.stat().st_size if pre_exists else None,
    )


def validate_upload(file: UploadFile) -> None:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file extension. Upload JPG, PNG, or WEBP.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported content type. Upload an image file.")


def save_upload(file: UploadFile) -> Path:
    extension = Path(file.filename or "").suffix.lower()
    filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename

    written = 0
    with destination.open("wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_SIZE_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            buffer.write(chunk)
    return destination


def _read_history() -> list[dict[str, Any]]:
    if not PREDICTIONS_FILE.exists():
        return []
    with PREDICTIONS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else []


def _write_history(rows: list[dict[str, Any]]) -> None:
    with PREDICTIONS_FILE.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def record_prediction(filename: str, saved_image: Path, predicted_biomass: float) -> PredictionRecord:
    now = datetime.now(tz=timezone.utc)
    record = PredictionRecord(
        prediction_id=uuid.uuid4().hex,
        filename=filename,
        saved_image=str(saved_image),
        predicted_biomass=predicted_biomass,
        predicted_at_utc=now,
        model_file=str(MODEL_FILE),
    )
    rows = _read_history()
    rows.append(record.model_dump(mode="json"))
    _write_history(rows)
    return record


def list_predictions(limit: int = 20) -> list[PredictionRecord]:
    rows = _read_history()
    parsed = [PredictionRecord.model_validate(row) for row in rows]
    return list(reversed(parsed[-limit:]))


def get_prediction(prediction_id: str) -> PredictionRecord:
    rows = _read_history()
    for row in rows:
        if row.get("prediction_id") == prediction_id:
            return PredictionRecord.model_validate(row)
    raise HTTPException(status_code=404, detail="Prediction record not found.")


def delete_upload(filename: str) -> dict[str, str]:
    # Prevent path traversal by reducing to basename only.
    safe_name = Path(filename).name
    target = UPLOAD_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found.")
    target.unlink()
    return {"deleted": safe_name}


def predict_from_upload(file: UploadFile) -> PredictionResponse:
    validate_upload(file)
    saved_path = save_upload(file)
    try:
        predicted = predict_biomass(image_path=saved_path)
    except (ModelNotReadyError, NotImplementedError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = record_prediction(
        filename=file.filename or saved_path.name,
        saved_image=saved_path,
        predicted_biomass=predicted,
    )
    return PredictionResponse(
        prediction_id=record.prediction_id,
        filename=record.filename,
        saved_image=record.saved_image,
        predicted_biomass=record.predicted_biomass,
        predicted_at_utc=record.predicted_at_utc,
    )
