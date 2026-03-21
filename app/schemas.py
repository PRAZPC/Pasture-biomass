from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_ready: bool
    model_path: str
    preprocessor_path: str


class ModelStatusResponse(BaseModel):
    model_ready: bool
    model_path: str
    preprocessor_path: str
    model_last_modified_utc: datetime | None
    preprocessor_last_modified_utc: datetime | None
    model_size_bytes: int | None
    preprocessor_size_bytes: int | None


class PredictionResponse(BaseModel):
    prediction_id: str
    filename: str
    saved_image: str
    height_cm: float | None = None
    sampling_date: date | None = None
    predicted_biomass: float
    predicted_at_utc: datetime


class PredictionRecord(PredictionResponse):
    model_file: str
