from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (
    STATIC_DIR,
    TEMPLATES_DIR,
    ensure_directories,
)
from .schemas import HealthResponse, ModelStatusResponse, PredictionRecord, PredictionResponse
from .services import (
    delete_upload,
    get_prediction,
    list_predictions,
    model_status,
    parse_sampling_date,
    predict_from_upload,
)

app = FastAPI(title="Biomass Estimator App", version="1.0.0")

ensure_directories()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    status = model_status()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_path": status.model_path,
            "preprocessor_path": status.preprocessor_path,
            "model_ready": status.model_ready,
            "height_cm": None,
            "sampling_date": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_page(
    request: Request,
    file: UploadFile = File(...),
    height_cm: float = Form(...),
    sampling_date: str = Form(...),
) -> HTMLResponse:
    response = None
    error = None
    parsed_sampling_date: date | None = None
    try:
        parsed_sampling_date = parse_sampling_date(sampling_date)
        response = predict_from_upload(
            file=file,
            height_cm=height_cm,
            sampling_date=parsed_sampling_date,
        )
    except HTTPException as exc:
        error = str(exc.detail)
    except Exception as exc:
        error = str(exc)
    status = model_status()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "uploaded_image": f"/static/uploads/{Path(response.saved_image).name}" if response else None,
            "uploaded_filename": response.filename if response else file.filename,
            "prediction": response.predicted_biomass if response else None,
            "error": error,
            "model_path": status.model_path,
            "preprocessor_path": status.preprocessor_path,
            "model_ready": status.model_ready,
            "height_cm": response.height_cm if response else height_cm,
            "sampling_date": (
                response.sampling_date.isoformat()
                if response
                else (parsed_sampling_date.isoformat() if parsed_sampling_date else sampling_date)
            ),
        },
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_api(
    file: UploadFile = File(...),
    height_cm: float = Form(...),
    sampling_date: str = Form(...),
) -> PredictionResponse:
    return predict_from_upload(
        file=file,
        height_cm=height_cm,
        sampling_date=parse_sampling_date(sampling_date),
    )


@app.get("/api/model/status", response_model=ModelStatusResponse, tags=["Model"])
def model_status_api() -> ModelStatusResponse:
    return model_status()


@app.get("/api/predictions", response_model=list[PredictionRecord], tags=["Prediction"])
def list_predictions_api(limit: int = Query(default=20, ge=1, le=200)) -> list[PredictionRecord]:
    return list_predictions(limit=limit)


@app.get("/api/predictions/{prediction_id}", response_model=PredictionRecord, tags=["Prediction"])
def get_prediction_api(prediction_id: str) -> PredictionRecord:
    return get_prediction(prediction_id)


@app.delete("/api/uploads/{filename}", tags=["Uploads"])
def delete_upload_api(filename: str) -> dict[str, str]:
    return delete_upload(filename)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    status = model_status()
    return HealthResponse(
        status="ok",
        model_ready=status.model_ready,
        model_path=status.model_path,
        preprocessor_path=status.preprocessor_path,
    )
