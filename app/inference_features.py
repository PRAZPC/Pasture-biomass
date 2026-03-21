from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

CONTEXT_FEATURE_COLUMNS = [
    "height_cm",
    "height_log",
    "month",
    "day_of_year",
    "week_of_year",
    "season",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "height_month_sin",
    "height_day_cos",
]


def build_context_feature_vector(height_cm: float, sampling_date: date) -> np.ndarray:
    if height_cm <= 0:
        raise ValueError("Height must be greater than 0 cm.")

    month = float(sampling_date.month)
    day_of_year = float(sampling_date.timetuple().tm_yday)
    week_of_year = float(sampling_date.isocalendar()[1])
    season = float(((sampling_date.month % 12) + 3) // 3)
    month_sin = float(np.sin(2.0 * np.pi * month / 12.0))
    month_cos = float(np.cos(2.0 * np.pi * month / 12.0))
    day_sin = float(np.sin(2.0 * np.pi * day_of_year / 366.0))
    day_cos = float(np.cos(2.0 * np.pi * day_of_year / 366.0))
    height_log = float(np.log1p(height_cm))

    values = np.array(
        [
            float(height_cm),
            height_log,
            month,
            day_of_year,
            week_of_year,
            season,
            month_sin,
            month_cos,
            day_sin,
            day_cos,
            float(height_cm) * month_sin,
            float(height_cm) * day_cos,
        ],
        dtype=np.float32,
    )
    return values


def build_context_feature_frame(
    df: pd.DataFrame,
    *,
    height_col: str = "Height_Ave_cm",
    date_col: str = "Sampling_Date",
) -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    heights = df[height_col].astype(np.float32)

    month = dates.dt.month.astype(np.float32)
    day_of_year = dates.dt.dayofyear.astype(np.float32)
    week_of_year = dates.dt.isocalendar().week.astype(np.float32)
    season = (((dates.dt.month % 12) + 3) // 3).astype(np.float32)
    month_sin = np.sin(2.0 * np.pi * month / 12.0).astype(np.float32)
    month_cos = np.cos(2.0 * np.pi * month / 12.0).astype(np.float32)
    day_sin = np.sin(2.0 * np.pi * day_of_year / 366.0).astype(np.float32)
    day_cos = np.cos(2.0 * np.pi * day_of_year / 366.0).astype(np.float32)
    height_log = np.log1p(heights.clip(lower=0.0)).astype(np.float32)

    context = pd.DataFrame(
        {
            "height_cm": heights,
            "height_log": height_log,
            "month": month,
            "day_of_year": day_of_year,
            "week_of_year": week_of_year,
            "season": season,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "height_month_sin": (heights * month_sin).astype(np.float32),
            "height_day_cos": (heights * day_cos).astype(np.float32),
        }
    )

    return context[CONTEXT_FEATURE_COLUMNS].astype(np.float32)
