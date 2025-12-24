from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import settings
from ..schemas.results import IndicatorAssessment


def generate_indicator_assessments(df: pd.DataFrame) -> list[IndicatorAssessment]:
    assessments: list[IndicatorAssessment] = []
    correlations = df.corr(method="pearson").fillna(0.0)

    for column in df.columns:
        series = df[column].dropna()
        if len(series) < max(10, settings.min_history_points // 2):
            continue

        latest_value = float(series.iloc[-1]) if not series.empty else None
        mean_value = float(series.mean()) if not series.empty else None
        std_value = float(series.std(ddof=0)) if not series.empty else None

        status = _classify_status(latest_value, mean_value, std_value)
        trend = _infer_trend(series)

        influence_factors = _top_correlated_factors(correlations, column)

        anomalies = _detect_local_extremes(series)

        assessments.append(
            IndicatorAssessment(
                parameter=column,
                status=status,
                trend=trend,
                latest_value=latest_value,
                mean_value=mean_value,
                std_dev=std_value,
                influence_factors=influence_factors,
                anomalies=anomalies,
            )
        )

    return assessments


def _classify_status(
    latest: float | None,
    mean_val: float | None,
    std_val: float | None,
    upper_z: float = 1.5,
    lower_z: float = -1.5,
) -> str:
    if latest is None or mean_val is None or std_val is None or std_val == 0:
        return "正常"

    z_score = (latest - mean_val) / std_val
    if z_score >= upper_z:
        return "高"
    if z_score <= lower_z:
        return "低"
    return "正常"


def _infer_trend(series: pd.Series, window: int = 10) -> str:
    tail = series.tail(window)
    if len(tail) < 3:
        return "稳定"

    x = np.arange(len(tail))
    y = tail.values.astype(float)
    slope, _ = np.polyfit(x, y, 1)

    threshold = np.std(y) * 0.05 if np.std(y) > 0 else 0.0

    if slope > threshold:
        return "上升"
    if slope < -threshold:
        return "下降"
    return "稳定"


def _top_correlated_factors(correlations: pd.DataFrame, column: str, limit: int = 3) -> list[str]:
    if column not in correlations:
        return []

    series = correlations[column].drop(labels=[column], errors="ignore")
    if series.empty:
        return []

    series = series.abs().sort_values(ascending=False)
    strong = series[series >= 0.6]
    if strong.empty:
        strong = series.head(limit)

    return [str(name) for name in strong.head(limit).index.tolist()]


def _detect_local_extremes(series: pd.Series, window: int = 5, z_threshold: float = 2.0) -> list[dict[str, float]]:
    if len(series) < window:
        return []

    values = series.values.astype(float)
    mean = values.mean()
    std = values.std(ddof=0) or 1.0

    anomalies: list[dict[str, float]] = []
    for idx in range(1, len(values) - 1):
        prev_val = values[idx - 1]
        curr_val = values[idx]
        next_val = values[idx + 1]

        if curr_val > prev_val and curr_val > next_val:
            z = (curr_val - mean) / std
            if z >= z_threshold:
                anomalies.append({
                    "type": "peak",
                    "index": str(series.index[idx]),
                    "value": float(curr_val),
                    "z_score": float(z),
                })
        elif curr_val < prev_val and curr_val < next_val:
            z = (curr_val - mean) / std
            if z <= -z_threshold:
                anomalies.append({
                    "type": "trough",
                    "index": str(series.index[idx]),
                    "value": float(curr_val),
                    "z_score": float(z),
                })

    return anomalies












