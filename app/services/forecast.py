from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

from ..core.config import settings
from ..schemas.results import ForecastResult, HistoryPoint


logger = logging.getLogger(__name__)


def forecast_indicators(df: pd.DataFrame) -> ForecastResult:
    horizon = settings.forecast_horizon
    index: List[str] = []
    predictions: Dict[str, List[float]] = {}
    trends: Dict[str, str] = {}
    summary: Dict[str, str] = {}
    history_tail: Dict[str, List[Dict[str, float]]] = {}

    for column in df.columns:
        series = df[column].dropna()
        if len(series) < settings.min_history_points:
            continue

        try:
            history_tail[column] = _last_observations(series, limit=15)
            latest_series = pd.Series([item["value"] for item in history_tail[column]], index=[item["index"] for item in history_tail[column]])
            index = list(latest_series.index)
            predictions[column] = latest_series.tolist()

            trend_direction = _determine_forecast_trend(latest_series)
            trends[column] = trend_direction

            summary[column] = _build_history_summary(latest_series, trend_direction)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("History extraction failed for %s: %s", column, exc)
            continue

    return ForecastResult(
        horizon=int(horizon),
        index=list(index),
        predictions={key: [float(v) for v in values] for key, values in predictions.items()},
        trends=dict(trends),
        summary=dict(summary),
        history_tail={
            key: [
                HistoryPoint(index=str(item["index"]), value=float(item["value"]))
                for item in value
            ]
            for key, value in history_tail.items()
        }
        or None,
    )


def _determine_forecast_trend(forecast: pd.Series) -> str:
    if forecast.empty:
        return "未知"

    x = np.arange(len(forecast))
    y = forecast.values.astype(float)
    slope, _ = np.polyfit(x, y, 1)
    threshold = np.std(y) * 0.05 if np.std(y) > 0 else 0.0
    if slope > threshold:
        return "预计上升"
    if slope < -threshold:
        return "预计下降"
    return "预计稳定"


def _build_forecast_summary(history: pd.Series, forecast: pd.Series, trend: str) -> str:
    latest = history.iloc[-1]
    future_mean = forecast.mean()
    delta = future_mean - latest
    direction = "增加" if delta > 0 else "下降" if delta < 0 else "持平"
    return f"当前值 {latest:.2f}，预测均值 {future_mean:.2f}，预计{direction}（{trend}）。"


def _last_observations(series: pd.Series, limit: int = 10) -> List[Dict[str, float]]:
    tail = series.tail(limit)
    result: List[Dict[str, float]] = []
    for idx, value in tail.items():
        result.append({
            "index": str(idx),
            "value": float(value),
        })
    return result

def _build_history_summary(series: pd.Series, trend: str) -> str:
    if series.empty:
        return "暂无历史数据"
    latest = series.iloc[-1]
    earliest = series.iloc[0]
    change = latest - earliest
    direction = "上升" if change > 0 else "下降" if change < 0 else "稳定"
    return f"最近15个点：从 {earliest:.2f} 到 {latest:.2f}，总体{direction}（历史趋势：{trend}）。"

