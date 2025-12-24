from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..schemas.results import (
    AutoInfluenceFactor,
    ExpertFactorEntry,
    ExpertSchemeCreate,
    ExpertSchemeInfo,
    ExpertSchemeUpdate,
    HistoryPoint,
    ParameterAnalysisRequest,
    ParameterAnalysisResponse,
    RecommendationEntry,
)
from .pipeline import build_analysis_artifacts


class ExpertKnowledgeStore:
    """In-memory storage for expert knowledge schemes."""

    def __init__(self) -> None:
        self._schemes: Dict[str, ExpertSchemeInfo] = {}

    def list(self, target: str | None = None) -> List[ExpertSchemeInfo]:
        schemes = list(self._schemes.values())
        if target:
            schemes = [item for item in schemes if item.target_parameter == target]
        return sorted(schemes, key=lambda item: item.updated_at or "", reverse=True)

    def get(self, scheme_id: str) -> Optional[ExpertSchemeInfo]:
        return self._schemes.get(scheme_id)

    def get_default_for(self, parameter: str | None) -> Optional[ExpertSchemeInfo]:
        if not parameter:
            return None
        schemes = self.list(parameter)
        return schemes[0] if schemes else None

    def create(self, payload: ExpertSchemeCreate) -> ExpertSchemeInfo:
        scheme_id = uuid.uuid4().hex
        now = _now_iso()
        model = ExpertSchemeInfo(
            scheme_id=scheme_id,
            updated_at=now,
            **_normalise_scheme_payload(payload.model_dump(exclude={"cache_key"})),
        )
        self._schemes[scheme_id] = model
        return model

    def update(self, scheme_id: str, payload: ExpertSchemeUpdate) -> ExpertSchemeInfo:
        if scheme_id not in self._schemes:
            raise KeyError("scheme not found")
        now = _now_iso()
        model = ExpertSchemeInfo(
            scheme_id=scheme_id,
            updated_at=now,
            **_normalise_scheme_payload(payload.model_dump(exclude={"cache_key"})),
        )
        self._schemes[scheme_id] = model
        return model

    def delete(self, scheme_id: str) -> None:
        self._schemes.pop(scheme_id, None)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _normalise_scheme_payload(data: Dict[str, object]) -> Dict[str, object]:
    factors: List[Dict[str, object]] = []
    for item in data.get("factors", []) or []:
        parameter = str(item.get("parameter", "")).strip()
        if not parameter:
            continue
        weight = item.get("weight")
        if weight is None:
            numeric_weight: Optional[float] = None
        else:
            try:
                numeric = float(weight)
                if math.isnan(numeric) or math.isinf(numeric):
                    numeric_weight = None
                else:
                    numeric_weight = numeric
            except (TypeError, ValueError):
                numeric_weight = None
        note = item.get("note")
        factors.append(
            ExpertFactorEntry(
                parameter=parameter,
                weight=numeric_weight,
                note=str(note).strip() if note else None,
            ).model_dump()
        )

    controlled = []
    for value in data.get("controlled_parameters", []) or []:
        value_str = str(value).strip()
        if value_str and value_str not in controlled:
            controlled.append(value_str)

    return {
        "name": str(data.get("name", "")).strip() or "未命名方案",
        "target_parameter": str(data.get("target_parameter", "")).strip(),
        "description": (str(data.get("description", "")).strip() or None),
        "controlled_parameters": controlled,
        "factors": [ExpertFactorEntry.model_validate(item) for item in factors],
    }


def generate_parameter_analysis(
    df: pd.DataFrame,
    request: ParameterAnalysisRequest,
    store: Optional[ExpertKnowledgeStore] = None,
) -> ParameterAnalysisResponse:
    artifacts = build_analysis_artifacts(df)

    if request.parameter not in artifacts.cleaned_data.columns:
        raise ValueError("目标参数不在可分析字段中，请检查选择的参数。")

    sliced_df = _slice_time_window(
        artifacts.cleaned_data,
        request.time_range,
        artifacts.metadata.timestamp_column,
    )

    if request.parameter not in sliced_df.columns:
        raise ValueError("所选时间范围内缺少目标参数的数据。")

    target_series = sliced_df[request.parameter].dropna()
    if target_series.empty:
        raise ValueError("目标参数缺少有效数据，请尝试放宽时间范围。")

    history = _build_history_points(
        target_series,
        request.horizon or 0,
    )

    assessment = _locate_assessment(artifacts.indicator_assessments, request.parameter)
    report_entry = _locate_report_entry(artifacts.report_entries, request.parameter)

    recommendations = _build_recommendations(report_entry)
    auto_factors = _compute_auto_factors(sliced_df, request.parameter)

    expert_scheme = store.get_default_for(request.parameter) if store else None

    mean_value = float(target_series.mean()) if len(target_series) else None
    std_value = float(target_series.std(ddof=0)) if len(target_series) > 1 else None
    latest_value = float(target_series.iloc[-1]) if len(target_series) else None

    response = ParameterAnalysisResponse(
        parameter=request.parameter,
        status=assessment.status if assessment else "未知",
        trend=assessment.trend if assessment else "稳定",
        latest_value=latest_value,
        mean_value=mean_value,
        std_dev=std_value,
        forecast_trend=artifacts.forecast.trends.get(request.parameter),
        forecast_summary=artifacts.forecast.summary.get(request.parameter),
        auto_factors=auto_factors,
        history=history,
        expert_scheme=expert_scheme,
        recommendations=recommendations,
        available_parameters=list(artifacts.metadata.analysis_columns),
        timestamp_column=artifacts.metadata.timestamp_column,
    )

    return response


def _locate_assessment(
    assessments: Iterable[ExpertFactorEntry] | Iterable,
    parameter: str,
):
    for item in assessments:
        if getattr(item, "parameter", None) == parameter:
            return item
    return None


def _locate_report_entry(report_entries: Iterable, parameter: str):
    for entry in report_entries:
        if getattr(entry, "parameter", None) == parameter:
            return entry
    return None


def _build_recommendations(report_entry) -> List[RecommendationEntry]:
    if not report_entry or not getattr(report_entry, "recommendations", None):
        return []
    recommendations: List[RecommendationEntry] = []
    for idx, text in enumerate(report_entry.recommendations, start=1):
        recommendations.append(
            RecommendationEntry(
                title=f"建议 {idx}",
                detail=text,
            )
        )
    return recommendations


def _compute_auto_factors(df: pd.DataFrame, parameter: str) -> List[AutoInfluenceFactor]:
    if parameter not in df.columns or df.shape[1] <= 1:
        return []

    correlations = df.corr(method="pearson").get(parameter)
    if correlations is None:
        return []

    correlations = correlations.drop(labels=[parameter], errors="ignore").dropna()
    if correlations.empty:
        return []

    items = []
    for name, corr_value in correlations.items():
        score = abs(float(corr_value))
        direction = "正向" if corr_value >= 0 else "负向"
        items.append(
            (score, AutoInfluenceFactor(name=name, score=score, direction=direction, notes=f"Pearson {corr_value:.2f}"))
        )

    items.sort(key=lambda item: item[0], reverse=True)
    top_items = [item[1] for item in items[:6]]

    if not top_items:
        return []

    max_score = max(item.score for item in top_items) or 1.0
    normalised: List[AutoInfluenceFactor] = []
    for item in top_items:
        normalised.append(
            AutoInfluenceFactor(
                name=item.name,
                score=item.score / max_score,
                direction=item.direction,
                notes=item.notes,
            )
        )
    return normalised


def _slice_time_window(
    df: pd.DataFrame,
    time_range: Optional[str],
    timestamp_column: Optional[str],
) -> pd.DataFrame:
    if not time_range:
        return df

    if time_range.endswith("d") and isinstance(df.index, pd.DatetimeIndex):
        try:
            days = int(time_range[:-1])
        except ValueError:
            days = 0
        if days > 0:
            cutoff = df.index.max() - timedelta(days=days)
            filtered = df.loc[df.index >= cutoff]
            if not filtered.empty:
                return filtered

    if time_range.endswith("h") and isinstance(df.index, pd.DatetimeIndex):
        try:
            hours = int(time_range[:-1])
        except ValueError:
            hours = 0
        if hours > 0:
            cutoff = df.index.max() - timedelta(hours=hours)
            filtered = df.loc[df.index >= cutoff]
            if not filtered.empty:
                return filtered

    # fallback: limit by number of rows
    try:
        limit = int(time_range)
    except (TypeError, ValueError):
        limit = 200
    limit = max(limit, 50)
    return df.tail(limit)


def _build_history_points(series: pd.Series, horizon: int) -> List[HistoryPoint]:
    limit = max(30, min(len(series), horizon or len(series)))
    tail = series.tail(limit)
    history: List[HistoryPoint] = []
    for idx, value in tail.items():
        if pd.isna(value):
            continue
        history.append(HistoryPoint(index=str(idx), value=float(value)))
    return history


