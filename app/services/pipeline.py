from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from ..schemas.results import (
    AgentPayload,
    AnalysisResponse,
    ForecastResult,
    IndicatorAssessment,
    ReportEntry,
)
from .preprocessing import PreparedMetadata, prepare_dataset
from .signals import generate_indicator_assessments
from .forecast import forecast_indicators


logger = logging.getLogger(__name__)


@dataclass
class AnalysisArtifacts:
    cleaned_data: pd.DataFrame
    metadata: PreparedMetadata
    indicator_assessments: List[IndicatorAssessment]
    forecast: ForecastResult
    report_entries: List[ReportEntry]
    agent_payload: AgentPayload


def run_full_analysis(
    raw_df: pd.DataFrame,
    include_columns: List[str] | None = None,
    timestamp_column: str | None = None,
) -> AnalysisResponse:
    artifacts = build_analysis_artifacts(
        raw_df,
        include_columns=include_columns,
        timestamp_column=timestamp_column,
    )

    response = AnalysisResponse(
        report=artifacts.report_entries,
        forecast=artifacts.forecast,
        agent_payload=artifacts.agent_payload,
        metadata=artifacts.metadata.model_dump(),
    )

    return response


def build_analysis_artifacts(
    raw_df: pd.DataFrame,
    include_columns: List[str] | None = None,
    timestamp_column: str | None = None,
) -> AnalysisArtifacts:
    cleaned_df, metadata = prepare_dataset(
        raw_df,
        include_columns=include_columns,
        timestamp_column=timestamp_column,
    )
    logger.debug("Prepared dataset with metadata: %s", metadata)

    indicator_assessments = generate_indicator_assessments(cleaned_df)
    forecast = forecast_indicators(cleaned_df)

    report_entries = _build_report(indicator_assessments, forecast)

    agent_payload = AgentPayload(
        timestamp_column=metadata.timestamp_column,
        key_parameters=metadata.key_parameters,
        indicator_status={assessment.parameter: assessment.status for assessment in indicator_assessments},
        forecast_summary={key: value for key, value in forecast.summary.items()},
    )

    return AnalysisArtifacts(
        cleaned_data=cleaned_df,
        metadata=metadata,
        indicator_assessments=indicator_assessments,
        forecast=forecast,
        report_entries=report_entries,
        agent_payload=agent_payload,
    )


def _build_report(
    indicator_assessments: List[IndicatorAssessment], forecast: ForecastResult
) -> List[ReportEntry]:
    report_entries: List[ReportEntry] = []

    for assessment in indicator_assessments:
        suggestions = []
        if assessment.status == "高":
            suggestions.append("指标偏高，请检查溶出条件或调整原料配比。")
        elif assessment.status == "低":
            suggestions.append("指标偏低，建议关注沉降或热平衡参数。")
        else:
            suggestions.append("指标稳定，保持当前工艺参数。")

        forecast_trend = forecast.trends.get(assessment.parameter)
        summary = forecast.summary.get(assessment.parameter)
        report_entries.append(
            ReportEntry(
                parameter=assessment.parameter,
                current_status=assessment.status,
                trend=assessment.trend,
                latest_value=assessment.latest_value,
                mean_value=assessment.mean_value,
                std_dev=assessment.std_dev,
                forecast_summary=summary or "",
                recommendations=suggestions,
                forecast_trend=forecast_trend or "稳定",
                anomalies=assessment.anomalies,
                influence_factors=assessment.influence_factors or [],
            )
        )

    return report_entries

