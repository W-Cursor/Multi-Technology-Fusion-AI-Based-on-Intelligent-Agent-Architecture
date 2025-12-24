from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.config import settings
from ..schemas.results import ColumnGroup, ColumnInfo, DatasetMetadata


@dataclass
class PreparedMetadata:
    timestamp_column: str | None
    timestamp_candidates: List[str]
    analysis_columns: List[str]
    key_parameters: List[str]
    row_count: int
    column_count: int
    missing_ratio: float
    target_candidates: List[str]
    grouped_columns: List[ColumnGroup] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "timestamp_column": self.timestamp_column,
            "timestamp_candidates": self.timestamp_candidates,
            "analysis_columns": self.analysis_columns,
            "key_parameters": self.key_parameters,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_ratio": self.missing_ratio,
            "target_candidates": self.target_candidates,
            "grouped_columns": [group.model_dump() for group in self.grouped_columns],
        }


def prepare_dataset(
    raw_df: pd.DataFrame,
    include_columns: Sequence[str] | None = None,
    timestamp_column: str | None = None,
) -> Tuple[pd.DataFrame, PreparedMetadata]:
    df = raw_df.copy()
    df = df.dropna(axis=1, how="all")

    detected_timestamp = timestamp_column or _detect_timestamp_column(df.columns)
    if detected_timestamp and detected_timestamp in df.columns:
        df[detected_timestamp] = pd.to_datetime(df[detected_timestamp], errors="coerce")
        df = df.dropna(subset=[detected_timestamp])
        df = df.sort_values(by=detected_timestamp)
        df = df.set_index(detected_timestamp)

    numeric_df = df.select_dtypes(include=["number", "float", "int"])
    if numeric_df.empty:
        numeric_df = df.apply(pd.to_numeric, errors="coerce")

    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.ffill().bfill()

    analysis_columns = list(numeric_df.columns)
    if include_columns:
        intersection = [col for col in include_columns if col in numeric_df.columns]
        if intersection:
            analysis_columns = intersection
            numeric_df = numeric_df[analysis_columns]

    key_parameters = _identify_key_parameters(analysis_columns)

    target_candidates = [col for col in analysis_columns if col in settings.target_column_candidates]
    if not target_candidates and analysis_columns:
        target_candidates = analysis_columns[:3]

    missing_ratio = float(df.isna().sum().sum()) / float(df.size) if df.size else 0.0

    metadata = PreparedMetadata(
        timestamp_column=detected_timestamp,
        timestamp_candidates=_collect_timestamp_candidates(raw_df.columns),
        analysis_columns=analysis_columns,
        key_parameters=key_parameters,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        missing_ratio=missing_ratio,
        target_candidates=target_candidates,
    )

    metadata.grouped_columns = group_columns(analysis_columns, key_parameters)

    return numeric_df, metadata


def _detect_timestamp_column(columns: Iterable[str]) -> str | None:
    for candidate in settings.timestamp_column_candidates:
        for column in columns:
            if candidate.lower() == column.lower():
                return column

    for column in columns:
        if "time" in column.lower() or "日期" in column:
            return column
    return None


def _collect_timestamp_candidates(columns: Iterable[str]) -> List[str]:
    candidates = []
    for column in columns:
        lowered = column.lower()
        if lowered in [cand.lower() for cand in settings.timestamp_column_candidates]:
            candidates.append(column)
        elif "time" in lowered or "日期" in column:
            candidates.append(column)
    return list(dict.fromkeys(candidates))


def _identify_key_parameters(columns: List[str]) -> List[str]:
    matched = [col for col in columns if "率" in col]
    if len(matched) >= 3:
        return matched[:5]

    keywords = ["Al2O3", "Na2O", "SiO2", "温度", "浓度", "压力", "密度"]
    matched_keywords = [col for col in columns if any(keyword.lower() in col.lower() for keyword in keywords)]
    if matched_keywords:
        return (matched + matched_keywords)[:5]

    return columns[:5]


def group_columns(columns: List[str], suggested_columns: List[str] | None = None) -> List[ColumnGroup]:
    groups: Dict[str, List[ColumnInfo]] = {}
    suggested = set(suggested_columns or [])

    for column in columns:
        prefix = _extract_prefix(column)
        groups.setdefault(prefix, [])
        groups[prefix].append(
            ColumnInfo(
                name=column,
                is_numeric=True,
                suggested=column in suggested,
            )
        )

    return [ColumnGroup(group=group, columns=cols) for group, cols in groups.items()]


def _extract_prefix(name: str) -> str:
    if "%" in name:
        return name.split("%", 1)[0].strip()
    parts = name.split(" ")
    if len(parts) > 1:
        return parts[0]
    return name[:4] if len(name) > 4 else name

