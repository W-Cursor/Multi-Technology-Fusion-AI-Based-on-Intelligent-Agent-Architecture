"""知识图谱驱动的归因分析服务"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from app.schemas.results import (
    KnowledgeGraphFactor,
    KnowledgeGraphRootCauseRequest,
    KnowledgeGraphRootCauseResponse,
    KnowledgeGraphSeriesPoint,
    KnowledgeGraphTemporalInsight,
)

logger = logging.getLogger(__name__)


BRACKET_TOKEN_PATTERN = re.compile(r"[\[\]{}]")
SANITIZE_STRATEGIES = {"drop_row", "neighbor_fill", "retain_row"}


def _detect_bracket_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Return boolean DataFrame marking cells that contain bracket characters."""

    str_df = df.select_dtypes(include=["object"])
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if str_df.empty:
        return mask

    detected = str_df.apply(
        lambda col: col.fillna("").astype(str).str.contains(BRACKET_TOKEN_PATTERN)
    )
    mask.loc[:, detected.columns] = detected.to_numpy()
    return mask


def generate_knowledge_graph_root_cause(
    df: pd.DataFrame,
    request: KnowledgeGraphRootCauseRequest,
) -> KnowledgeGraphRootCauseResponse:
    """基于上传数据的相关性分析，输出类知识图谱归因结果"""

    if request.target_column not in df.columns:
        raise ValueError(f"未找到目标字段: {request.target_column}")

    timestamp_series = None
    if request.timestamp_column and request.timestamp_column in df.columns:
        timestamp_series = pd.to_datetime(df[request.timestamp_column], errors="coerce")

    cleaning_strategy = request.cleaning_strategy or "drop_row"
    working_df, cleaning_meta = _sanitize_numeric_values(df, cleaning_strategy)

    timestamp_col = None
    if timestamp_series is not None and timestamp_series.notna().any():
        working_df = working_df.assign(__timestamp=timestamp_series).dropna(subset=["__timestamp"])
        working_df = working_df.sort_values("__timestamp")
        timestamp_col = request.timestamp_column

        if timestamp_col != request.target_column and timestamp_col in working_df.columns:
            working_df = working_df.drop(columns=[timestamp_col])

        if request.lookback_window_hours:
            cutoff = (
                working_df["__timestamp"].max()
                - pd.Timedelta(hours=request.lookback_window_hours)
            )
            working_df = working_df[working_df["__timestamp"] >= cutoff]

    if timestamp_col is None and request.timestamp_column:
        working_df = working_df.drop(columns=[request.timestamp_column], errors="ignore")

    numeric_df = working_df.select_dtypes(include=["number", "float", "int"])
    if request.target_column not in numeric_df.columns:
        numeric_df[request.target_column] = pd.to_numeric(
            working_df[request.target_column], errors="coerce"
        )

    target_series = pd.to_numeric(
        numeric_df[request.target_column], errors="coerce"
    ).dropna()
    if target_series.empty:
        raise ValueError("目标字段缺少有效的数值数据，无法进行分析")

    correlation_factors = _compute_correlations(numeric_df, request.target_column)
    regression_scores = _compute_regression_scores(numeric_df, request.target_column)
    model_contrib, model_metrics = _compute_model_contributions(
        numeric_df,
        request.target_column,
        cleaning_strategy,
    )
    factors = _merge_factor_scores(
        correlation_factors,
        regression_scores,
        model_contrib,
        target_series.mean(),
    )

    temporal_map, temporal_insights, temporal_diag = _compute_temporal_insights(
        working_df,
        request.target_column,
        timestamp_col,
    )

    if temporal_map:
        for factor in factors:
            col_key = factor.matched_column or factor.label
            info = temporal_map.get(col_key) or temporal_map.get(factor.label)
            if not info:
                continue
            factor.lead_lag_steps = info.get("lag_steps")
            factor.lead_lag_hours = info.get("lead_hours")
            factor.lead_lag_correlation = info.get("correlation")
            factor.granger_p_value = info.get("granger_p")
            factor.lead_direction = info.get("direction")

    trend_points = _build_trend_points(working_df, request.target_column, timestamp_col)
    summary = _build_summary(request.target_column, factors, temporal_insights)

    processed_columns = [item.matched_column for item in factors if item.matched_column]
    diagnostics: Dict[str, Any] = {
        "method": "Pearson 相关 + 标准化线性回归 + XGBoost",
        "processed_columns": sorted(set(processed_columns)),
    }

    if cleaning_meta:
        diagnostics["cleaning"] = cleaning_meta

    if temporal_diag:
        diagnostics["temporal"] = temporal_diag

    model_cleaning_meta = model_metrics.get("cleaning") if isinstance(model_metrics, dict) else None
    if model_cleaning_meta:
        diagnostics["model_cleaning"] = model_cleaning_meta

    return KnowledgeGraphRootCauseResponse(
        cache_key=request.cache_key,
        target_column=request.target_column,
        target_node=request.target_node or request.target_column,
        target_label=request.target_column,
        anomaly_type=request.anomaly_type,
        timestamp_column=timestamp_col,
        lookback_window_hours=request.lookback_window_hours,
        summary=summary,
        root_causes=factors,
        impacts=factors[:5],
        trend=trend_points,
        diagnostics=diagnostics,
        model_summary=model_metrics,
        temporal_insights=temporal_insights,
    )


def _compute_correlations(df: pd.DataFrame, target_col: str) -> Dict[str, KnowledgeGraphFactor]:
    results: Dict[str, KnowledgeGraphFactor] = {}
    target = pd.to_numeric(df[target_col], errors="coerce")

    for column in df.columns:
        if column == target_col:
            continue

        series = pd.to_numeric(df[column], errors="coerce")
        aligned = pd.concat([target, series], axis=1, join="inner").dropna()
        if aligned.empty or len(aligned) < 3:
            continue

        correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if pd.isna(correlation):
            continue

        latest_value = float(series.dropna().iloc[-1]) if not series.dropna().empty else None
        baseline_value = float(series.mean()) if not series.dropna().empty else None
        delta_value = (
            latest_value - baseline_value if latest_value is not None and baseline_value is not None else None
        )

        results[column] = KnowledgeGraphFactor(
            node_id=column,
            label=column,
            relationship="统计相关",
            graph_weight=None,
            correlation=float(correlation),
            combined_score=abs(float(correlation)),
            latest_value=latest_value,
            baseline_value=baseline_value,
            delta_value=delta_value,
            matched_column=column,
        )

    return results


def _compute_regression_scores(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    features = df.drop(columns=[target_col], errors="ignore")
    if features.empty:
        return {}

    X = features.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    combined = pd.concat([X, y], axis=1)
    problematic_mask = _detect_bracket_strings(combined)
    if problematic_mask.values.any():
        drop_count = int(problematic_mask.any(axis=1).sum())
        logger.warning("Dropping %d rows containing bracketed string values", drop_count)
        combined = combined[~problematic_mask.any(axis=1)]

    combined = combined.apply(lambda series: pd.to_numeric(series, errors="coerce"))
    combined = combined.dropna()
    if combined.shape[0] < 5:
        return {}

    X_clean = combined.drop(columns=[target_col]).values
    y_clean = combined[target_col].values

    X_std = np.std(X_clean, axis=0)
    non_zero_mask = X_std > 0
    if not np.any(non_zero_mask):
        return {}

    X_scaled = np.zeros_like(X_clean)
    X_scaled[:, non_zero_mask] = X_clean[:, non_zero_mask] / X_std[non_zero_mask]
    y_std = np.std(y_clean)
    if y_std == 0:
        return {}

    y_scaled = y_clean / y_std

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    coefs = model.coef_
    columns = features.columns.tolist()
    regression_scores = {
        col: float(coef)
        for col, coef, valid in zip(columns, coefs, non_zero_mask)
        if valid and not np.isnan(coef)
    }

    return regression_scores


def _merge_factor_scores(
    correlation_factors: Dict[str, KnowledgeGraphFactor],
    regression_scores: Dict[str, float],
    model_contrib: Dict[str, Dict[str, float | str | None]],
    target_mean: float,
) -> List[KnowledgeGraphFactor]:
    merged: List[KnowledgeGraphFactor] = []

    for col, factor in correlation_factors.items():
        coef = regression_scores.get(col)
        factor.regression_coef = coef

        model_info = model_contrib.get(col, {})
        shap_value = model_info.get("shap")
        importance = model_info.get("importance")
        direction = model_info.get("direction")

        if coef is not None:
            corr = factor.correlation or 0.0
            factor.influence_score = abs(corr) * 0.4 + abs(coef) * 0.4
        else:
            factor.influence_score = factor.combined_score or 0.0

        if importance:
            factor.influence_score = (factor.influence_score or 0.0) + float(importance) * 0.2

        adjustment, advice = _build_adjustment_advice(
            factor,
            coef if coef is not None else (shap_value or 0.0),
            target_mean,
        )
        factor.adjustment = adjustment
        factor.recommendation = advice

        factor.shap_value = float(shap_value) if shap_value is not None else None
        factor.model_importance = float(importance) if importance is not None else None
        factor.model_direction = direction if direction else None

        merged.append(factor)

    merged.sort(key=lambda item: (item.influence_score or 0.0), reverse=True)
    return merged


def _compute_temporal_insights(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: Optional[str],
    max_lag_steps: int = 2,
    min_samples: int = 30,
) -> tuple[Dict[str, Dict[str, Any]], List[KnowledgeGraphTemporalInsight], Dict[str, Any]]:

    base_df = df.dropna(subset=[target_col]).copy()
    if base_df.empty:
        return {}, [], {"enabled": False, "reason": "目标字段缺少有效数据"}

    diagnostics: Dict[str, Any] = {"enabled": True}

    if "__timestamp" in base_df.columns:
        base_df = base_df.sort_values("__timestamp").reset_index(drop=True)
        diagnostics["note"] = "滞后分析使用时间列排序，滞后单位为记录组"
    else:
        base_df = base_df.sort_index(ignore_index=True)
        diagnostics["note"] = "数据缺少时间列，按记录顺序估算滞后"

    temporal_map: Dict[str, Dict[str, Any]] = {}
    insights: List[KnowledgeGraphTemporalInsight] = []
    statsmodels_status = "available"

    try:
        from statsmodels.tsa.stattools import grangercausalitytests  # type: ignore
    except ImportError:
        grangercausalitytests = None  # type: ignore
        statsmodels_status = "statsmodels 未安装"

    for column in df.columns:
        if column in {target_col, "__timestamp"}:
            continue

        if df[column].dropna().empty:
            continue

        joined = (
            base_df[[target_col]]
            .join(df[[column]], how="inner")
            .dropna()
        )

        joined[target_col] = pd.to_numeric(joined[target_col], errors="coerce")
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
        joined = joined.dropna(subset=[target_col, column])

        if len(joined) < min_samples:
            continue

        y_series = joined[target_col]
        x_series = joined[column]

        best_corr = 0.0
        best_lag = None
        best_samples = 0

        for lag in range(1, max_lag_steps + 1):
            lagged_x = x_series.shift(lag)
            aligned = pd.concat([y_series, lagged_x], axis=1).dropna()
            if len(aligned) < min_samples:
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if pd.isna(corr):
                continue
            if best_lag is None or abs(corr) > abs(best_corr):
                best_corr = float(corr)
                best_lag = lag
                best_samples = len(aligned)

        if best_lag is None:
            continue

        direction = "positive" if best_corr >= 0 else "negative"

        granger_p = None
        if grangercausalitytests is not None:
            max_granger_lag = min(best_lag + 1, max_lag_steps)
            granger_data = joined[[target_col, column]].dropna()
            if len(granger_data) >= max_granger_lag + 5:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="verbose is deprecated since functions should not print results",
                            category=FutureWarning,
                        )
                        tests = grangercausalitytests(
                            granger_data.values,
                            maxlag=max_granger_lag,
                            verbose=False,
                        )
                    p_values = []
                    for result in tests.values():
                        ssr_ftest = result[0].get("ssr_ftest")
                        if ssr_ftest and len(ssr_ftest) > 1:
                            p_values.append(ssr_ftest[1])
                    if p_values:
                        granger_p = float(min(p_values))
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Granger test failed for %s: %s", column, exc)

        effect_direction = "升高" if direction == "positive" else "降低"
        feature_change = "上升"
        description = (
            f"{column} 提前 {best_lag} 组变化会推动 {target_col} {effect_direction}"
            f"（滞后相关 {best_corr:+.2f}"
        )
        if granger_p is not None:
            description += f", Granger p={granger_p:.3f}"
        description += "）"

        temporal_map[column] = {
            "lag_steps": best_lag,
            "lead_hours": float(best_lag),
            "correlation": best_corr,
            "direction": direction,
            "samples": best_samples,
            "granger_p": granger_p,
        }

        insights.append(
            KnowledgeGraphTemporalInsight(
                parameter=column,
                lead_hours=float(best_lag),
                correlation=best_corr,
                direction=direction,
                granger_p_value=granger_p,
                sample_size=best_samples,
                summary=description,
            )
        )

    insights.sort(key=lambda item: abs(item.correlation or 0.0), reverse=True)

    diagnostics.update(
        {
            "enabled": bool(insights),
            "max_lag_steps": max_lag_steps,
            "statsmodels": statsmodels_status,
            "top": [item.summary for item in insights[:5]],
        }
    )

    if not insights:
        diagnostics["reason"] = "未识别到显著的滞后相关关系"

    return temporal_map, insights, diagnostics


def _build_adjustment_advice(
    factor: KnowledgeGraphFactor,
    coef: float,
    target_mean: float,
) -> tuple[str | None, str | None]:
    if factor.latest_value is None or factor.baseline_value is None:
        return None, None

    delta = factor.delta_value or 0.0
    direction = "正向" if coef >= 0 else "负向"

    if abs(coef) < 1e-6:
        return "保持观察", f"{factor.label} 当前对目标影响有限，建议关注趋势。"

    if coef > 0:
        if delta > 0:
            return "下调", f"{factor.label} 高于均值 {delta:.2f}，正向拉升目标，建议适度下调。"
        if delta < -1e-6:
            return "保持/上调", f"{factor.label} 低于均值 {abs(delta):.2f}，适当维持或上调可支撑目标。"
        return "保持观察", f"{factor.label} 接近均值，维持稳定即可。"

    if coef < 0:
        if delta > 0:
            return "下调", f"{factor.label} 高于均值 {delta:.2f}，负向影响目标，建议下调。"
        if delta < -1e-6:
            return "上调", f"{factor.label} 低于均值 {abs(delta):.2f}，提升有助于目标恢复。"
        return "保持观察", f"{factor.label} 接近均值，暂时维持。"

    return direction, None


def _compute_model_contributions(
    df: pd.DataFrame,
    target_col: str,
    cleaning_strategy: str = "drop_row",
) -> tuple[Dict[str, Dict[str, float | str | None]], Dict[str, Any]]:
    try:
        import xgboost as xgb
    except ImportError:
        return {}, {"enabled": False, "reason": "未安装 xgboost，跳过模型增强"}

    features = df.drop(columns=[target_col], errors="ignore")
    if features.empty:
        return {}, {"enabled": False, "reason": "缺少可用于建模的特征"}

    object_cols = [col for col in features.columns if features[col].dtype == object]
    if object_cols:
        logger.warning("Non-numeric columns detected prior to cleaning: %s", object_cols)

    combined = pd.concat([features, df[target_col]], axis=1)

    raw_bracket_mask = _detect_bracket_strings(combined)
    if raw_bracket_mask.values.any():
        sample_rows = combined[raw_bracket_mask.any(axis=1)].head(3)
        logger.warning(
            "Raw combined dataset contains bracketed strings before sanitization: samples=%s",
            sample_rows.to_dict(orient="records"),
        )

    combined_clean, model_clean_meta = _sanitize_numeric_values(combined, strategy=cleaning_strategy)

    if model_clean_meta.get("rows_dropped"):
        logger.warning(
            "Dropping %d rows during numeric sanitization for model training",
            model_clean_meta["rows_dropped"],
        )

    if combined_clean.shape[0] < 30:
        return {}, {
            "enabled": False,
            "reason": "有效样本不足，需至少 30 条记录",
            "cleaning": model_clean_meta,
        }

    X_mat = combined_clean.drop(columns=[target_col])
    y_vec = combined_clean[target_col]

    X_mat = X_mat.fillna(0.0)
    y_vec = y_vec.fillna(0.0).astype(float)

    try:
        X_numpy = np.asarray(X_mat.values, dtype=np.float64)
    except ValueError as exc:
        string_cells: Dict[str, str] = {}
        object_subset = X_mat.select_dtypes(include=["object"])
        for col in object_subset.columns:
            sample = object_subset[col].dropna().astype(str).head(3).tolist()
            if sample:
                string_cells[col] = sample
        logger.error("Non-numeric residues detected before XGBoost: cols=%s", string_cells)
        raise ValueError(
            "模型输入仍包含无法转换的字符串，请检查日志中的列示例"
        ) from exc

    X_mat = pd.DataFrame(X_numpy, columns=X_mat.columns, index=X_mat.index)

    dtrain = xgb.DMatrix(X_numpy, label=y_vec.values, feature_names=list(X_mat.columns))
    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "verbosity": 0,
    }
    num_round = 200
    model = xgb.train(params, dtrain, num_boost_round=num_round)

    preds = model.predict(dtrain)
    r2 = float(r2_score(y_vec, preds))
    mae = float(mean_absolute_error(y_vec, preds))

    gain_importance = model.get_score(importance_type="gain") or {}
    total_gain = sum(gain_importance.values()) or 1.0

    contributions: Dict[str, Dict[str, float | str | None]] = {}

    shap_strength_vector = None
    shap_direction_vector = None
    shap_mean = None
    shap_reason: str | None = None
    shap_source = "shap"
    try:
        import shap  # pylint: disable=import-error

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_numpy)
        if isinstance(shap_values, list):  # 分类模型时 shap 返回列表
            shap_values = shap_values[0]
        shap_direction_vector = np.mean(shap_values, axis=0)
        shap_abs = np.abs(shap_values)
        shap_mean = np.mean(shap_abs, axis=0)
        shap_strength_vector = np.percentile(shap_abs, 80, axis=0)
    except ImportError:
        shap_reason = "shap 未安装"
        shap_source = "unavailable"
    except Exception as exc:  # pylint: disable=broad-except
        shap_strength_vector = None
        shap_direction_vector = None
        shap_mean = None
        shap_reason = f"SHAP 计算失败：{exc}"
        shap_source = "error"

        try:
            contribs = model.predict(dtrain, pred_contribs=True)
            if contribs.ndim == 2 and contribs.shape[1] >= len(X_mat.columns):
                contrib_matrix = contribs[:, : len(X_mat.columns)]
                shap_direction_vector = np.mean(contrib_matrix, axis=0)
                contrib_abs = np.abs(contrib_matrix)
                shap_mean = np.mean(contrib_abs, axis=0)
                shap_strength_vector = np.percentile(contrib_abs, 80, axis=0)
                shap_reason = "shap 库计算失败，已回退使用 XGBoost pred_contribs 结果"
                shap_source = "pred_contribs"
        except Exception as fallback_exc:  # pylint: disable=broad-except
            logger.error("XGBoost pred_contribs fallback failed: %s", fallback_exc)
            shap_source = "unavailable"

    for idx, column in enumerate(X_mat.columns):
        importance = gain_importance.get(column, 0.0) / total_gain
        shap_value = (
            float(shap_strength_vector[idx])
            if shap_strength_vector is not None
            else (float(shap_mean[idx]) if shap_mean is not None else None)
        )
        direction_score = (
            float(shap_direction_vector[idx])
            if shap_direction_vector is not None
            else (shap_value if shap_value is not None else None)
        )
        direction = None
        if direction_score is not None:
            direction = "positive" if direction_score >= 0 else "negative"

        contributions[column] = {
            "importance": float(importance),
            "shap": shap_value,
            "direction": direction,
        }

    metrics = {
        "enabled": True,
        "model": "XGBoostRegressor",
        "samples": int(len(y_vec)),
        "r2": r2,
        "mae": mae,
        "shap": shap_reason is None,
        "shap_reason": shap_reason,
        "shap_source": shap_source,
        "cleaning": model_clean_meta,
    }

    return contributions, metrics


def _sanitize_numeric_values(
    df: pd.DataFrame,
    strategy: str = "drop_row",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if strategy not in SANITIZE_STRATEGIES:
        logger.warning("Unknown cleaning strategy %s, fallback to drop_row", strategy)
        strategy = "drop_row"

    replacements = [
        (r"[\[\](){}]", " "),
        (r"[‘’＇'\"]", " "),
        (",", " "),
        ("\u3000", " "),
    ]
    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def _clean_value(val: Any) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan

        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)

        if isinstance(val, (list, tuple)):
            for item in val:
                result = _clean_value(item)
                if not np.isnan(result):
                    return result
            return np.nan

        text = str(val).strip()
        if not text:
            return np.nan

        for pat, repl in replacements:
            text = re.sub(pat, repl, text)

        matches = number_pattern.findall(text)
        for candidate in matches:
            try:
                return float(candidate)
            except ValueError:
                continue
        return np.nan

    cleaned = df.copy()
    invalid_counts: Dict[str, int] = {}

    for col in cleaned.columns:
        original_col = cleaned[col]
        cleaned_col = original_col.apply(_clean_value)
        invalid_mask = cleaned_col.isna() & original_col.notna()
        invalid_counts[col] = int(invalid_mask.sum())
        cleaned[col] = cleaned_col

    numeric_df = cleaned.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    rows_before = len(df)
    cells_flagged = int((numeric_df.isna() & df.notna()).sum().sum())
    columns_flagged = sorted(col for col, count in invalid_counts.items() if count)

    filled_cells = 0

    if strategy == "drop_row":
        cleaned_df = numeric_df.dropna()
    elif strategy == "neighbor_fill":
        filled_df = numeric_df.fillna(method="ffill").fillna(method="bfill")
        filled_cells = int((numeric_df.isna() & ~filled_df.isna()).sum().sum())
        cleaned_df = filled_df
    else:  # retain_row
        column_means = numeric_df.mean()
        filled_df = numeric_df.fillna(column_means)
        filled_cells = int((numeric_df.isna() & ~filled_df.isna()).sum().sum())
        cleaned_df = filled_df

    if strategy != "drop_row" and cleaned_df.isna().values.any():
        additional_fill = int(cleaned_df.isna().sum().sum())
        if additional_fill:
            cleaned_df = cleaned_df.fillna(0.0)
            filled_cells += additional_fill

    rows_after = len(cleaned_df)
    metadata = {
        "strategy": strategy,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_dropped": rows_before - rows_after,
        "cells_flagged": cells_flagged,
        "columns_flagged": columns_flagged,
        "filled_cells": filled_cells,
        "remaining_nan": int(cleaned_df.isna().sum().sum()),
    }

    return cleaned_df, metadata


def _build_trend_points(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: Optional[str],
    limit: int = 60,
) -> List[KnowledgeGraphSeriesPoint]:
    points: List[KnowledgeGraphSeriesPoint] = []

    if target_column not in df.columns:
        return points

    series = pd.to_numeric(df[target_column], errors="coerce")
    if timestamp_column and "__timestamp" in df.columns:
        timestamps = df["__timestamp"].astype("datetime64[ns]")
    elif timestamp_column and timestamp_column in df.columns:
        timestamps = pd.to_datetime(df[timestamp_column], errors="coerce")
    else:
        timestamps = pd.RangeIndex(start=0, stop=len(series))

    tail_df = pd.DataFrame({"timestamp": timestamps, "value": series}).dropna().tail(limit)

    for _, row in tail_df.iterrows():
        ts = row["timestamp"]
        ts_value = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        points.append(
            KnowledgeGraphSeriesPoint(timestamp=ts_value, value=float(row["value"]))
        )

    return points


def _build_summary(
    target_label: str,
    factors: List[KnowledgeGraphFactor],
    temporal_insights: Optional[List[KnowledgeGraphTemporalInsight]] = None,
) -> str:
    if not factors:
        return f"暂未识别到与 {target_label} 显著相关的参数，请检查数据质量。"

    top = factors[0]
    msg = (
        f"{target_label} 与 {top.label} 存在较强相关性"
        f"（|r| = {top.combined_score:.2f}）。"
    )
    if top.delta_value is not None and abs(top.delta_value) > 1e-6:
        msg += f" 最近偏离均值 {top.delta_value:.2f}。"

    if len(factors) > 1:
        others = ", ".join(item.label for item in factors[1:4])
        msg += f" 其它相关参数包括：{others}。"

    adjustments = [
        f"{factor.label}:{factor.adjustment}" for factor in factors[:3] if factor.adjustment
    ]
    if adjustments:
        msg += "\n" + f"建议优先动作：{'；'.join(adjustments)}。"

    if temporal_insights:
        top_temporal = temporal_insights[:2]
        if top_temporal:
            temporal_text = "；".join(item.summary for item in top_temporal)
            msg += "\n" + f"时序洞察：{temporal_text}。"

    return msg

