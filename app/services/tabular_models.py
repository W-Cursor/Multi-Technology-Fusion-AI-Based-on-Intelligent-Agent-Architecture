from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .lstm_model import LSTMModelBundle, predict_with_lstm, train_lstm_model

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore

TabPFNRegressor = None
try:  # pragma: no cover - optional dependency
    from tabpfn import TabPFNRegressor as _TabPFNRegressor  # type: ignore

    TabPFNRegressor = _TabPFNRegressor  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    try:
        from tabpfn.scripts.transformer_prediction_interface import (  # type: ignore
            TabPFNRegressor as _TabPFNRegressor,  # type: ignore
        )

        TabPFNRegressor = _TabPFNRegressor  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        TabPFNRegressor = None  # type: ignore


MIN_SAMPLE_THRESHOLD = 20


@dataclass
class TabularModelBundle:
    model_type: str
    model: Any
    input_columns: List[str]
    target_column: str
    lag_map: Dict[str, str] = field(default_factory=dict)
    feature_schema: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


LagColumns = Sequence[str] | None
ParamsDict = Dict[str, Any] | None


def train_model(
    model_type: str,
    df: pd.DataFrame,
    input_columns: Sequence[str],
    target_column: str,
    lag_columns: LagColumns = None,
    params: ParamsDict = None,
) -> Tuple[TabularModelBundle, Dict[str, float]]:
    model_type = (model_type or "").lower().strip()
    params = params or {}
    lag_columns = [col for col in (lag_columns or []) if col in input_columns]

    if not input_columns:
        raise ValueError("请至少选择一个输入字段用于模型训练。")

    if not target_column:
        raise ValueError("请选择一个输出字段用于模型训练。")

    if model_type == "xgboost":
        return _train_xgboost(df, input_columns, target_column, lag_columns, params)
    if model_type == "tabpfn":
        return _train_tabpfn(df, input_columns, target_column, lag_columns, params)
    if model_type == "lstm":
        return _train_lstm(df, input_columns, target_column, lag_columns, params)

    raise ValueError(f"不支持的模型类型: {model_type}")


def predict_model(bundle: TabularModelBundle, inputs: Sequence[float]) -> float:
    expected = len(bundle.input_columns)
    if len(inputs) != expected:
        raise ValueError(f"输入参数数量应为 {expected}，实际为 {len(inputs)}")

    if bundle.model_type == "lstm":
        sequence_length = int(bundle.metadata.get("sequence_length", 1))
        return float(predict_with_lstm(bundle.model, list(inputs), sequence_length=sequence_length))

    if bundle.model_type == "xgboost":
        arr = np.asarray(inputs, dtype=np.float32).reshape(1, -1)
        prediction = bundle.model.predict(arr)
        return float(np.asarray(prediction).flatten()[0])

    if bundle.model_type == "tabpfn":
        arr = np.asarray(inputs, dtype=np.float32).reshape(1, -1)
        prediction = bundle.model.predict(arr)
        return float(np.asarray(prediction).flatten()[0])

    raise ValueError(f"不支持的模型类型: {bundle.model_type}")


def _train_xgboost(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    target_column: str,
    lag_columns: Sequence[str],
    params: Dict[str, Any],
) -> Tuple[TabularModelBundle, Dict[str, float]]:
    if XGBRegressor is None:
        raise RuntimeError("未安装 xgboost，请执行 pip install xgboost 再试。")

    working_df, feature_columns, lag_map = _prepare_training_frame(
        df, input_columns, target_column, lag_columns
    )

    X = working_df[feature_columns].to_numpy(dtype=np.float32)
    y = working_df[target_column].to_numpy(dtype=np.float32)

    _validate_samples(len(X))

    val_ratio = float(params.get("validation_ratio", 0.2))
    X_train, y_train, X_val, y_val = _chronological_split(X, y, val_ratio)

    model = XGBRegressor(
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 6)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=int(params.get("random_state", 42)),
        tree_method=params.get("tree_method", "hist"),
        n_jobs=int(params.get("n_jobs", 0)),
    )

    model.fit(X_train, y_train)

    metrics = _compute_regression_metrics(model, X_train, y_train, X_val, y_val)

    bundle = TabularModelBundle(
        model_type="xgboost",
        model=model,
        input_columns=list(feature_columns),
        target_column=target_column,
        lag_map=lag_map,
        feature_schema=_build_feature_schema(feature_columns, lag_map),
        metadata={
            "original_inputs": list(input_columns),
            "lag_source_columns": list(lag_columns),
        },
    )

    return bundle, metrics


def _train_tabpfn(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    target_column: str,
    lag_columns: Sequence[str],
    params: Dict[str, Any],
) -> Tuple[TabularModelBundle, Dict[str, float]]:
    if TabPFNRegressor is None:
        raise RuntimeError(
            "未安装 tabpfn，请执行 pip install tabpfn 再试。"
        )

    working_df, feature_columns, lag_map = _prepare_training_frame(
        df, input_columns, target_column, lag_columns
    )

    X = working_df[feature_columns].to_numpy(dtype=np.float32)
    y = working_df[target_column].to_numpy(dtype=np.float32)

    _validate_samples(len(X))

    val_ratio = float(params.get("validation_ratio", 0.2))
    X_train, y_train, X_val, y_val = _chronological_split(X, y, val_ratio)

    device = "cpu"  # 强制使用 CPU 以确保部署兼容性
    ensemble = int(params.get("ensemble", 32))

    model_kwargs: Dict[str, Any] = {"device": device}
    if TabPFNRegressor is not None:
        signature = inspect.signature(TabPFNRegressor.__init__)
        if "N_ensemble_configurations" in signature.parameters:
            model_kwargs["N_ensemble_configurations"] = ensemble
        elif "N_ensembles" in signature.parameters:
            model_kwargs["N_ensembles"] = ensemble

    model = TabPFNRegressor(**model_kwargs)
    model.fit(X_train, y_train)

    metrics = _compute_regression_metrics(model, X_train, y_train, X_val, y_val)

    bundle = TabularModelBundle(
        model_type="tabpfn",
        model=model,
        input_columns=list(feature_columns),
        target_column=target_column,
        lag_map=lag_map,
        feature_schema=_build_feature_schema(feature_columns, lag_map),
        metadata={
            "original_inputs": list(input_columns),
            "lag_source_columns": list(lag_columns),
            "device": device,
            "ensemble": ensemble,
        },
    )

    return bundle, metrics


def _train_lstm(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    target_column: str,
    lag_columns: Sequence[str],
    params: Dict[str, Any],
) -> Tuple[TabularModelBundle, Dict[str, float]]:
    working_df, feature_columns, lag_map = _prepare_training_frame(
        df, input_columns, target_column, lag_columns
    )

    epochs = int(params.get("epochs", 60))
    sequence_length = int(params.get("sequence_length", 1))
    learning_rate = float(params.get("learning_rate", 1e-3))

    lstm_bundle, metrics = train_lstm_model(
        working_df,
        input_columns=list(feature_columns),
        target_column=target_column,
        sequence_length=sequence_length,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    lstm_metrics = dict(metrics)
    lstm_metrics.update({
        "epochs": epochs,
        "sequence_length": sequence_length,
    })

    bundle = TabularModelBundle(
        model_type="lstm",
        model=lstm_bundle,
        input_columns=list(lstm_bundle.input_columns),
        target_column=target_column,
        lag_map=lag_map,
        feature_schema=_build_feature_schema(lstm_bundle.input_columns, lag_map),
        metadata={
            "original_inputs": list(input_columns),
            "lag_source_columns": list(lag_columns),
            "sequence_length": sequence_length,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
    )

    return bundle, lstm_metrics


def _prepare_training_frame(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    target_column: str,
    lag_columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    missing = [col for col in list(input_columns) + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"数据集中缺少以下字段：{', '.join(missing)}")

    working_df = df[list(dict.fromkeys(list(input_columns) + [target_column]))].copy()
    lag_map: Dict[str, str] = {}
    feature_columns: List[str] = []

    for column in input_columns:
        if column in lag_columns:
            lag_name = f"{column}__lag1"
            working_df[lag_name] = working_df[column].shift(1)
            feature_columns.append(lag_name)
            lag_map[lag_name] = column
        else:
            feature_columns.append(column)

    working_df = working_df.dropna(subset=feature_columns + [target_column])

    if working_df.empty:
        raise ValueError("滞后处理后数据为空，可能是样本过少或存在大量缺失值。")

    return working_df, feature_columns, lag_map


def _chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validation_ratio = max(0.0, min(validation_ratio, 0.8))
    n_samples = len(X)
    val_size = int(round(n_samples * validation_ratio))

    if n_samples <= 5 or val_size == 0 or val_size >= n_samples:
        return X, y, np.empty((0, X.shape[1])), np.empty((0,))

    split_idx = n_samples - val_size
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def _compute_regression_metrics(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    def _safe_r2(true: np.ndarray, pred: np.ndarray) -> float:
        if len(true) <= 1:
            return float("nan")
        return float(r2_score(true, pred))

    train_pred = np.asarray(model.predict(X_train)).reshape(-1)
    metrics["train_mse"] = float(mean_squared_error(y_train, train_pred))
    metrics["train_mae"] = float(mean_absolute_error(y_train, train_pred))
    metrics["train_r2"] = _safe_r2(y_train, train_pred)

    if len(y_val):
        val_pred = np.asarray(model.predict(X_val)).reshape(-1)
        metrics["val_mse"] = float(mean_squared_error(y_val, val_pred))
        metrics["val_mae"] = float(mean_absolute_error(y_val, val_pred))
        metrics["val_r2"] = _safe_r2(y_val, val_pred)

    metrics["samples"] = float(len(y_train) + len(y_val))
    clean_metrics: Dict[str, float] = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and not isinstance(value, (list, dict))
    }
    return clean_metrics


def _build_feature_schema(feature_columns: Sequence[str], lag_map: Dict[str, str]) -> List[Dict[str, Any]]:
    schema: List[Dict[str, Any]] = []
    for name in feature_columns:
        if name in lag_map:
            schema.append({
                "name": name,
                "label": f"{lag_map[name]} (lag 1)",
                "source": lag_map[name],
                "lag": 1,
            })
        else:
            schema.append({
                "name": name,
                "label": name,
                "source": name,
                "lag": 0,
            })
    return schema


def _validate_samples(sample_count: int) -> None:
    if sample_count < MIN_SAMPLE_THRESHOLD:
        raise ValueError(f"样本数量不足，至少需要 {MIN_SAMPLE_THRESHOLD} 条有效记录才能训练模型。")


