from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("缺少 PyTorch 依赖，请执行 pip install torch") from exc

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class LSTMModelBundle:
    model: nn.Module
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    input_columns: List[str]
    target_column: str


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc(output)
        return output


def _prepare_dataset(
    data: np.ndarray,
    targets: np.ndarray,
    sequence_length: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if sequence_length <= 1:
        return data[:, None, :], targets[:, None]

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in range(len(data) - sequence_length + 1):
        xs.append(data[i : i + sequence_length])
        ys.append(targets[i + sequence_length - 1])
    return np.stack(xs), np.array(ys)[:, None]


def train_lstm_model(
    df,
    input_columns: List[str],
    target_column: str,
    sequence_length: int = 1,
    epochs: int = 60,
    learning_rate: float = 1e-3,
) -> Tuple[LSTMModelBundle, Dict[str, float]]:
    working_df = df[input_columns + [target_column]].dropna().copy()
    if working_df.shape[0] < 20:
        raise ValueError("样本数量不足，至少需要 20 条有效记录才能训练模型。")

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    scaled_inputs = input_scaler.fit_transform(working_df[input_columns].values)
    scaled_targets = target_scaler.fit_transform(working_df[[target_column]].values).reshape(-1)

    X, y = _prepare_dataset(scaled_inputs, scaled_targets, sequence_length=sequence_length)

    split_idx = int(len(X) * 0.8)
    if split_idx <= sequence_length:
        split_idx = len(X) - 2

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()), batch_size=32, shuffle=True)
    val_tensor_x = torch.from_numpy(X_val).float()
    val_tensor_y = torch.from_numpy(y_val).float()

    device = torch.device("cpu")
    model = LSTMRegressor(input_size=len(input_columns)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.from_numpy(X_train).float()).numpy()
        val_pred = model(val_tensor_x).numpy()

    train_true = y_train
    val_true = y_val

    def _safe_r2(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
        if len(true_vals) <= 1:
            return float("nan")
        return float(r2_score(true_vals, pred_vals))

    def _metrics(pred, true):
        mse = float(np.mean((pred - true) ** 2))
        mae = float(np.mean(np.abs(pred - true)))
        return mse, mae

    train_mse, train_mae = _metrics(train_pred, train_true)
    val_mse, val_mae = _metrics(val_pred, val_true)

    bundle = LSTMModelBundle(
        model=model,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        input_columns=input_columns,
        target_column=target_column,
    )

    metrics = {
        "train_mse": train_mse,
        "train_mae": train_mae,
        "train_r2": _safe_r2(train_true, train_pred),
        "val_mse": val_mse,
        "val_mae": val_mae,
        "val_r2": _safe_r2(val_true, val_pred),
        "samples": working_df.shape[0],
    }

    return bundle, metrics


def predict_with_lstm(bundle: LSTMModelBundle, inputs: List[float], sequence_length: int = 1) -> float:
    if len(inputs) != len(bundle.input_columns):
        raise ValueError("输入参数数量与训练时不一致。")

    scaled = bundle.input_scaler.transform([inputs])
    if sequence_length <= 1:
        tensor_x = torch.from_numpy(scaled[:, None, :]).float()
    else:
        raise ValueError("当前预测接口仅支持单步输入。")

    with torch.no_grad():
        pred = bundle.model(tensor_x).numpy()

    restored = bundle.target_scaler.inverse_transform(pred)
    return float(restored[0, 0])

