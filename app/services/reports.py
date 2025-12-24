from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Iterable, List, Sequence

from ..schemas.results import AnalysisResponse, ReportEntry


def build_analysis_report(
    analysis: AnalysisResponse,
    selected_columns: Sequence[str] | None = None,
    timestamp_column: str | None = None,
) -> Dict[str, Any]:
    """Compile a markdown-style report based on the latest analysis output."""

    metadata = analysis.metadata or {}
    effective_timestamp = (
        timestamp_column
        or analysis.agent_payload.timestamp_column
        or metadata.get("timestamp_column")
        or "自动识别"
    )
    row_count = metadata.get("row_count")
    column_count = metadata.get("column_count")
    missing_ratio = metadata.get("missing_ratio")

    effective_columns = list(selected_columns or [])
    if not effective_columns and metadata.get("analysis_columns"):
        effective_columns = list(metadata.get("analysis_columns"))

    generated_at = _dt.datetime.now().isoformat(timespec="seconds")

    lines: list[str] = []
    lines.append("# 指标分析报表")
    lines.append(f"生成时间：{_format_datetime(generated_at)}")
    lines.append("")

    lines.append("## 数据概览")
    overview_parts = []
    if row_count is not None:
        overview_parts.append(f"记录数：{row_count}")
    if column_count is not None:
        overview_parts.append(f"字段数：{column_count}")
    if missing_ratio is not None:
        overview_parts.append(f"缺失率：{missing_ratio * 100:.2f}%")
    lines.append("，".join(overview_parts) or "暂无数据概览信息。")
    lines.append(f"时间字段：{effective_timestamp or '未指定'}")
    if effective_columns:
        lines.append("参与分析的字段：" + "，".join(str(item) for item in effective_columns))
    lines.append("")

    lines.append("## 指标详情")
    for entry in analysis.report:
        if effective_columns and entry.parameter not in effective_columns:
            continue
        lines.extend(_render_report_entry(entry))

    if analysis.forecast.summary:
        lines.append("")
        lines.append("## 趋势摘要")
        for name, summary in analysis.forecast.summary.items():
            if effective_columns and name not in effective_columns:
                continue
            lines.append(f"- {name}：{summary}")

    return {
        "content": "\n".join(lines),
        "generated_at": generated_at,
        "columns": effective_columns,
        "timestamp_column": effective_timestamp,
    }


def _render_report_entry(entry: ReportEntry) -> Iterable[str]:
    lines: list[str] = []
    lines.append(f"### {entry.parameter}")
    lines.append(
        f"- 当前状态：{entry.current_status}（趋势：{entry.trend}，预测：{entry.forecast_trend}）"
    )

    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    lines.append(
        f"- 最新值：{_fmt(entry.latest_value)}；均值：{_fmt(entry.mean_value)}；标准差：{_fmt(entry.std_dev)}"
    )
    if entry.forecast_summary:
        lines.append(f"- 预测解读：{entry.forecast_summary}")

    if entry.recommendations:
        lines.append("- 操作建议：" + "；".join(entry.recommendations))

    if entry.anomalies:
        lines.append("- 重要波动：")
        for anomaly in entry.anomalies:
            label = "突增" if anomaly.get("type") == "peak" else "突降"
            idx = anomaly.get("index", "")
            value = anomaly.get("value")
            z_score = anomaly.get("z_score")
            detail_parts = [label]
            if idx:
                detail_parts.append(f"位置：{idx}")
            if value is not None:
                detail_parts.append(f"数值：{value:.4f}")
            if z_score is not None:
                detail_parts.append(f"Z：{z_score:.2f}")
            lines.append("  - " + "，".join(detail_parts))
    else:
        lines.append("- 重要波动：未检测到显著突变")

    if entry.influence_factors:
        lines.append("- 主要相关因子：" + "，".join(entry.influence_factors))

    lines.append("")
    return lines


def _format_datetime(value: str) -> str:
    try:
        dt = _dt.datetime.fromisoformat(value)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:  # pragma: no cover - defensive
        return value

