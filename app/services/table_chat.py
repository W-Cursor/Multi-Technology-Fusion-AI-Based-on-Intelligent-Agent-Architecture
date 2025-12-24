from __future__ import annotations

from typing import Any, Dict, List, Tuple

import asyncio
import datetime
import os
import re

import logging
import json

from urllib.parse import urlparse

import pandas as pd
import numpy as np
import requests

from .ai import call_ai_provider


logger = logging.getLogger(__name__)

class TableChatError(RuntimeError):
    """Raised when table chat cannot execute."""


async def describe_table_with_agent(
    df: pd.DataFrame,
    question: str,
    provider_name: str = "bailian",
    history: List[Dict[str, str]] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    if df.empty:
        raise TableChatError("上传的数据为空，无法进行分析。")

    df = df.copy()
    timeline = _extract_timeline(df)
    if timeline is not None:
        common_index = df.index.intersection(timeline.index)
        timeline = timeline.loc[common_index].copy()
        df = df.loc[common_index]

    numeric_df = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
    if numeric_df.empty:
        raise TableChatError("未检测到可分析的数值字段。")

    indicator_columns = _select_indicator_columns(numeric_df, question)
    if not indicator_columns:
        raise TableChatError("缺少可用于分析的关键指标列。")

    indicator_summaries: List[Dict[str, Any]] = []
    all_anomalies: List[str] = []

    for column in indicator_columns:
        series = numeric_df[column].dropna()
        if len(series) < 3:
            continue

        aligned_timeline = timeline.reindex(series.index) if timeline is not None else None
        summary = _summarize_indicator(series, column, numeric_df, aligned_timeline)
        indicator_summaries.append(summary)

        for anomaly in summary.get("anomalies", []):
            all_anomalies.append(
                f"{column} 在 {anomaly['label']} 出现异常值 {anomaly['value']:.3f} (z={anomaly['z_score']:.1f})"
            )

    if not indicator_summaries:
        raise TableChatError("关键指标有效数据不足，无法完成分析。")

    overview = _build_overview(df, timeline)
    recent_focus = _recent_focus_rows(df, timeline, indicator_columns)

    analysis_brief = _compile_analysis_brief(
        question=question,
        overview=overview,
        indicators=indicator_summaries,
        anomalies=all_anomalies,
        recent_focus=recent_focus,
    )

    history = history or []
    trimmed_history: List[Dict[str, str]] = []
    for entry in history[-6:]:  # keep last 3 turns
        role = entry.get("role")
        content = (entry.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            trimmed_history.append({"role": role, "content": content})

    history_text = ""
    if trimmed_history:
        lines = []
        for item in trimmed_history:
            prefix = "用户" if item["role"] == "user" else "智能体"
            lines.append(f"{prefix}：{item['content']}")
        history_text = "\n=== 历史对话 ===\n" + "\n".join(lines) + "\n"

    ai_prompt = (
        "你是拜耳法氧化铝生产安全风控专家。以下是系统依据上传表格生成的量化分析摘要，请结合用户问题给出结构化结论。\n"
        "=== 数据分析摘要 ===\n"
        f"{analysis_brief}\n"
        f"{history_text}"
        "=== 用户问题 ===\n"
        f"{question}\n\n"
        "【输出要求】\n"
        "1. 直接给出结论，每个要点不超过3句话\n"
        "2. 避免重复描述相同内容\n"
        "3. 优先列出TOP3关键风险和建议\n"
        "4. 总字数控制在600字以内\n\n"
        "请按以下结构输出（简洁版）：\n"
        "1. 关键指标趋势（仅列出最重要的2-3个）\n"
        "2. 主要影响因素（每个指标最多1句话）\n"
        "3. 异常风险与建议（TOP3，每项1-2句话）"
    )

    provider = (provider_name or "bailian").lower()

    if provider == "tablegpt":
        response_text, meta = await _run_tablegpt(ai_prompt)
    elif provider == "ollama":
        response_text, meta = await _run_ollama(ai_prompt)
    else:
        response_text, meta = await _run_bailian(ai_prompt)

    return response_text, meta


async def _run_bailian(prompt: str) -> Tuple[str, Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    def _invoke() -> Dict[str, Any]:
        return call_ai_provider("bailian", prompt)

    result = await loop.run_in_executor(None, _invoke)
    if not result.get("success"):
        raise TableChatError(result.get("error") or "百炼模型返回失败")

    answer = str(result.get("response", "")).strip()
    return answer, {
        "remaining": result.get("remaining"),
        "reset_at": result.get("reset_at"),
        "model_used": result.get("model_used"),
    }


async def _run_tablegpt(prompt: str) -> Tuple[str, Dict[str, Any]]:
    try:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:  # pragma: no cover - 向后兼容旧包
            from langchain_community.chat_models import ChatOpenAI
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
        # 尝试新版本的导入
        try:
            from tablegpt_agent import create_tablegpt_graph
            from tablegpt_agent.file_reading import Stage
        except ImportError:
            # 回退到旧版本的导入
            from tablegpt.agent import create_tablegpt_graph
            from tablegpt.agent.file_reading import Stage
        from pybox.local import LocalPyBoxManager
    except Exception as exc:  # pragma: no cover - optional dependency
        raise TableChatError(f"TableGPT 依赖未安装或导入失败: {exc}") from exc

    api_key = os.getenv("TABLEGPT_API_KEY")
    if not api_key:
        raise TableChatError("未配置 TABLEGPT_API_KEY 环境变量，无法调用 TableGPT 模型。")

    model_name = os.getenv("TABLEGPT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    base_url = os.getenv("TABLEGPT_BASE_URL")
    if not base_url:
        raise TableChatError("未配置 TABLEGPT_BASE_URL 环境变量，无法连接本地 TableGPT 服务。")

    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"

    max_tokens = int(os.getenv("TABLEGPT_MAX_TOKENS", "800"))

    llm_kwargs: Dict[str, Any] = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "top_p": 0.8,
    }
    llm_kwargs["api_key"] = api_key
    llm_kwargs["openai_api_key"] = api_key
    llm_kwargs["base_url"] = normalized
    llm_kwargs["openai_api_base"] = normalized

    # Ensure local/internal endpoints bypass corporate proxies.
    parsed_url = urlparse(normalized)
    hostname = parsed_url.hostname
    no_proxy_hosts = [host.strip() for host in os.getenv("NO_PROXY", "").split(",") if host.strip()]
    if hostname and hostname not in no_proxy_hosts:
        no_proxy_hosts.append(hostname)
        os.environ["NO_PROXY"] = ",".join(no_proxy_hosts)

    logger.warning(
        "TableGPT LLM config: model=%s, base=%s, max_tokens=%s",
        model_name,
        normalized,
        max_tokens,
    )

    llm = ChatOpenAI(**llm_kwargs)
    llm.model_kwargs["frequency_penalty"] = 0.3
    pybox_manager = LocalPyBoxManager()
    graph = create_tablegpt_graph(llm, pybox_manager)

    # Convert prompt string + history to TableGPT initial state
    entry_message = HumanMessage(content=prompt)
    state = {
        "messages": [entry_message],
        "entry_message": entry_message,
        "parent_id": None,
        "processing_stage": Stage.UPLOADED,
        "date": datetime.date.today(),
    }

    try:
        result = await graph.ainvoke(state)
    except Exception as exc:  # pragma: no cover - diagnostics for integration issues
        error_details: Dict[str, Any] = {}
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status is not None:
            error_details["status"] = status

        response = getattr(exc, "response", None)
        if response is not None:
            error_details["response_status"] = getattr(response, "status_code", None)
            body = getattr(response, "body", None) or getattr(response, "text", None)
            if callable(body):
                try:
                    body = body()
                except Exception:  # pragma: no cover - best effort to read body
                    body = None
            if body is not None:
                # Avoid logging extremely long payloads
                text = str(body)
                error_details["response_body_preview"] = text[:1000]

        message = getattr(exc, "message", None) or str(exc)
        error_details["message"] = message
        logger.error("TableGPT graph invocation failed: %s", json.dumps(error_details, ensure_ascii=False), exc_info=True)
        raise TableChatError("TableGPT 调用失败，请检查后台日志。") from exc

    if isinstance(result, dict):
        messages_payload = result.get("messages")
        if messages_payload and isinstance(messages_payload, list):
            last_msg = messages_payload[-1]
            if isinstance(last_msg, BaseMessage):
                content = getattr(last_msg, "content", "")
            else:
                content = last_msg.get("content", "")
            answer = str(content).strip()
        else:
            answer = str(result).strip()
    elif hasattr(result, "messages"):
        messages_payload = getattr(result, "messages", None)
        if messages_payload:
            last_msg = messages_payload[-1]
            answer = str(getattr(last_msg, "content", "") or "").strip()
        else:
            answer = str(result).strip()
    else:
        answer = str(result).strip()

    return answer, {
        "remaining": None,
        "reset_at": None,
        "model_used": model_name,
    }


async def _run_ollama(prompt: str) -> Tuple[str, Dict[str, Any]]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen")
    timeout_raw = os.getenv("OLLAMA_TIMEOUT", "60")
    try:
        timeout = float(timeout_raw)
    except (TypeError, ValueError):
        timeout = 60.0

    endpoint = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.3,
            "num_predict": 800,
        },
    }

    loop = asyncio.get_running_loop()

    def _invoke() -> Dict[str, Any]:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    try:
        result = await loop.run_in_executor(None, _invoke)
    except requests.RequestException as exc:
        raise TableChatError(f"Ollama 服务调用失败: {exc}") from exc

    answer = str(result.get("response", "")).strip()
    if not answer:
        raise TableChatError("Ollama 返回空结果，请检查模型是否正在运行。")

    return answer, {
        "remaining": None,
        "reset_at": None,
        "model_used": model_name,
    }


def _extract_timeline(df: pd.DataFrame) -> pd.Series | None:
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        series = df[datetime_cols[0]].dropna()
        return series.sort_values()

    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= max(3, len(df) * 0.1):
                parsed = parsed.dropna()
                return parsed.sort_values()
    return None


def _select_indicator_columns(
    numeric_df: pd.DataFrame,
    question: str,
    limit: int = 5,
) -> List[str]:
    matches = _match_columns_by_question(question, list(numeric_df.columns))
    if matches:
        return matches[:limit]

    missing_terms = _find_missing_terms(question, list(numeric_df.columns))
    if missing_terms:
        raise TableChatError(
            f"上传数据中未包含提问中提及的字段：{', '.join(missing_terms)}。请核实并重新上传或选择其他字段。"
        )

    stds_all = numeric_df.std().sort_values(ascending=False)
    return stds_all.head(limit).index.tolist()


def _normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^\w%\u4e00-\u9fff]+", "", value)
    return value


def _match_columns_by_question(question: str, columns: List[str]) -> List[str]:
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return []

    matches: List[str] = []

    for column in columns:
        normalized_column = _normalize_text(column)
        if not normalized_column:
            continue

        if normalized_column in normalized_question or normalized_question in normalized_column:
            matches.append(column)
            continue

        candidates = {
            normalized_question,
            normalized_question.replace("nt", ""),
            normalized_question.replace("nk", ""),
        }
        if any(token and token in normalized_column for token in candidates):
            matches.append(column)

    return list(dict.fromkeys(matches))


_QUESTION_STOP_TERMS = {
    "如何",
    "指标",
    "情况",
    "趋势",
    "变化",
    "状态",
    "分析",
    "检查",
    "工序",
    "质量",
    "数据",
}


def _find_missing_terms(question: str, columns: List[str]) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9%]+", question)
    candidates: List[str] = []
    normalized_columns = [_normalize_text(col) for col in columns]

    for token in tokens:
        token = token.strip()
        if len(token) <= 1 or token in _QUESTION_STOP_TERMS:
            continue
        normalized_token = _normalize_text(token)
        if not normalized_token:
            continue
        if any(
            normalized_token in normalized_column or normalized_column in normalized_token
            for normalized_column in normalized_columns
            if normalized_column
        ):
            continue
        if token not in candidates:
            candidates.append(token)

    return candidates


def _expand_correlated_columns(*args, **kwargs):  # pragma: no cover - legacy stub
    return []


def _summarize_indicator(
    series: pd.Series,
    column: str,
    numeric_df: pd.DataFrame,
    timeline: pd.Series | None,
) -> Dict[str, Any]:
    recent = series.tail(min(len(series), 30))
    baseline = recent.iloc[0]
    latest = recent.iloc[-1]
    mean_val = recent.mean()
    std_val = recent.std(ddof=0)

    if abs(baseline) > 1e-6:
        change_pct = (latest - baseline) / abs(baseline) * 100
    else:
        change_pct = np.nan

    slope = _compute_trend_slope(recent)
    trend_label = _trend_label(slope, change_pct)

    anomalies = _detect_anomalies(recent, timeline)
    influences = _top_correlations(column, numeric_df)

    recent_values = recent.tail(5)
    if timeline is not None:
        recent_labels = [_format_timestamp(timeline.loc[idx]) if idx in timeline.index else str(idx) for idx in recent_values.index]
    else:
        recent_labels = [str(idx) for idx in recent_values.index]

    return {
        "column": column,
        "latest": float(latest),
        "mean": float(mean_val),
        "std": float(std_val) if not np.isnan(std_val) else None,
        "trend": trend_label,
        "change_pct": None if np.isnan(change_pct) else float(change_pct),
        "recent_points": [
            {
                "label": recent_labels[i],
                "value": float(value),
            }
            for i, value in enumerate(recent_values)
        ],
        "influences": influences,
        "anomalies": anomalies,
    }


def _compute_trend_slope(series: pd.Series) -> float:
    if len(series) < 3:
        return 0.0
    x = np.arange(len(series))
    y = series.to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def _trend_label(slope: float, change_pct: float | None) -> str:
    if change_pct is not None and abs(change_pct) >= 0.5:
        return "上升" if change_pct > 0 else "下降"
    if abs(slope) < 1e-6:
        return "平稳"
    return "上升" if slope > 0 else "下降"


def _detect_anomalies(series: pd.Series, timeline: pd.Series | None) -> List[Dict[str, Any]]:
    if len(series) < 5:
        return []
    std_val = series.std()
    if std_val == 0 or np.isnan(std_val):
        return []

    mean_val = series.mean()
    z_scores = (series - mean_val) / std_val
    anomalies: List[Dict[str, Any]] = []

    for idx, z in z_scores.items():
        if abs(z) >= 2.0:
            if timeline is not None and idx in timeline.index:
                label = _format_timestamp(timeline.loc[idx])
            else:
                label = str(idx)
            anomalies.append({
                "label": label,
                "value": float(series.loc[idx]),
                "z_score": float(z),
            })
    return anomalies[:3]


def _top_correlations(column: str, numeric_df: pd.DataFrame, limit: int = 3) -> List[Dict[str, Any]]:
    target = numeric_df[column]
    correlations = numeric_df.corrwith(target).dropna()
    correlations = correlations.drop(column, errors="ignore")
    if correlations.empty:
        return []

    top = correlations.abs().sort_values(ascending=False).head(limit)
    influences: List[Dict[str, Any]] = []
    for other in top.index:
        corr_value = correlations[other]
        influences.append({
            "column": other,
            "corr": float(corr_value),
            "direction": "正" if corr_value >= 0 else "负",
            "strength": _corr_strength(abs(corr_value)),
        })
    return influences


def _corr_strength(value: float) -> str:
    if value >= 0.7:
        return "强"
    if value >= 0.4:
        return "中"
    return "弱"


def _build_overview(df: pd.DataFrame, timeline: pd.Series | None) -> Dict[str, Any]:
    row_count = len(df)
    col_count = len(df.columns)
    if timeline is not None and not timeline.empty:
        start = _format_timestamp(timeline.iloc[0])
        end = _format_timestamp(timeline.iloc[-1])
    else:
        start = end = "时间列缺失"

    return {
        "rows": row_count,
        "columns": col_count,
        "start": start,
        "end": end,
    }


def _recent_focus_rows(
    df: pd.DataFrame,
    timeline: pd.Series | None,
    columns: List[str],
    window: int = 3,
) -> List[str]:
    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        return []

    subset = df[available_cols].tail(window)
    rows: List[str] = []

    for idx, row in subset.iterrows():
        label = _format_timestamp(timeline.loc[idx]) if timeline is not None and idx in timeline.index else str(idx)
        values = ", ".join(f"{col}={row[col]:.3f}" for col in available_cols if pd.notna(row[col]))
        rows.append(f"{label}: {values}")
    return rows


def _compile_analysis_brief(
    question: str,
    overview: Dict[str, Any],
    indicators: List[Dict[str, Any]],
    anomalies: List[str],
    recent_focus: List[str],
) -> str:
    lines: List[str] = []
    lines.append(
        f"数据概况：样本 {overview['rows']} 条，字段 {overview['columns']} 个，时间范围 {overview['start']} 至 {overview['end']}。"
    )
    lines.append(f"用户提问焦点：{question}")

    if recent_focus:
        lines.append("最近记录：")
        lines.extend(f"  - {entry}" for entry in recent_focus)

    lines.append("关键指标：")
    for idx, info in enumerate(indicators, 1):
        influence_text = ", ".join(
            f"{item['column']}({item['direction']}相关，{item['strength']}；r={item['corr']:.2f})"
            for item in info.get("influences", [])
        )
        std_display = f"{info['std']:.3f}" if info.get("std") is not None else "n/a"
        change_desc = (
            f"，相对变化 {info['change_pct']:.1f}%" if info.get("change_pct") is not None else ""
        )
        lines.append(
            f"  {idx}. {info['column']}：最新值 {info['latest']:.3f}，均值 {info['mean']:.3f}，波动(σ) {std_display}，趋势 {info['trend']}{change_desc}."
        )
        if influence_text:
            lines.append(f"     主要影响因素：{influence_text}。")
    if anomalies:
        lines.append("异常提示：")
        for entry in anomalies[:5]:
            lines.append(f"  - {entry}")

    return "\n".join(lines)


def _format_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "未知时间"
    if isinstance(value, pd.Timestamp):
        if value.time() == pd.Timestamp.min.time():
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M")
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M")
        except Exception:  # pragma: no cover - defensive
            return str(value)
    return str(value)
