from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Any, Dict, Optional

import requests

try:
    from dashscope import Application  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Application = None  # type: ignore

logger = logging.getLogger(__name__)

# ----------------------------- 配置定义 ----------------------------------

BAILIAN_CONFIG = {
    "api_key": "Please use your own information",
    "app_id": "Please use your own information",
    "base_url": "https://dashscope.aliyuncs.com/api/v1/",
    "model": "qwen-max-0919",
}

TENCENT_CONFIG = {
    "base_url": "https://wss.lke.cloud.tencent.com/v1/qbot/chat/sse",
    "app_key": "Please use your own information",
    "timeout": 30,
    "model": "tencent-lke",
}

BAIDU_CONFIG = {
    "api_key": "Please use your own information",
    "app_id": "Please use your own information",
    "model": "baidu-appbuilder",
    "timeout": 30,
}

AI_CACHE_DURATION = 3600  # seconds
RATE_WINDOW = timedelta(hours=1)
RATE_LIMIT = 20

_ai_cache: Dict[str, Dict[str, Any]] = {}
_rate_usage: Dict[str, List[datetime]] = {}


# --------------------------- 缓存工具函数 ---------------------------------

def _cache_key(provider: str, message: str) -> str:
    return f"{provider}:{hash(message)}"


def get_ai_analysis_cache() -> Dict[str, Dict[str, Any]]:
    return _ai_cache


def clear_expired_cache() -> None:
    now = datetime.now()
    expired = [key for key, value in _ai_cache.items() if (now - value["timestamp"]).total_seconds() > AI_CACHE_DURATION]
    for key in expired:
        _ai_cache.pop(key, None)


def cache_ai_result(cache_key: str, result: Dict[str, Any]) -> None:
    _ai_cache[cache_key] = {
        "response": result.get("response"),
        "timestamp": datetime.now(),
        "model_used": result.get("model_used"),
        "tokens_used": result.get("tokens_used", 0),
    }


# --------------------------- 百度 AppBuilder ------------------------------

def call_baidu_agent(user_message: str, alarm_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start_time = datetime.now()

    try:
        import appbuilder  # type: ignore

        if not BAIDU_CONFIG["api_key"] or not BAIDU_CONFIG["app_id"]:
            raise RuntimeError("百度AI配置缺失，请设置 BAIDU_API_KEY 与 BAIDU_APP_ID")

        os.environ["APPBUILDER_TOKEN"] = BAIDU_CONFIG["api_key"]

        if alarm_context:
            content = (
                "作为铝拜耳法工艺专家，请分析以下报警：\n\n"
                f"参数：{alarm_context.get('parameter_name', 'N/A')}\n"
                f"工序：{alarm_context.get('process', '未知')}\n"
                f"当前值：{alarm_context.get('current_value', 'N/A')}{alarm_context.get('unit', '')}\n"
                f"正常范围：{alarm_context.get('lower_limit', 'N/A')} - {alarm_context.get('upper_limit', 'N/A')}\n"
                f"报警级别：{alarm_context.get('alarm_level', 'N/A')}\n\n"
                f"用户问题：{user_message}"\
            )
        else:
            content = user_message

        client = appbuilder.AppBuilderClient(BAIDU_CONFIG["app_id"])
        conversation_id = client.create_conversation()
        resp = client.run(conversation_id, content)

        response_time = (datetime.now() - start_time).total_seconds()
        ai_response = resp.content.answer

        if not ai_response or not ai_response.strip():
            return {
                "success": False,
                "error": "百度AI返回空回复，请重新尝试",
                "response_time": response_time,
                "model_used": BAIDU_CONFIG["model"],
            }

        return {
            "success": True,
            "response": ai_response,
            "response_time": response_time,
            "model_used": BAIDU_CONFIG["model"],
            "conversation_id": conversation_id,
        }

    except ImportError:
        return {
            "success": False,
            "error": "请先安装 AppBuilder SDK: pip install appbuilder-sdk",
            "response_time": (datetime.now() - start_time).total_seconds(),
            "model_used": BAIDU_CONFIG["model"],
        }
    except Exception as exc:  # pragma: no cover - network failure
        logger.exception("百度AI调用失败")
        return {
            "success": False,
            "error": f"百度AI服务异常: {exc}",
            "response_time": (datetime.now() - start_time).total_seconds(),
            "model_used": BAIDU_CONFIG["model"],
        }


# --------------------------- 百炼 dashscope -------------------------------

def call_bailian_agent(user_message: str, alarm_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start_time = datetime.now()
    if Application is None:
        return {
            "success": False,
            "error": "未安装 dashscope SDK，请先执行 pip install dashscope",
            "response_time": 0.0,
            "model_used": BAILIAN_CONFIG["model"],
        }

    if not BAILIAN_CONFIG["api_key"] or not BAILIAN_CONFIG["app_id"]:
        return {
            "success": False,
            "error": "百炼配置缺失，请设置 BAILIAN_API_KEY 与 BAILIAN_APP_ID",
            "response_time": 0.0,
            "model_used": BAILIAN_CONFIG["model"],
        }

    if alarm_context:
        context_message = (
            "作为铝拜耳法工艺专家，请简要分析以下报警：\n\n"
            f"参数：{alarm_context.get('parameter_name', 'N/A')}\n"
            f"工序：{alarm_context.get('process', '未知')}\n"
            f"当前值：{alarm_context.get('current_value', 'N/A')}{alarm_context.get('unit', '')}\n"
            f"正常范围：{alarm_context.get('lower_limit', 'N/A')} - {alarm_context.get('upper_limit', 'N/A')}\n"
            f"报警级别：{alarm_context.get('alarm_level', 'N/A')}\n\n"
            f"请简洁回答：{user_message}"
        )
    else:
        context_message = user_message

    try:
        response = Application.call(
            api_key=BAILIAN_CONFIG["api_key"],
            app_id=BAILIAN_CONFIG["app_id"],
            prompt=context_message,
            stream=False,
            has_thoughts=False,
            enable_system_time=True,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.exception("百炼智能体调用失败")
        return {
            "success": False,
            "error": f"百炼智能体调用失败: {exc}",
            "response_time": (datetime.now() - start_time).total_seconds(),
            "model_used": BAILIAN_CONFIG["model"],
        }

    response_time = (datetime.now() - start_time).total_seconds()

    if getattr(response, "status_code", HTTPStatus.BAD_REQUEST) != HTTPStatus.OK:
        return {
            "success": False,
            "error": f"百炼API调用失败: {getattr(response, 'message', '未知错误')}",
            "response_time": response_time,
            "model_used": BAILIAN_CONFIG["model"],
        }

    text = getattr(response.output, "text", "").strip()

    usage = getattr(response, "usage", None)
    tokens_used = 0
    input_tokens = 0
    output_tokens = 0
    if usage and getattr(usage, "models", None):
        model_usage = usage.models[0]
        input_tokens = getattr(model_usage, "input_tokens", 0)
        output_tokens = getattr(model_usage, "output_tokens", 0)
        tokens_used = input_tokens + output_tokens

    return {
        "success": True,
        "response": text,
        "response_time": response_time,
        "model_used": BAILIAN_CONFIG["model"],
        "tokens_used": tokens_used,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# --------------------------- 腾讯 LKE -------------------------------------

def call_tencent_agent(user_message: str, alarm_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    start_time = datetime.now()

    if not TENCENT_CONFIG["app_key"]:
        return {
            "success": False,
            "error": "腾讯云配置缺失，请设置 TENCENT_APP_KEY",
            "response_time": 0.0,
            "model_used": TENCENT_CONFIG["model"],
        }

    if alarm_context:
        content = (
            "作为铝拜耳法工艺专家，请分析以下报警：\n\n"
            f"参数：{alarm_context.get('parameter_name', 'N/A')}\n"
            f"工序：{alarm_context.get('process', '未知')}\n"
            f"当前值：{alarm_context.get('current_value', 'N/A')}{alarm_context.get('unit', '')}\n"
            f"正常范围：{alarm_context.get('lower_limit', 'N/A')} - {alarm_context.get('upper_limit', 'N/A')}\n"
            f"报警级别：{alarm_context.get('alarm_level', 'N/A')}\n\n"
            f"请简洁回答：{user_message}"
        )
    else:
        content = user_message

    session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    payload = {
        "content": content,
        "bot_app_key": TENCENT_CONFIG["app_key"],
        "session_id": session_id,
        "visitor_biz_id": f"user_{int(datetime.now().timestamp())}",
        "request_id": request_id,
        "incremental": True,
        "streaming_throttle": 5,
        "stream": "enable",
        "workflow_status": "enable",
        "model_name": "lke-deepseek-v3",
    }

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    try:
        response = requests.post(
            TENCENT_CONFIG["base_url"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            stream=True,
            timeout=TENCENT_CONFIG["timeout"],
        )
        response.raise_for_status()
    except Exception as exc:
        logger.exception("腾讯云LKE调用失败")
        return {
            "success": False,
            "error": f"腾讯云调用异常: {exc}",
            "response_time": (datetime.now() - start_time).total_seconds(),
            "model_used": TENCENT_CONFIG["model"],
        }

    ai_content: str = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload_line = line[5:].strip()
        if payload_line in ("[DONE]", "ping", ""):
            continue
        try:
            data = json.loads(payload_line)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "reply":
            chunk = data.get("payload", {}).get("content", "")
            ai_content += chunk
        elif data.get("type") == "error":
            error_detail = data.get("payload", {}).get("error", {}).get("message", "未知错误")
            return {
                "success": False,
                "error": f"腾讯云返回错误: {error_detail}",
                "response_time": (datetime.now() - start_time).total_seconds(),
                "model_used": TENCENT_CONFIG["model"],
            }

    response_time = (datetime.now() - start_time).total_seconds()

    if not ai_content.strip():
        return {
            "success": False,
            "error": "腾讯云返回空回复",
            "response_time": response_time,
            "model_used": TENCENT_CONFIG["model"],
        }

    return {
        "success": True,
        "response": ai_content.strip(),
        "response_time": response_time,
        "model_used": TENCENT_CONFIG["model"],
    }


# --------------------------- 聚合入口 -------------------------------------

PROVIDER_MAPPING = {
    "bailian": call_bailian_agent,
    "baidu": call_baidu_agent,
    "tencent": call_tencent_agent,
}


def call_ai_provider(provider: str, message: str, alarm_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    provider = provider.lower()
    if provider not in PROVIDER_MAPPING:
        raise ValueError(f"不支持的AI提供商: {provider}")

    clear_expired_cache()
    cache_key = _cache_key(provider, message)
    cached = _ai_cache.get(cache_key)
    if cached:
        return {
            "success": True,
            "response": cached["response"],
            "cached": True,
            "model_used": cached.get("model_used"),
            "response_time": 0.0,
            "remaining": max(RATE_LIMIT - len(_rate_usage.get(provider, [])), 0),
            "reset_at": next_reset(provider),
        }

    now = datetime.now()
    history = _rate_usage.get(provider, [])
    history = [ts for ts in history if now - ts <= RATE_WINDOW]
    _rate_usage[provider] = history

    if len(history) >= RATE_LIMIT:
        reset_time = next_reset(provider, history)
        wait_minutes = int((reset_time - now).total_seconds() // 60) + 1
        return {
            "success": False,
            "error": f"当前AI调用频率过高，请在{wait_minutes}分钟后再试（每小时最多{RATE_LIMIT}次）",
            "model_used": PROVIDER_MAPPING[provider].__name__,
            "response_time": 0.0,
            "remaining": 0,
            "reset_at": reset_time.isoformat(timespec="minutes"),
        }

    result = PROVIDER_MAPPING[provider](message, alarm_context=alarm_context)
    if result.get("success"):
        cache_ai_result(cache_key, result)
        history.append(now)
        _rate_usage[provider] = history
        result["remaining"] = max(RATE_LIMIT - len(history), 0)
        result["reset_at"] = next_reset(provider, history)
    return result


def next_reset(provider: str, history: Optional[List[datetime]] = None) -> str:
    recent = history if history is not None else _rate_usage.get(provider, [])
    if not recent:
        return (datetime.now() + RATE_WINDOW).isoformat(timespec="minutes")
    earliest = min(recent)
    reset_time = earliest + RATE_WINDOW
    return reset_time.isoformat(timespec="minutes")
