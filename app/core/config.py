from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv


load_dotenv()


class Settings:
    base_dir: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
    templates_dir: pathlib.Path = base_dir / "templates"
    static_dir: pathlib.Path = base_dir / "static"

    # Analysis defaults
    target_column_candidates = [
        "Al2O3_concentration",
        "粒度",
        "浓度",
        "产量",
        "Alumina",
    ]
    timestamp_column_candidates = [
        "timestamp",
        "datetime",
        "date",
        "时间",
        "采集时间",
    ]
    forecast_horizon = 24  # hours by default
    min_history_points = 30

    # PandasAI integration
    pandasai_api_key: str = os.getenv("PANDASAI_API_KEY", "")
    pandasai_model: str = os.getenv("PANDASAI_MODEL", "gpt-4o-mini")

    # Ollama integration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen")
    try:
        ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "60"))
    except (TypeError, ValueError):
        ollama_timeout = 60.0

    tablegpt_base_url: str | None = os.getenv("TABLEGPT_BASE_URL")
    tablegpt_api_key: str = os.getenv("TABLEGPT_API_KEY", "")


settings = Settings()



