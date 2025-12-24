from __future__ import annotations

import io
import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .core.config import settings
from .schemas.results import (
    AgentPayload,
    AnalysisResponse,
    ColumnPreviewResponse,
    EquilibriumRequest,
    EquilibriumResult,
    ExpertSchemeCreate,
    ExpertSchemeInfo,
    ExpertSchemePayload,
    ExpertSchemeUpdate,
    ExpertSchemeUpdatePayload,
    KnowledgeGraphBuildRequest,
    KnowledgeGraphBuildResponse,
    KnowledgeGraphRootCauseRequest,
    KnowledgeGraphRootCauseResponse,
    KnowledgeGraphPathRequest,
    KnowledgeGraphPathResponse,
    ParameterAnalysisRequest,
    ParameterAnalysisResponse,
)
from .services.ai import call_ai_provider
from .services.equilibrium import calculate_equilibrium
from .services.pipeline import run_full_analysis
from .services.reports import build_analysis_report
from .services.preprocessing import prepare_dataset
from .services.tabular_models import (
    TabularModelBundle,
    predict_model as predict_tabular_model,
    train_model as train_tabular_model,
)
from .services.table_chat import TableChatError, describe_table_with_agent
from .services.parameter_analysis import (
    ExpertKnowledgeStore,
    generate_parameter_analysis,
)
from .services.kg_root_cause import generate_knowledge_graph_root_cause
from .services.export_excel import export_analysis_to_excel
from .services.kg_graph_builder import build_knowledge_graph, find_graph_paths


logger = logging.getLogger(__name__)

MODEL_TYPES = {"lstm", "xgboost", "tabpfn"}
_MODEL_STORE: Dict[str, Dict[str, TabularModelBundle]] = {
    model_type: {} for model_type in MODEL_TYPES
}


def create_app() -> FastAPI:
    app = FastAPI(title="Bayer Process Analytics", version="0.1.0")

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    templates = Jinja2Templates(directory=settings.templates_dir)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request, "settings": settings})

    @app.get("/parameter-analysis", response_class=HTMLResponse)
    async def parameter_analysis(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("parameter_analysis.html", {"request": request, "settings": settings})

    @app.get("/model-lab", response_class=HTMLResponse)
    async def model_lab(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("model_lab.html", {"request": request, "settings": settings})

    @app.get("/kg-root-cause-lab", response_class=HTMLResponse)
    async def knowledge_graph_root_cause_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "kg_root_cause_lab.html",
            {"request": request, "settings": settings},
        )

    @app.get("/kg-graph-lab", response_class=HTMLResponse)
    async def knowledge_graph_builder_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "kg_graph_lab.html",
            {"request": request, "settings": settings},
        )

    @app.post("/kg-root-cause/analyze", response_model=KnowledgeGraphRootCauseResponse)
    async def knowledge_graph_root_cause(payload: KnowledgeGraphRootCauseRequest) -> KnowledgeGraphRootCauseResponse:
        df = _restore_dataframe(payload.cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请重新上传并预览")

        try:
            return generate_knowledge_graph_root_cause(df, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Knowledge graph root cause analysis failed")
            raise HTTPException(status_code=500, detail=f"图谱归因分析失败: {exc}") from exc

    @app.post("/kg-graph/generate", response_model=KnowledgeGraphBuildResponse)
    async def knowledge_graph_generate(payload: KnowledgeGraphBuildRequest) -> KnowledgeGraphBuildResponse:
        df = _restore_dataframe(payload.cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请重新上传并预览")

        try:
            return build_knowledge_graph(df, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Knowledge graph generation failed")
            raise HTTPException(status_code=500, detail=f"知识图谱生成失败: {exc}") from exc

    @app.post("/kg-graph/path", response_model=KnowledgeGraphPathResponse)
    async def knowledge_graph_path(payload: KnowledgeGraphPathRequest) -> KnowledgeGraphPathResponse:
        df = _restore_dataframe(payload.cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请重新上传并预览")

        try:
            return find_graph_paths(df, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Knowledge graph path search failed")
            raise HTTPException(status_code=500, detail=f"图谱链路搜索失败: {exc}") from exc

    @app.get("/analysis/parameters")
    def analysis_parameters(cache_key: str) -> JSONResponse:
        df = _restore_dataframe(cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请重新上传并预览")

        numeric_columns = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
        return JSONResponse({"parameters": numeric_columns})

    @app.post("/parameter-analysis", response_model=ParameterAnalysisResponse)
    async def perform_parameter_analysis(payload: ParameterAnalysisRequest) -> ParameterAnalysisResponse:
        df = _restore_dataframe(payload.cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请从指标分析页面重新进入。")

        try:
            result = generate_parameter_analysis(df, payload, _get_expert_store())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Parameter analysis failed")
            raise HTTPException(status_code=500, detail=f"参数分析失败: {exc}") from exc

        return result

    @app.post("/ai-chat")
    async def ai_chat(request: Request) -> JSONResponse:
        payload = await request.json()
        provider = payload.get("provider", "bailian")
        message = (payload.get("message") or "").strip()
        alarm_context = payload.get("alarm_context")

        if not message:
            raise HTTPException(status_code=400, detail="提问内容不能为空")

        result = call_ai_provider(provider, message, alarm_context=alarm_context)
        return JSONResponse(result)

    @app.post("/preview", response_model=ColumnPreviewResponse)
    async def preview_columns(file: UploadFile = File(...)) -> ColumnPreviewResponse:
        df = await _read_upload_to_dataframe(file)
        cleaned_df, metadata = prepare_dataset(df)
        cache_key = _store_dataframe(df)

        response = ColumnPreviewResponse(
            columns=list(df.columns),
            numeric_columns=list(cleaned_df.columns),
            timestamp_candidates=metadata.timestamp_candidates,
            suggested_key_parameters=metadata.key_parameters,
            grouped_columns=metadata.grouped_columns,
            metadata=metadata.model_dump(),
            cache_key=cache_key,
        )
        return response

    @app.post("/kg/dataset-metadata", response_model=ColumnPreviewResponse)
    async def get_dataset_metadata(cache_key: str) -> ColumnPreviewResponse:
        df = _restore_dataframe(cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请重新上传并预览")

        cleaned_df, metadata = prepare_dataset(df)

        return ColumnPreviewResponse(
            columns=list(df.columns),
            numeric_columns=list(cleaned_df.columns),
            timestamp_candidates=metadata.timestamp_candidates,
            suggested_key_parameters=metadata.key_parameters,
            grouped_columns=metadata.grouped_columns,
            metadata=metadata.model_dump(),
            cache_key=cache_key,
        )

    @app.post("/model-lab/upload", response_model=ColumnPreviewResponse)
    async def upload_model_lab_dataset(file: UploadFile = File(...)) -> ColumnPreviewResponse:
        df = await _read_upload_to_dataframe(file)
        cleaned_df, metadata = prepare_dataset(df)
        cache_key = _store_dataframe(df)

        return ColumnPreviewResponse(
            columns=list(df.columns),
            numeric_columns=list(cleaned_df.columns),
            timestamp_candidates=metadata.timestamp_candidates,
            suggested_key_parameters=metadata.key_parameters,
            grouped_columns=metadata.grouped_columns,
            metadata=metadata.model_dump(),
            cache_key=cache_key,
        )

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze(
        file: UploadFile = File(...),
        selected_columns: Optional[str] = Form(default=None),
        selected_timestamp: Optional[str] = Form(default=None),
    ) -> AnalysisResponse:
        df = await _read_upload_to_dataframe(file)

        columns: List[str] | None = None
        if selected_columns:
            columns = [col.strip() for col in selected_columns.split(",") if col.strip()]

        try:
            result = run_full_analysis(
                df,
                include_columns=columns,
                timestamp_column=selected_timestamp,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Analysis pipeline failed")
            raise HTTPException(status_code=500, detail=f"分析过程中出现错误: {exc}") from exc

        return result

    @app.post("/report/generate")
    async def generate_report(request: Request) -> JSONResponse:
        payload = await request.json()
        cache_key = (payload.get("cache_key") or "").strip()
        columns = payload.get("columns") or []
        timestamp_column = (payload.get("timestamp_column") or "").strip() or None

        if not cache_key:
            raise HTTPException(status_code=400, detail="缺少缓存键，无法生成报表")
        if not columns:
            raise HTTPException(status_code=400, detail="请选择至少一个指标字段")

        df = _restore_dataframe(cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请重新上传并预览")

        try:
            analysis = run_full_analysis(
                df,
                include_columns=columns,
                timestamp_column=timestamp_column,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        report = build_analysis_report(analysis, selected_columns=columns, timestamp_column=timestamp_column)
        return JSONResponse({"success": True, "report": report})

    @app.post("/report/export/excel")
    async def export_report_excel(request: Request) -> Response:
        payload = await request.json()
        cache_key = (payload.get("cache_key") or "").strip()
        columns = payload.get("columns") or []
        timestamp_column = (payload.get("timestamp_column") or "").strip() or None

        if not cache_key:
            raise HTTPException(status_code=400, detail="缺少缓存键，无法导出报表")
        if not columns:
            raise HTTPException(status_code=400, detail="请选择至少一个指标字段")

        df = _restore_dataframe(cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="未找到缓存数据，请重新上传并预览")

        try:
            analysis = run_full_analysis(
                df,
                include_columns=columns,
                timestamp_column=timestamp_column,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            excel_bytes = export_analysis_to_excel(analysis, selected_columns=columns, timestamp_column=timestamp_column)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Excel export failed")
            raise HTTPException(status_code=500, detail=f"Excel 导出失败: {exc}") from exc

        timestamp = _get_timestamp_str()
        # Use URL encoding for Chinese filename to avoid latin-1 encoding issues
        filename_ascii = f"Bayer_Analysis_Report_{timestamp}.xlsx"
        filename_utf8 = f"拜耳法分析报告_{timestamp}.xlsx"
        
        # RFC 5987 encoding for UTF-8 filenames
        from urllib.parse import quote
        encoded_filename = quote(filename_utf8.encode('utf-8'))
        
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename_ascii}\"; filename*=UTF-8''{encoded_filename}"
            },
        )

    @app.post("/agent-payload", response_model=AgentPayload)
    async def agent_payload(
        file: UploadFile = File(...),
        selected_columns: Optional[str] = Form(default=None),
        selected_timestamp: Optional[str] = Form(default=None),
    ) -> AgentPayload:
        df = await _read_upload_to_dataframe(file)

        columns: List[str] | None = None
        if selected_columns:
            columns = [col.strip() for col in selected_columns.split(",") if col.strip()]

        analysis = run_full_analysis(
            df,
            include_columns=columns,
            timestamp_column=selected_timestamp,
        )
        return analysis.agent_payload

    @app.post("/calculate-equilibrium", response_model=EquilibriumResult)
    async def calculate_equilibrium_api(request: EquilibriumRequest) -> EquilibriumResult:
        result = calculate_equilibrium(
            mineral_data=request.mineral_data,
            process_params=request.process_params,
        )
        return EquilibriumResult.model_validate(result)

    @app.get("/status/ollama")
    def status_ollama() -> JSONResponse:
        try:
            resp = requests.get(settings.ollama_base_url.rstrip('/') + "/api/version", timeout=2)
            resp.raise_for_status()
        except Exception as exc:  # pylint: disable=broad-except
            return JSONResponse({"success": False, "error": str(exc)}, status_code=503)
        return JSONResponse({"success": True})

    @app.get("/status/tablegpt")
    def status_tablegpt(base_url: str | None = None) -> JSONResponse:
        url_raw = base_url or settings.tablegpt_base_url or ""
        url = url_raw.rstrip("/")
        if not url:
            return JSONResponse({"success": False, "error": "未配置 TableGPT Base URL"}, status_code=400)

        candidates = ["/v1/models", "/models", "/health"]
        probe_errors: list[str] = []
        headers: dict[str, str] | None = None

        api_key = settings.tablegpt_api_key if hasattr(settings, "tablegpt_api_key") else None
        if api_key:
            headers = {"Authorization": f"Bearer {api_key}"}
            ##test

        for suffix in candidates:
            try:
                resp = requests.get(url + suffix, timeout=2, headers=headers)
                if resp.status_code == 200:
                    return JSONResponse({"success": True})
                probe_errors.append(f"{suffix}: HTTP {resp.status_code}")
            except Exception as exc:  # pylint: disable=broad-except
                probe_errors.append(f"{suffix}: {exc}")

        return JSONResponse(
            {"success": False, "error": "; ".join(probe_errors) or "TableGPT 服务不可用"},
            status_code=503,
        )

    @app.get("/config/tablegpt")
    def get_tablegpt_config() -> JSONResponse:
        return JSONResponse({"base_url": settings.tablegpt_base_url or ""})

    @app.post("/model/train")
    async def model_train(request: Request) -> JSONResponse:
        payload = await request.json()

        model_type = (payload.get("model_type") or "lstm").lower()
        if model_type not in MODEL_TYPES:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}")

        cache_key = (payload.get("cache_key") or "").strip()
        input_cols = payload.get("input_columns") or []
        target_col = (payload.get("target_column") or "").strip()
        lag_columns = payload.get("lag_columns") or []

        if not cache_key or not input_cols or not target_col:
            raise HTTPException(status_code=400, detail="请提供缓存键、输入字段和输出字段")

        df = _restore_dataframe(cache_key)
        if df is None:
            raise HTTPException(status_code=400, detail="请先上传并预览数据")

        # 收集超参数（兼容旧字段与 hyperparams 字段）
        hyperparams: Dict[str, Any] = dict(payload.get("hyperparams") or {})
        passthrough_keys = [
            "epochs",
            "sequence_length",
            "learning_rate",
            "validation_ratio",
            "n_estimators",
            "max_depth",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "random_state",
            "tree_method",
            "n_jobs",
            "device",
            "ensemble",
        ]
        for key in passthrough_keys:
            if key in payload and key not in hyperparams:
                hyperparams[key] = payload[key]

        try:
            bundle, metrics = train_tabular_model(
                model_type=model_type,
                df=df,
                input_columns=input_cols,
                target_column=target_col,
                lag_columns=lag_columns,
                params=hyperparams,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        model_id = str(uuid.uuid4())
        _MODEL_STORE[model_type][model_id] = bundle

        response: Dict[str, Any] = {
            "success": True,
            "model_id": model_id,
            "model_type": model_type,
            "metrics": metrics,
            "input_columns": bundle.input_columns,
            "target_column": bundle.target_column,
            "feature_schema": bundle.feature_schema,
            "lag_map": bundle.lag_map,
        }
        return JSONResponse(response)

    @app.post("/model/predict")
    async def model_predict(request: Request) -> JSONResponse:
        payload = await request.json()

        model_type = (payload.get("model_type") or "lstm").lower()
        if model_type not in MODEL_TYPES:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}")

        model_id = (payload.get("model_id") or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="请提供 model_id")

        bundle = _MODEL_STORE.get(model_type, {}).get(model_id)
        if bundle is None:
            raise HTTPException(status_code=400, detail="模型不存在或已过期，请重新训练。")

        raw_inputs = payload.get("inputs")
        if raw_inputs is None:
            raise HTTPException(status_code=400, detail="请提供预测所需的输入参数")

        try:
            inputs = _resolve_prediction_inputs(bundle, raw_inputs)
            prediction = predict_tabular_model(bundle, inputs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response: Dict[str, Any] = {
            "success": True,
            "prediction": prediction,
            "target": bundle.target_column,
            "model_type": bundle.model_type,
        }
        return JSONResponse(response)

    @app.post("/table-chat")
    async def table_chat(request: Request) -> JSONResponse:
        payload = await request.json()
        question = (payload.get("question") or "").strip()
        cached_key = payload.get("cache_key") or ""
        history_raw = payload.get("history") or []
        if not isinstance(history_raw, list):
            history_raw = []

        if not question:
            raise HTTPException(status_code=400, detail="请提供提问内容")

        df = _restore_dataframe(cached_key)
        if df is None:
            raise HTTPException(status_code=400, detail="请先上传并预览数据")

        provider = payload.get("provider") or "bailian"

        try:
            answer, meta = await describe_table_with_agent(df, question, provider, history_raw)
        except TableChatError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Table chat failed")
            raise HTTPException(status_code=500, detail=f"表格对话失败: {exc}") from exc

        payload_response: Dict[str, Any] = {"success": True, "answer": answer}
        payload_response.update(meta)
        return JSONResponse(payload_response)

    @app.get("/expert-schemes")
    def list_expert_schemes(target: str | None = None) -> JSONResponse:
        store = _get_expert_store()
        schemes = [scheme.model_dump() for scheme in store.list(target)]
        return JSONResponse({"schemes": schemes})

    @app.post("/expert-schemes", response_model=ExpertSchemeInfo)
    def create_expert_scheme(payload: ExpertSchemePayload) -> ExpertSchemeInfo:
        if not payload.target_parameter:
            raise HTTPException(status_code=400, detail="请指定目标参数")
        store = _get_expert_store()
        create_payload = ExpertSchemeCreate.model_validate(payload.model_dump())
        return store.create(create_payload)

    @app.put("/expert-schemes/{scheme_id}", response_model=ExpertSchemeInfo)
    def update_expert_scheme(scheme_id: str, payload: ExpertSchemeUpdatePayload) -> ExpertSchemeInfo:
        store = _get_expert_store()
        try:
            update_payload = ExpertSchemeUpdate.model_validate(payload.model_dump())
            return store.update(scheme_id, update_payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="知识方案不存在") from exc

    @app.delete("/expert-schemes/{scheme_id}")
    def delete_expert_scheme(scheme_id: str) -> JSONResponse:
        store = _get_expert_store()
        store.delete(scheme_id)
        return JSONResponse({"success": True})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        payload: Dict[str, Any] = {"detail": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=payload)

    return app


async def _read_upload_to_dataframe(file: UploadFile) -> pd.DataFrame:
    try:
        raw_bytes = await file.read()
        buffer = io.BytesIO(raw_bytes)
        df = _load_dataframe_with_fallback(buffer, file.filename)
    except ValueError as exc:
        logger.exception("Failed to parse uploaded file: %s", exc)
        raise

    return df


def _load_dataframe_with_fallback(buffer: io.BytesIO, filename: str | None) -> pd.DataFrame:
    if not filename:
        raise ValueError("上传文件缺少文件名")

    dataframe: pd.DataFrame | None = None

    if filename.lower().endswith(".csv"):
        dataframe = pd.read_csv(buffer)
    elif filename.lower().endswith((".xlsx", ".xls")):
        dataframe = pd.read_excel(buffer)
    else:
        try:
            dataframe = pd.read_csv(buffer)
        except Exception as csv_exc:  # pylint: disable=broad-except
            buffer.seek(0)
            try:
                dataframe = pd.read_excel(buffer)
            except Exception as excel_exc:  # pylint: disable=broad-except
                raise ValueError("暂不支持的文件类型，请上传 CSV 或 Excel 文件") from excel_exc
            raise ValueError("无法解析上传的 CSV 文件") from csv_exc

    if dataframe is None or dataframe.empty:
        raise ValueError("上传的表格数据为空或无法识别")

    return dataframe


app = create_app()

if not hasattr(app.state, "table_cache"):
    app.state.table_cache = {}

if not hasattr(app.state, "expert_store"):
    app.state.expert_store = ExpertKnowledgeStore()


def _get_expert_store() -> ExpertKnowledgeStore:
    store = getattr(app.state, "expert_store", None)
    if store is None:
        store = ExpertKnowledgeStore()
        app.state.expert_store = store
    return store


def _store_dataframe(df: pd.DataFrame) -> str:
    key = uuid.uuid4().hex
    buffer = io.BytesIO()
    df.to_pickle(buffer)
    buffer.seek(0)
    app.state.table_cache[key] = buffer.getvalue()
    return key


def _restore_dataframe(cache_key: str) -> pd.DataFrame | None:
    payload = app.state.table_cache.get(cache_key)
    if payload is None:
        return None
    buffer = io.BytesIO(payload)
    buffer.seek(0)
    try:
        return pd.read_pickle(buffer)
    except Exception:  # pylint: disable=broad-except
        return None


def _get_timestamp_str() -> str:
    """Generate timestamp string for filenames."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_prediction_inputs(bundle: TabularModelBundle, raw_inputs: Any) -> List[float]:
    feature_names = list(bundle.input_columns)
    values: List[float] = []

    if isinstance(raw_inputs, list):
        if len(raw_inputs) != len(feature_names):
            raise ValueError(f"输入参数数量应为 {len(feature_names)}，实际为 {len(raw_inputs)}")
        try:
            values = [float(v) for v in raw_inputs]
        except (TypeError, ValueError) as exc:
            raise ValueError("输入参数无法转换为数值") from exc
        return values

    if isinstance(raw_inputs, dict):
        for name in feature_names:
            if name in raw_inputs:
                source_value = raw_inputs[name]
            else:
                original_name = bundle.lag_map.get(name, name)
                if original_name not in raw_inputs:
                    raise ValueError(f"缺少输入字段: {name}")
                source_value = raw_inputs[original_name]
            try:
                values.append(float(source_value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"字段 {name} 的值无法转换为数值") from exc
        return values

    raise ValueError("预测输入格式不支持，请传入数组或对象。")

