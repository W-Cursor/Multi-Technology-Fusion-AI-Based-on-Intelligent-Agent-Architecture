from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    row_count: int
    column_count: int
    timestamp_column: str | None = Field(default=None)
    timestamp_candidates: List[str] = Field(default_factory=list)
    analysis_columns: List[str]
    key_parameters: List[str]
    missing_ratio: float
    target_candidates: List[str]


class IndicatorAssessment(BaseModel):
    parameter: str
    status: str  # 高 / 低 / 正常
    trend: str  # 上升 / 下降 / 稳定
    latest_value: float | None
    mean_value: float | None
    std_dev: float | None
    influence_factors: List[str]
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)


class HistoryPoint(BaseModel):
    index: str
    value: float


class ForecastResult(BaseModel):
    horizon: int
    index: List[str]
    predictions: Dict[str, List[float]]
    trends: Dict[str, str]
    summary: Dict[str, str]
    confidence_bounds: Dict[str, List[Dict[str, float]]] | None = None
    history_tail: Dict[str, List[HistoryPoint]] | None = None


class ReportEntry(BaseModel):
    parameter: str
    current_status: str
    trend: str
    latest_value: float | None = None
    mean_value: float | None = None
    std_dev: float | None = None
    forecast_summary: str
    recommendations: List[str]
    forecast_trend: str
    anomalies: List[Dict[str, Any]] | None = None
    influence_factors: List[str] = Field(default_factory=list)


class AgentPayload(BaseModel):
    timestamp_column: str | None
    key_parameters: List[str]
    indicator_status: Dict[str, str]
    forecast_summary: Dict[str, str]


class AnalysisResponse(BaseModel):
    report: List[ReportEntry]
    forecast: ForecastResult
    agent_payload: AgentPayload
    metadata: Dict[str, Any]


class ParameterAnalysisRequest(BaseModel):
    cache_key: str
    parameter: str
    horizon: int | None = Field(default=30, ge=1, le=365)
    time_range: str | None = None


class AutoInfluenceFactor(BaseModel):
    name: str
    score: float
    direction: str
    notes: str | None = None


class ExpertFactorEntry(BaseModel):
    parameter: str
    weight: float | None = None
    note: str | None = None


class ExpertSchemeBase(BaseModel):
    name: str
    target_parameter: str
    description: str | None = None
    controlled_parameters: List[str] = Field(default_factory=list)
    factors: List[ExpertFactorEntry] = Field(default_factory=list)


class ExpertSchemeInfo(ExpertSchemeBase):
    scheme_id: str
    updated_at: str | None = None


class ExpertSchemeCreate(ExpertSchemeBase):
    cache_key: str | None = None


class ExpertSchemeUpdate(ExpertSchemeBase):
    cache_key: str | None = None


class ExpertSchemePayload(BaseModel):
    name: str
    target_parameter: str
    description: str | None = None
    controlled_parameters: List[str] = Field(default_factory=list)
    factors: List[ExpertFactorEntry] = Field(default_factory=list)
    cache_key: str | None = None


class ExpertSchemeUpdatePayload(ExpertSchemePayload):
    pass


class ExpertSchemeListResponse(BaseModel):
    schemes: List[ExpertSchemeInfo]


class RecommendationEntry(BaseModel):
    title: str
    detail: str
    risk: str | None = None


class ParameterAnalysisResponse(BaseModel):
    parameter: str
    status: str
    trend: str
    latest_value: float | None
    mean_value: float | None
    std_dev: float | None
    forecast_trend: str | None
    forecast_summary: str | None
    auto_factors: List[AutoInfluenceFactor]
    history: List[HistoryPoint]
    expert_scheme: ExpertSchemeInfo | None = None
    recommendations: List[RecommendationEntry] = Field(default_factory=list)
    available_parameters: List[str] = Field(default_factory=list)
    timestamp_column: str | None = None


class ColumnInfo(BaseModel):
    name: str
    is_numeric: bool
    suggested: bool = False


class ColumnGroup(BaseModel):
    group: str
    columns: List[ColumnInfo]

    def model_dump(self, *args, **kwargs):  # pragma: no cover - used for JSON compatibility
        return super().model_dump(*args, **kwargs)


class ColumnPreviewResponse(BaseModel):
    columns: List[str]
    numeric_columns: List[str]
    timestamp_candidates: List[str]
    suggested_key_parameters: List[str]
    grouped_columns: List[ColumnGroup]
    metadata: Dict[str, Any]
    cache_key: str | None = None


class EquilibriumResult(BaseModel):
    materials: Dict[str, float]
    efficiency: Dict[str, float]
    intermediate: Dict[str, float]


class EquilibriumRequest(BaseModel):
    mineral_data: Dict[str, List[float | int | str]] | None = None
    process_params: Dict[str, float | int | str] | None = None


class KnowledgeGraphRootCauseRequest(BaseModel):
    """知识图谱归因分析请求参数"""

    cache_key: str
    target_column: str
    target_node: str
    anomaly_type: str = Field(default="unknown")
    timestamp_column: str | None = Field(default=None)
    lookback_window_hours: int | None = Field(default=168, ge=1, le=24 * 90)
    cleaning_strategy: Literal["drop_row", "neighbor_fill", "retain_row"] = Field(
        default="drop_row",
        description="数据清洗策略。drop_row=遇到异常值时丢弃整行，neighbor_fill=使用相邻数值填补，retain_row=保留行并用均值兜底。",
    )


class KnowledgeGraphSeriesPoint(BaseModel):
    """时间序列点位信息"""

    timestamp: str
    value: float


class KnowledgeGraphFactor(BaseModel):
    """知识图谱关系因子信息"""

    node_id: str
    label: str
    relationship: str | None = None
    graph_weight: float | None = None
    correlation: float | None = None
    combined_score: float | None = None
    latest_value: float | None = None
    baseline_value: float | None = None
    delta_value: float | None = None
    matched_column: str | None = None
    regression_coef: float | None = None
    influence_score: float | None = None
    adjustment: str | None = None
    recommendation: str | None = None
    shap_value: float | None = None
    model_importance: float | None = None
    model_direction: str | None = None
    lead_lag_steps: int | None = None
    lead_lag_hours: float | None = None
    lead_lag_correlation: float | None = None
    granger_p_value: float | None = None
    lead_direction: str | None = None


class KnowledgeGraphTemporalInsight(BaseModel):
    parameter: str
    lead_hours: float | None = None
    correlation: float | None = None
    direction: str | None = None
    granger_p_value: float | None = None
    sample_size: int | None = None
    summary: str


class KnowledgeGraphRootCauseResponse(BaseModel):
    """知识图谱归因分析结果"""

    cache_key: str
    target_column: str
    target_node: str
    target_label: str
    anomaly_type: str
    timestamp_column: str | None = None
    lookback_window_hours: int | None = None
    summary: str | None = None
    root_causes: List[KnowledgeGraphFactor] = Field(default_factory=list)
    impacts: List[KnowledgeGraphFactor] = Field(default_factory=list)
    trend: List[KnowledgeGraphSeriesPoint] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    model_summary: Dict[str, Any] = Field(default_factory=dict)
    temporal_insights: List[KnowledgeGraphTemporalInsight] = Field(default_factory=list)


class KnowledgeGraphNode(BaseModel):
    node_id: str
    label: str
    degree: int
    mean_value: float | None = None
    std_value: float | None = None


class KnowledgeGraphEdge(BaseModel):
    source: str
    target: str
    coefficient: float
    weight: float
    direction: str


class KnowledgeGraphBuildRequest(BaseModel):
    cache_key: str
    timestamp_column: str | None = None
    correlation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_edges: int | None = Field(default=200, ge=1, le=2000)


class KnowledgeGraphBuildResponse(BaseModel):
    nodes: List[KnowledgeGraphNode]
    edges: List[KnowledgeGraphEdge]
    summary: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    isolated_columns: List[str] = Field(default_factory=list)


class KnowledgeGraphPathRequest(BaseModel):
    cache_key: str
    start: str
    end: str
    correlation_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    max_depth: int = Field(default=4, ge=1, le=6)
    max_paths: int = Field(default=5, ge=1, le=20)
    max_edges: int | None = Field(default=400, ge=10, le=2000)


class KnowledgeGraphPathEntry(BaseModel):
    nodes: List[str]
    score: float


class KnowledgeGraphBridgeEntry(BaseModel):
    node: str
    source_weight: float
    target_weight: float
    combined_score: float


class KnowledgeGraphPathResponse(BaseModel):
    paths: List[KnowledgeGraphPathEntry] = Field(default_factory=list)
    bridges: List[KnowledgeGraphBridgeEntry] = Field(default_factory=list)
    summary: List[str] = Field(default_factory=list)

