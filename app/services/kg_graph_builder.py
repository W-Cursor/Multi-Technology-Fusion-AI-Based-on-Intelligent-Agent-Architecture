from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.schemas.results import (
    KnowledgeGraphBridgeEntry,
    KnowledgeGraphBuildRequest,
    KnowledgeGraphBuildResponse,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    KnowledgeGraphPathEntry,
    KnowledgeGraphPathRequest,
    KnowledgeGraphPathResponse,
)


def build_knowledge_graph(
    df: pd.DataFrame,
    request: KnowledgeGraphBuildRequest,
) -> KnowledgeGraphBuildResponse:
    """根据历史数据生成基于相关性的知识图谱，并输出质量提示"""

    numeric_df = _prepare_numeric_dataframe(df)
    threshold = float(request.correlation_threshold or 0.0)

    corr_df = numeric_df.corr(method="pearson").fillna(0.0)
    edges, truncated = _build_edges(corr_df, threshold, request.max_edges)
    if not edges:
        raise ValueError("未找到满足阈值的参数关系，请适当降低阈值或补充数据。")

    nodes = _build_nodes(numeric_df, edges)
    metrics = _build_metrics(nodes, edges)
    summary = _build_summary(nodes, edges, threshold)
    warnings, isolated = _build_warnings(
        numeric_df,
        nodes,
        edges,
        threshold,
        truncated,
        request.max_edges,
    )

    metrics.update({
        "threshold": float(threshold),
        "isolated_count": float(len(isolated)),
    })

    return KnowledgeGraphBuildResponse(
        nodes=nodes,
        edges=edges,
        summary=summary,
        metrics=metrics,
        warnings=warnings,
        isolated_columns=isolated,
    )


def find_graph_paths(
    df: pd.DataFrame,
    request: KnowledgeGraphPathRequest,
) -> KnowledgeGraphPathResponse:
    """在生成的图谱上执行链路探索与共现分析"""

    numeric_df = _prepare_numeric_dataframe(df)
    threshold = float(request.correlation_threshold or 0.0)
    corr_df = numeric_df.corr(method="pearson").fillna(0.0)
    edges, _ = _build_edges(corr_df, threshold, request.max_edges)
    if not edges:
        raise ValueError("当前阈值下没有有效的关联关系，无法计算路径。")

    nodes = _build_nodes(numeric_df, edges)
    available = {node.node_id for node in nodes}
    if request.start not in available:
        raise ValueError(
            f"起点 {request.start} 在当前阈值下未形成有效关联，请降低阈值或选择其他节点。"
        )
    if request.end not in available:
        raise ValueError(
            f"终点 {request.end} 在当前阈值下未形成有效关联，请降低阈值或选择其他节点。"
        )

    adjacency, weight_lookup = _build_adjacency(edges)
    paths = _search_paths(
        adjacency,
        weight_lookup,
        request.start,
        request.end,
        request.max_depth,
        request.max_paths,
    )
    bridges = _find_bridge_nodes(adjacency, weight_lookup, request.start, request.end)

    summary: List[str] = []
    if paths:
        summary.append(
            f"找到 {len(paths)} 条深度 ≤ {request.max_depth} 的关联路径，综合评分越高代表链路越稳定。"
        )
    else:
        summary.append(
            "当前阈值下未找到直接的多跳链路，可尝试降低阈值或提高最大深度。"
        )

    if bridges:
        summary.append(
            "检测到与起终点同时高度相关的共现节点，可优先关注这些中间参数的状态。"
        )

    return KnowledgeGraphPathResponse(paths=paths, bridges=bridges[:5], summary=summary)


def _prepare_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number, "number", "float", "int"]).copy()
    if numeric_df.empty:
        raise ValueError("缺少可用于构建知识图谱的数值字段，请检查数据集。")

    numeric_df = numeric_df.dropna(how="all")
    if numeric_df.empty:
        raise ValueError("数值字段全部为空，无法计算相关性。")

    return numeric_df


def _build_edges(
    corr_df: pd.DataFrame,
    threshold: float,
    max_edges: int | None,
) -> Tuple[List[KnowledgeGraphEdge], bool]:
    correlations: List[Tuple[str, str, float]] = []
    columns = list(corr_df.columns)

    for i, col_i in enumerate(columns[:-1]):
        for j in range(i + 1, len(columns)):
            coeff = corr_df.iloc[i, j]
            if np.isnan(coeff) or abs(coeff) < threshold:
                continue
            correlations.append((col_i, columns[j], float(coeff)))

    correlations.sort(key=lambda item: abs(item[2]), reverse=True)
    truncated = False
    if max_edges and len(correlations) > max_edges:
        correlations = correlations[: max_edges]
        truncated = True

    edges: List[KnowledgeGraphEdge] = []
    for source, target, coeff in correlations:
        direction = "positive" if coeff >= 0 else "negative"
        edges.append(
            KnowledgeGraphEdge(
                source=source,
                target=target,
                coefficient=coeff,
                weight=abs(coeff),
                direction=direction,
            )
        )

    return edges, truncated


def _build_nodes(
    numeric_df: pd.DataFrame,
    edges: List[KnowledgeGraphEdge],
) -> List[KnowledgeGraphNode]:
    node_ids = set()
    for edge in edges:
        node_ids.add(edge.source)
        node_ids.add(edge.target)

    nodes: List[KnowledgeGraphNode] = []
    for node_id in sorted(node_ids):
        series = pd.to_numeric(numeric_df.get(node_id), errors="coerce")
        series = series.dropna()
        mean_value = float(series.mean()) if not series.empty else None
        std_value = float(series.std(ddof=0)) if not series.empty else None
        degree = sum(1 for edge in edges if edge.source == node_id or edge.target == node_id)
        nodes.append(
            KnowledgeGraphNode(
                node_id=node_id,
                label=node_id,
                degree=degree,
                mean_value=mean_value,
                std_value=std_value,
            )
        )

    return nodes


def _build_metrics(
    nodes: List[KnowledgeGraphNode],
    edges: List[KnowledgeGraphEdge],
) -> dict[str, float]:
    if not edges:
        return {
            "node_count": float(len(nodes)),
            "edge_count": 0.0,
            "avg_abs_corr": 0.0,
            "max_abs_corr": 0.0,
        }

    abs_coeffs = [edge.weight for edge in edges]
    return {
        "node_count": float(len(nodes)),
        "edge_count": float(len(edges)),
        "avg_abs_corr": float(np.mean(abs_coeffs)),
        "max_abs_corr": float(np.max(abs_coeffs)),
    }


def _build_summary(
    nodes: List[KnowledgeGraphNode],
    edges: List[KnowledgeGraphEdge],
    threshold: float,
) -> List[str]:
    summary = [
        f"图谱共识别 {len(nodes)} 个参数节点，{len(edges)} 条关系（阈值 ≥ {threshold:.2f}）。"
    ]

    if edges:
        top_edge = max(edges, key=lambda edge: edge.weight)
        relation = "正相关" if top_edge.coefficient >= 0 else "负相关"
        summary.append(
            f"最强关系：{top_edge.source} 与 {top_edge.target} {relation} (r = {top_edge.coefficient:.2f})。"
        )

    high_degree = [node for node in nodes if node.degree >= 3]
    if high_degree:
        hub_names = ", ".join(
            node.label for node in sorted(high_degree, key=lambda n: n.degree, reverse=True)[:5]
        )
        summary.append(f"高连接度节点建议重点关注：{hub_names}。")

    return summary


def _build_warnings(
    numeric_df: pd.DataFrame,
    nodes: List[KnowledgeGraphNode],
    edges: List[KnowledgeGraphEdge],
    threshold: float,
    truncated: bool,
    max_edges: int | None,
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    used_columns = {node.node_id for node in nodes}
    all_columns = list(numeric_df.columns)
    isolated = [col for col in all_columns if col not in used_columns]

    if isolated:
        preview = ", ".join(isolated[:6])
        suffix = " 等" if len(isolated) > 6 else ""
        warnings.append(f"以下 {len(isolated)} 个字段未形成有效关系：{preview}{suffix}")

    zero_var = [col for col in all_columns if numeric_df[col].std(ddof=0) == 0]
    if zero_var:
        preview = ", ".join(zero_var[:6])
        suffix = " 等" if len(zero_var) > 6 else ""
        warnings.append(f"检测到方差为 0 的字段：{preview}{suffix}，请检查数据是否存在常量列。")

    if threshold >= 0.7 and len(edges) < 5:
        warnings.append("阈值较高导致有效关系较少，可尝试下调阈值。")

    if threshold <= 0.2 and len(edges) > 150:
        warnings.append("阈值偏低使关系数量过多，建议适当提高阈值以突出核心链路。")

    if truncated and max_edges:
        warnings.append(f"关系数量达到上限 {max_edges} 条，部分弱关系被截断，可调高上限或阈值。")

    return warnings, isolated


def _build_adjacency(
    edges: List[KnowledgeGraphEdge],
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[Tuple[str, str], float]]:
    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    weight_lookup: Dict[Tuple[str, str], float] = {}

    for edge in edges:
        adjacency.setdefault(edge.source, []).append((edge.target, edge.weight))
        adjacency.setdefault(edge.target, []).append((edge.source, edge.weight))
        weight_lookup[_edge_key(edge.source, edge.target)] = edge.weight

    for neighbors in adjacency.values():
        neighbors.sort(key=lambda item: item[1], reverse=True)

    return adjacency, weight_lookup


def _search_paths(
    adjacency: Dict[str, List[Tuple[str, float]]],
    weight_lookup: Dict[Tuple[str, str], float],
    start: str,
    end: str,
    max_depth: int,
    max_paths: int,
) -> List[KnowledgeGraphPathEntry]:
    if start == end:
        return [KnowledgeGraphPathEntry(nodes=[start], score=1.0)]

    paths: List[KnowledgeGraphPathEntry] = []
    queue = deque([(start, [start], float("inf"))])

    while queue and len(paths) < max_paths:
        current_node, current_path, current_score = queue.popleft()

        if len(current_path) > max_depth + 1:
            continue

        if current_node == end and len(current_path) > 1:
            score = current_score if current_score != float("inf") else 0.0
            paths.append(KnowledgeGraphPathEntry(nodes=current_path, score=score))
            continue

        for neighbor, weight in adjacency.get(current_node, []):
            if neighbor in current_path:
                continue
            next_path = current_path + [neighbor]
            next_score = min(current_score, weight) if current_score != float("inf") else weight
            queue.append((neighbor, next_path, next_score))

    paths.sort(key=lambda item: item.score, reverse=True)
    return paths[:max_paths]


def _find_bridge_nodes(
    adjacency: Dict[str, List[Tuple[str, float]]],
    weight_lookup: Dict[Tuple[str, str], float],
    start: str,
    end: str,
) -> List[KnowledgeGraphBridgeEntry]:
    start_neighbors = {neighbor: weight for neighbor, weight in adjacency.get(start, [])}
    end_neighbors = {neighbor: weight for neighbor, weight in adjacency.get(end, [])}
    common = set(start_neighbors) & set(end_neighbors)

    bridges: List[KnowledgeGraphBridgeEntry] = []
    for node in common:
        source_weight = start_neighbors[node]
        target_weight = end_neighbors[node]
        combined_score = min(source_weight, target_weight)
        bridges.append(
            KnowledgeGraphBridgeEntry(
                node=node,
                source_weight=source_weight,
                target_weight=target_weight,
                combined_score=combined_score,
            )
        )

    bridges.sort(key=lambda item: item.combined_score, reverse=True)
    return bridges


def _edge_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))
