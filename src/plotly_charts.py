"""Interactive Plotly charts for the Streamlit analyst workspace only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.chart_generator import (
    build_spec_from_ui,
    chart_insight_for_spec,
    chart_reason_for_spec,
    find_matching_recommended_spec,
)
from src.config import COLORS

CHART_HEIGHT = 560
PALETTE_NAMES: tuple[str, ...] = (
    "Default",
    "Vibrant",
    "Professional",
    "High contrast",
    "Pastel",
)
DEFAULT_PALETTE = "Default"

POINT_COLOR_NAMES: tuple[str, ...] = ("Blue", "Orange", "Green", "Purple", "Red", "Black")
DEFAULT_POINT_COLOR = "Blue"
POINT_COLORS: dict[str, str] = {
    "Blue": "#2563EB",
    "Orange": "#EA580C",
    "Green": "#16A34A",
    "Purple": "#7C3AED",
    "Red": "#DC2626",
    "Black": "#0F172A",
}

_COLOR_PALETTES: dict[str, dict[str, Any]] = {
    "Default": {
        "sequence": ["#2563EB", "#1D4ED8", "#0EA5E9", "#2E5090", "#3B82F6", "#0284C7"],
        "trend": "#1E3A5F",
    },
    "Vibrant": {
        "sequence": ["#E11D48", "#2563EB", "#16A34A", "#D97706", "#7C3AED", "#0891B2"],
        "trend": "#0F172A",
    },
    "Professional": {
        "sequence": ["#1E3A5F", "#2E5090", "#4A6FA5", "#0F766E", "#B45309", "#5B21B6"],
        "trend": "#0F172A",
    },
    "High contrast": {
        "sequence": ["#000000", "#E11D48", "#1D4ED8", "#15803D", "#A16207", "#6D28D9"],
        "trend": "#000000",
    },
    "Pastel": {
        "sequence": ["#60A5FA", "#F472B6", "#4ADE80", "#FBBF24", "#A78BFA", "#22D3EE"],
        "trend": "#475569",
    },
}


def normalize_palette_name(name: str | None) -> str:
    """Return a valid palette key from user-facing label."""
    if not name:
        return DEFAULT_PALETTE
    key = str(name).strip()
    if key in _COLOR_PALETTES:
        return key
    lowered = key.lower()
    for palette in PALETTE_NAMES:
        if palette.lower() == lowered:
            return palette
    return DEFAULT_PALETTE


def normalize_point_color_name(name: str | None) -> str:
    """Return a valid point color label."""
    if not name:
        return DEFAULT_POINT_COLOR
    key = str(name).strip()
    if key in POINT_COLORS:
        return key
    for label in POINT_COLOR_NAMES:
        if label.lower() == key.lower():
            return label
    return DEFAULT_POINT_COLOR


def resolve_point_color_hex(name: str | None) -> str:
    """Return hex color for ungrouped scatter/bar marks."""
    return POINT_COLORS[normalize_point_color_name(name)]


def get_palette_colors(palette: str | None) -> dict[str, Any]:
    """Return trend and discrete sequence colors for grouped charts."""
    key = normalize_palette_name(palette)
    return _COLOR_PALETTES[key]


def _display_label(column_name: str) -> str:
    """Format a column name for chart titles and axis labels."""
    label = str(column_name).strip().replace("_", " ")
    return label if label else "Value"


def numeric_series_for_chart(df: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to numeric values for interactive charts."""
    return _numeric_series(df, column)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to numeric values for interactive charts."""
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _chart_height_for_bar(category_count: int) -> int:
    """Scale bar chart height with category count."""
    return int(max(480, min(620, 40 * category_count + 160)))


def _title_html(title: str, subtitle: str | None = None) -> str:
    """Build a single title block for the top margin (no in-plot annotations)."""
    if subtitle:
        return (
            f"<b>{title}</b><br>"
            f"<span style='font-size:11px;color:#64748B;font-weight:400'>{subtitle}</span>"
        )
    return f"<b>{title}</b>"


def _apply_plotly_theme(
    fig: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    height: int = CHART_HEIGHT,
    show_legend: bool = False,
    legend_on_right: bool = False,
) -> go.Figure:
    """Apply layout styling with title in margin and non-overlapping legend."""
    top_margin = 54 if subtitle else 44
    right_margin = 28
    layout_kwargs: dict[str, Any] = {"showlegend": False}

    if show_legend:
        if legend_on_right:
            layout_kwargs = {
                "showlegend": True,
                "legend": dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255,255,255,0.92)",
                    borderwidth=0,
                    font=dict(size=11),
                ),
            }
            right_margin = 108
        else:
            layout_kwargs = {
                "showlegend": True,
                "legend": dict(
                    orientation="h",
                    yanchor="top",
                    y=0.90,
                    xanchor="right",
                    x=0.98,
                    bgcolor="rgba(255,255,255,0.92)",
                    borderwidth=0,
                    font=dict(size=11),
                ),
            }
            top_margin = max(top_margin, 58)

    fig.update_layout(
        title={
            "text": _title_html(title, subtitle),
            "x": 0,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "pad": {"t": 2, "b": 4},
        },
        template="plotly_white",
        height=height,
        margin=dict(l=56, r=right_margin, t=top_margin, b=56),
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=COLORS["text"]),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        hovermode="closest",
        **layout_kwargs,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
        zeroline=False,
        automargin=True,
        title_font=dict(size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
        zeroline=False,
        automargin=True,
        title_font=dict(size=12),
    )
    return fig


def get_chart_metadata(
    df: pd.DataFrame,
    recommended: list[dict[str, Any]],
    chart_type: str,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    group_column: str | None = None,
) -> dict[str, str]:
    """Return insight and recommendation rationale for the metadata panel."""
    ui_spec = build_spec_from_ui(
        df,
        chart_type,
        x_column=x_column,
        y_column=y_column,
        group_column=group_column,
    )
    matched = find_matching_recommended_spec(
        recommended,
        chart_type,
        x_column=x_column,
        y_column=y_column,
        group_column=group_column,
    )
    spec = matched or ui_spec
    return {
        "insight": chart_insight_for_spec(df, spec),
        "reason": chart_reason_for_spec(spec),
        "is_recommended": "Yes" if matched else "Custom exploration",
    }


def build_plotly_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    color_column: str | None = None,
    show_trendline: bool = True,
    color_palette: str | None = None,
    point_color: str | None = None,
) -> go.Figure:
    """Build an interactive scatterplot with hover, grouping, and optional trendline."""
    palette = get_palette_colors(color_palette)
    sequence = palette["sequence"]
    trend_color = palette["trend"]
    mark_color = resolve_point_color_hex(point_color)

    plot_df = df.copy()
    plot_df[x_column] = _numeric_series(df, x_column)
    plot_df[y_column] = _numeric_series(df, y_column)
    columns = [x_column, y_column] + ([color_column] if color_column else [])
    data = plot_df[columns].dropna()
    if data.empty:
        raise ValueError("Not enough numeric values to plot a scatter chart.")

    use_trendline = show_trendline and len(data) >= 3 and not color_column
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        trendline=None,
        color_discrete_sequence=sequence,
        opacity=0.9 if color_column else 0.9,
        labels={x_column: _display_label(x_column), y_column: _display_label(y_column)},
    )
    if not color_column:
        fig.update_traces(
            marker=dict(size=11, color=mark_color, line=dict(width=0.6, color="white")),
            selector=dict(mode="markers"),
        )
    else:
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))

    show_legend = bool(color_column)
    if use_trendline:
        x_vals = data[x_column].to_numpy(dtype=float)
        y_vals = data[y_column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend",
                line=dict(color=trend_color, width=2.5),
                hoverinfo="skip",
                showlegend=True,
            )
        )
        show_legend = True

    if color_column:
        fig.for_each_trace(
            lambda trace: trace.update(
                hovertemplate=(
                    f"{_display_label(color_column)}=%{{fullData.name}}<br>"
                    f"{_display_label(x_column)}=%{{x}}<br>"
                    f"{_display_label(y_column)}=%{{y}}<extra></extra>"
                )
            )
        )
    else:
        fig.update_traces(
            hovertemplate=(
                f"{_display_label(x_column)}=%{{x}}<br>"
                f"{_display_label(y_column)}=%{{y}}<extra></extra>"
            ),
            selector=dict(mode="markers"),
        )

    title = f"{_display_label(y_column)} vs {_display_label(x_column)}"
    return _apply_plotly_theme(
        fig,
        title=title,
        subtitle=None,
        show_legend=show_legend,
        legend_on_right=bool(color_column),
    )


def build_plotly_boxplot(
    df: pd.DataFrame,
    group_column: str,
    y_column: str,
    *,
    color_palette: str | None = None,
) -> go.Figure:
    """Build an interactive boxplot with hover detail and grouped coloring."""
    sequence = get_palette_colors(color_palette)["sequence"]
    plot_df = df[[group_column, y_column]].copy()
    plot_df[y_column] = _numeric_series(df, y_column)
    plot_df[group_column] = plot_df[group_column].astype(str)
    plot_df = plot_df.dropna(subset=[group_column, y_column])
    if plot_df.empty:
        raise ValueError("Not enough numeric values to build a boxplot.")

    fig = px.box(
        plot_df,
        x=group_column,
        y=y_column,
        color=group_column,
        points="outliers",
        color_discrete_sequence=sequence,
        labels={group_column: _display_label(group_column), y_column: _display_label(y_column)},
    )
    fig.update_traces(
        hovertemplate=(
            f"{_display_label(group_column)}=%{{x}}<br>"
            f"{_display_label(y_column)}=%{{y}}<extra></extra>"
        ),
        boxmean=False,
        line=dict(width=1.2),
    )
    fig.update_layout(showlegend=False, boxmode="group")
    title = f"{_display_label(y_column)} by {_display_label(group_column)}"
    return _apply_plotly_theme(fig, title=title, subtitle=None, show_legend=False)


def build_plotly_bar(
    df: pd.DataFrame,
    x_column: str,
    *,
    top_n: int = 8,
    sort_desc: bool = True,
    color_palette: str | None = None,
    point_color: str | None = None,
) -> go.Figure:
    """Build an interactive horizontal bar chart with top-N filtering and hover shares."""
    mark_color = resolve_point_color_hex(point_color)
    series = df[x_column].astype(str)
    counts = series.value_counts(dropna=False).head(top_n)
    if counts.empty:
        raise ValueError("No category values available for a bar chart.")

    total = max(len(df), 1)
    bar_df = counts.reset_index()
    bar_df.columns = ["category", "count"]
    bar_df["share"] = bar_df["count"] / total
    bar_df["category"] = bar_df["category"].astype(str)
    bar_df = bar_df.sort_values("count", ascending=not sort_desc)

    fig = px.bar(
        bar_df,
        x="count",
        y="category",
        orientation="h",
        text="count",
        custom_data=["share"],
        color_discrete_sequence=[mark_color],
        labels={"count": "Records", "category": _display_label(x_column)},
    )
    fig.update_traces(
        marker_color=mark_color,
        texttemplate="%{text:,}",
        textposition="outside",
        marker=dict(line=dict(width=0.5, color="white")),
        hovertemplate=(
            f"{_display_label(x_column)}=%{{y}}<br>"
            "Count=%{x:,}<br>"
            "Share=%{customdata[0]:.1%}<extra></extra>"
        ),
        customdata=bar_df[["share"]].values,
    )
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=bar_df["category"].tolist()))
    title = f"{_display_label(x_column)} distribution"
    subtitle = f"Top {len(bar_df)} categories"
    height = _chart_height_for_bar(len(bar_df))
    return _apply_plotly_theme(fig, title=title, subtitle=subtitle, height=height, show_legend=False)


def build_interactive_chart(
    df: pd.DataFrame,
    chart_type: str,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    group_column: str | None = None,
    color_column: str | None = None,
    show_trendline: bool = True,
    top_n: int = 8,
    sort_desc: bool = True,
    color_palette: str | None = None,
    point_color: str | None = None,
) -> go.Figure:
    """Build the appropriate interactive Plotly chart for the current selection."""
    if chart_type == "scatter":
        return build_plotly_scatter(
            df,
            x_column or "",
            y_column or "",
            color_column=color_column,
            show_trendline=show_trendline,
            color_palette=color_palette,
            point_color=point_color,
        )
    if chart_type == "boxplot":
        return build_plotly_boxplot(
            df,
            group_column or "",
            y_column or "",
            color_palette=color_palette,
        )
    return build_plotly_bar(
        df,
        x_column or "",
        top_n=top_n,
        sort_desc=sort_desc,
        color_palette=color_palette,
        point_color=point_color,
    )
