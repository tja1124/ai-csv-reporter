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

CHART_HEIGHT = 480
COLOR_SEQUENCE = [COLORS["secondary"], COLORS["primary"], COLORS["accent"], "#7BA7D7", "#4A6FA5"]


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


def _apply_plotly_theme(
    fig: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    height: int = CHART_HEIGHT,
) -> go.Figure:
    """Apply consistent layout styling across interactive charts."""
    fig.update_layout(
        title={"text": f"<b>{title}</b><br><sup>{subtitle}</sup>" if subtitle else f"<b>{title}</b>", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=height,
        margin=dict(l=48, r=24, t=72 if subtitle else 56, b=48),
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color=COLORS["text"]),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], linecolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], linecolor=COLORS["grid"], zeroline=False)
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
) -> go.Figure:
    """Build an interactive scatterplot with hover, grouping, and optional trendline."""
    plot_df = df.copy()
    plot_df[x_column] = _numeric_series(df, x_column)
    plot_df[y_column] = _numeric_series(df, y_column)
    columns = [x_column, y_column] + ([color_column] if color_column else [])
    data = plot_df[columns].dropna()
    if data.empty:
        raise ValueError("Not enough numeric values to plot a scatter chart.")

    use_trendline = show_trendline and len(data) >= 3
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        trendline=None,
        color_discrete_sequence=COLOR_SEQUENCE,
        opacity=0.82,
        labels={x_column: _display_label(x_column), y_column: _display_label(y_column)},
    )
    if use_trendline and not color_column:
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
                line=dict(color=COLORS["primary"], width=2),
                hoverinfo="skip",
                showlegend=True,
            )
        )
    fig.update_traces(
        marker=dict(size=9, line=dict(width=0.5, color="white")),
    )
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
            )
        )

    title = f"{_display_label(y_column)} vs {_display_label(x_column)}"
    subtitle = "Drag to zoom · Double-click to reset · Hover for values"
    return _apply_plotly_theme(fig, title=title, subtitle=subtitle)


def build_plotly_boxplot(df: pd.DataFrame, group_column: str, y_column: str) -> go.Figure:
    """Build an interactive boxplot with hover detail and grouped coloring."""
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
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={group_column: _display_label(group_column), y_column: _display_label(y_column)},
    )
    fig.update_traces(
        hovertemplate=(
            f"{_display_label(group_column)}=%{{x}}<br>"
            f"{_display_label(y_column)}=%{{y}}<extra></extra>"
        ),
        boxmean=False,
    )
    fig.update_layout(showlegend=False, boxmode="group")
    title = f"{_display_label(y_column)} by {_display_label(group_column)}"
    subtitle = "Compare distributions across categories"
    return _apply_plotly_theme(fig, title=title, subtitle=subtitle)


def build_plotly_bar(
    df: pd.DataFrame,
    x_column: str,
    *,
    top_n: int = 8,
    sort_desc: bool = True,
) -> go.Figure:
    """Build an interactive horizontal bar chart with top-N filtering and hover shares."""
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
        color_discrete_sequence=[COLORS["secondary"]],
        labels={"count": "Records", "category": _display_label(x_column)},
    )
    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate=(
            f"{_display_label(x_column)}=%{{y}}<br>"
            "Count=%{x:,}<br>"
            "Share=%{customdata[0]:.1%}<extra></extra>"
        ),
        customdata=bar_df[["share"]].values,
    )
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=bar_df["category"].tolist()))
    title = f"{_display_label(x_column)} distribution"
    subtitle = f"Top {len(bar_df)} categories · Hover for count and share"
    return _apply_plotly_theme(fig, title=title, subtitle=subtitle, height=max(420, 36 * len(bar_df) + 120))


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
) -> go.Figure:
    """Build the appropriate interactive Plotly chart for the current selection."""
    if chart_type == "scatter":
        return build_plotly_scatter(
            df,
            x_column or "",
            y_column or "",
            color_column=color_column,
            show_trendline=show_trendline,
        )
    if chart_type == "boxplot":
        return build_plotly_boxplot(df, group_column or "", y_column or "")
    return build_plotly_bar(df, x_column or "", top_n=top_n, sort_desc=sort_desc)
