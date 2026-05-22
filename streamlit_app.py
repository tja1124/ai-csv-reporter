"""Minimal Streamlit analyst workspace for interactive CSV exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.analyzer import (
    get_categorical_summary,
    get_dataset_overview,
    get_missing_values,
)
from src.column_utils import assess_grouped_comparison, get_groupable_categorical_columns
from src.config import COLORS, OUTPUTS_DIR
from src.data_loader import load_csv, validate_dataframe
from src.format_utils import apply_axis_format, format_number
from src.insight_generator import generate_report_insights
from src.logger import setup_logging
from src.main import run_report

UPLOAD_DIR = OUTPUTS_DIR / "uploads"
CHART_TYPES = ("scatter", "boxplot", "bar")


def _apply_chart_style() -> None:
    """Apply the same visual theme used by the PDF chart pipeline."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.6,
            "font.family": "sans-serif",
            "savefig.facecolor": COLORS["background"],
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    """Apply consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.set_axisbelow(True)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns suitable for chart axes."""
    return [column for column in df.select_dtypes(include="number").columns]


def _group_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns suitable for grouping."""
    return get_groupable_categorical_columns(df)


def _bar_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns suitable for bar charts."""
    return _group_columns(df)


def _missing_percentage(missing_values: dict[str, Any], overview: dict[str, Any]) -> float:
    """Calculate overall missing-value percentage across the dataset."""
    total_cells = max(overview["row_count"] * overview["column_count"], 1)
    return round((missing_values["total_missing"] / total_cells) * 100, 1)


def _save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded CSV so existing loader/CLI paths can reuse it."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _load_dataset(uploaded_file: Any) -> tuple[pd.DataFrame, Path, list[str]]:
    """Load and validate an uploaded CSV through the existing backend."""
    csv_path = _save_uploaded_file(uploaded_file)
    df = load_csv(csv_path)
    warnings = validate_dataframe(df)
    return df, csv_path, warnings


def _analyze_dataset(
    df: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    """Run the same core analysis used by the CLI report pipeline."""
    overview = get_dataset_overview(df)
    missing_values = get_missing_values(df)
    categorical_summary = get_categorical_summary(df)
    report_insights = generate_report_insights(
        df=df,
        overview=overview,
        missing_values=missing_values,
        categorical_summary=categorical_summary,
        validation_warnings=warnings,
    )
    return {
        "overview": overview,
        "missing_values": missing_values,
        "report_insights": report_insights,
    }


def _validate_chart_selection(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    group_column: str | None,
) -> str | None:
    """Return an error message when the selected chart configuration is invalid."""
    if chart_type == "scatter":
        if not x_column or not y_column:
            return "Select both an X column and a Y column."
        if x_column == y_column:
            return "Choose two different numeric columns for a scatterplot."
        return None

    if chart_type == "boxplot":
        if not group_column or not y_column:
            return "Select a Group by column and a Y column."
        assessment = assess_grouped_comparison(df, group_column, y_column)
        if assessment is None:
            return "This grouping does not have enough data for a reliable boxplot."
        return None

    if chart_type == "bar":
        if not x_column:
            return "Select an X column for the bar chart."
        if x_column not in _bar_columns(df):
            return "Choose a groupable categorical column for the bar chart."
        return None

    return "Unsupported chart type."


def _build_scatter_figure(df: pd.DataFrame, x_column: str, y_column: str) -> plt.Figure:
    """Build a scatterplot figure for the Streamlit workspace."""
    _apply_chart_style()
    data = df[[x_column, y_column]].dropna()
    x = data[x_column]
    y = data[y_column]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, y, color=COLORS["secondary"], alpha=0.8, edgecolors="white", s=48)

    if len(x) >= 3:
        trend = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.poly1d(trend)(x_line), color=COLORS["primary"], linewidth=2)

    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    apply_axis_format(ax, x_column, axis="x")
    apply_axis_format(ax, y_column, axis="y")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def _build_boxplot_figure(df: pd.DataFrame, group_column: str, y_column: str) -> plt.Figure:
    """Build a boxplot figure for the Streamlit workspace."""
    _apply_chart_style()
    grouped = [
        group[y_column].dropna().values
        for _, group in df.groupby(group_column, observed=True)
        if not group[y_column].dropna().empty
    ]
    labels = [
        str(label)
        for label, group in df.groupby(group_column, observed=True)
        if not group[y_column].dropna().empty
    ]

    rotation = 20 if len(labels) > 3 else 0
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(
        grouped,
        tick_labels=labels,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": COLORS["accent"], "color": COLORS["primary"]},
        medianprops={"color": COLORS["primary"], "linewidth": 2},
    )
    ax.set_title(f"{y_column} by {group_column}")
    ax.set_xlabel(group_column)
    ax.set_ylabel(y_column)
    ax.tick_params(axis="x", rotation=rotation)
    if rotation:
        for label in ax.get_xticklabels():
            label.set_ha("right")
    apply_axis_format(ax, y_column, axis="y")
    _style_axes(ax)
    fig.subplots_adjust(bottom=0.20 if rotation else 0.12)
    fig.tight_layout()
    return fig


def _build_bar_figure(df: pd.DataFrame, x_column: str) -> plt.Figure:
    """Build a horizontal bar chart for a categorical distribution."""
    _apply_chart_style()
    value_counts = df[x_column].value_counts(dropna=False).head(8)
    labels = [str(label) for label in value_counts.index[::-1]]
    values = value_counts.values[::-1]

    fig, ax = plt.subplots(figsize=(8, max(3.8, 0.45 * len(labels) + 2.0)))
    bars = ax.barh(labels, values, color=COLORS["secondary"], edgecolor="white", height=0.65)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            format_number(value),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_title(f"{x_column} Distribution")
    ax.set_xlabel("Count")
    apply_axis_format(ax, None, axis="x")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def _build_chart_figure(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    group_column: str | None,
) -> plt.Figure:
    """Build the matplotlib figure for the current dropdown selection."""
    if chart_type == "scatter":
        return _build_scatter_figure(df, x_column or "", y_column or "")
    if chart_type == "boxplot":
        return _build_boxplot_figure(df, group_column or "", y_column or "")
    return _build_bar_figure(df, x_column or "")


def _render_metrics(analysis: dict[str, Any]) -> None:
    """Show top-level dataset metrics."""
    overview = analysis["overview"]
    missing_values = analysis["missing_values"]
    quality = analysis["report_insights"]["quality_score"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{overview['row_count']:,}")
    col2.metric("Columns", overview["column_count"])
    col3.metric("Quality Score", f"{quality['score']}/100")
    col4.metric("Missing Data", f"{_missing_percentage(missing_values, overview)}%")


def _render_chart_controls(df: pd.DataFrame) -> tuple[str, str | None, str | None, str | None]:
    """Render chart dropdown controls and return the selected configuration."""
    numeric_cols = _numeric_columns(df)
    group_cols = _group_columns(df)
    bar_cols = _bar_columns(df)

    chart_type = st.selectbox("Chart type", CHART_TYPES)

    x_column: str | None = None
    y_column: str | None = None
    group_column: str | None = None

    if chart_type == "scatter":
        x_column = st.selectbox("X column", numeric_cols, key="scatter_x")
        y_options = [column for column in numeric_cols if column != x_column]
        y_column = st.selectbox("Y column", y_options or numeric_cols, key="scatter_y")
    elif chart_type == "boxplot":
        group_column = st.selectbox("Group by", group_cols or df.columns.tolist(), key="box_group")
        y_column = st.selectbox("Y column", numeric_cols or df.columns.tolist(), key="box_y")
    else:
        x_column = st.selectbox("X column", bar_cols or df.columns.tolist(), key="bar_x")

    return chart_type, x_column, y_column, group_column


def main() -> None:
    """Run the Streamlit analyst workspace."""
    setup_logging()

    st.set_page_config(page_title="CSV Analyst Workspace", layout="wide")
    st.title("CSV Analyst Workspace")
    st.caption("Local prototype for exploring datasets and generating the existing PDF report pipeline.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV to begin analysis.")
        return

    try:
        df, csv_path, warnings = _load_dataset(uploaded_file)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    if warnings:
        for warning in warnings:
            st.warning(warning)

    analysis = _analyze_dataset(df, warnings)
    _render_metrics(analysis)

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Chart Builder")
    chart_type, x_column, y_column, group_column = _render_chart_controls(df)

    validation_error = _validate_chart_selection(
        df,
        chart_type,
        x_column,
        y_column,
        group_column,
    )

    if validation_error:
        st.info(validation_error)
    else:
        figure = _build_chart_figure(df, chart_type, x_column, y_column, group_column)
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)

    st.divider()
    st.subheader("Export PDF Report")
    use_ai_summary = st.checkbox(
        "Include AI-assisted executive summary",
        help="Requires OPENAI_API_KEY. Falls back to the deterministic summary if unavailable.",
    )

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Running the existing report pipeline..."):
            try:
                report_path = run_report(csv_path, use_ai_summary=use_ai_summary)
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")
            else:
                st.success("PDF report generated successfully.")
                st.code(str(report_path))


if __name__ == "__main__":
    main()
