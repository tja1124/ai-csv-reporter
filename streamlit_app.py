"""Polished Streamlit analyst workspace for interactive CSV exploration."""

from __future__ import annotations

import hashlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.chart_generator import get_chart_column_options, get_recommended_charts
from src.column_utils import assess_grouped_comparison, get_groupable_categorical_columns
from src.config import COLORS
from src.format_utils import apply_axis_format, format_correlation, format_number, format_value_range
from src.logger import setup_logging
from src.main import run_report
from src.config import ensure_output_dirs
from src.streamlit_helpers import (
    build_column_summary,
    filter_preview_dataframe,
    inject_app_styles,
    load_and_analyze_dataset,
    openai_key_available,
    render_app_header,
    render_kpi_dashboard,
    spec_to_chart_state,
)

CHART_TYPES = ("scatter", "boxplot", "bar")
NAV_PAGES = ("Overview", "Explore Data", "Chart Builder", "Export Report")
CHART_HEIGHT = 4.6


def _apply_chart_style() -> None:
    """Apply the shared chart theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "font.family": "sans-serif",
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    """Apply consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def _chart_caption(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    group_column: str | None,
) -> str:
    """Build a short analyst-style caption for the active chart."""
    if chart_type == "scatter" and x_column and y_column:
        corr = df[[x_column, y_column]].corr(numeric_only=True).iloc[0, 1]
        if pd.isna(corr):
            return f"Review how {y_column} changes across {x_column}."
        direction = "rises" if corr > 0 else "falls"
        series = df[y_column].dropna()
        return (
            f"{y_column} {direction} with {x_column} (r={format_correlation(corr)}); "
            f"{format_value_range(y_column, series.min(), series.max())}."
        )

    if chart_type == "boxplot" and group_column and y_column:
        grouped = df.groupby(group_column, observed=True)[y_column].mean(numeric_only=True)
        top_group = grouped.idxmax()
        return (
            f"Highest average {y_column} in '{top_group}' "
            f"({format_number(grouped.max(), y_column)}) by {group_column}."
        )

    if chart_type == "bar" and x_column:
        counts = df[x_column].value_counts(dropna=False)
        top_value = counts.index[0]
        share = counts.iloc[0] / max(len(df), 1)
        return f"'{top_value}' leads {x_column} ({share:.0%} of records)."

    return "Adjust the chart controls to explore a different view."


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
        if assess_grouped_comparison(df, group_column, y_column) is None:
            return "This grouping does not have enough data for a reliable boxplot."
        return None

    if chart_type == "bar":
        if not x_column:
            return "Select an X column for the bar chart."
        if x_column not in get_groupable_categorical_columns(df):
            return "Choose a groupable categorical column for the bar chart."
        return None

    return "Unsupported chart type."


def _build_scatter_figure(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    show_trendline: bool,
    color_column: str | None,
) -> plt.Figure:
    """Build a scatterplot figure for the workspace."""
    _apply_chart_style()
    columns = [x_column, y_column] + ([color_column] if color_column else [])
    data = df[columns].dropna()
    x = data[x_column]
    y = data[y_column]

    fig, ax = plt.subplots(figsize=(9, CHART_HEIGHT))
    if color_column and color_column in data.columns:
        groups = data[color_column].astype(str)
        for group_name, subset in data.groupby(groups, observed=True):
            ax.scatter(
                subset[x_column],
                subset[y_column],
                alpha=0.78,
                s=46,
                label=str(group_name),
            )
        ax.legend(frameon=False, title=color_column, loc="best", fontsize=8)
    else:
        ax.scatter(x, y, color=COLORS["secondary"], alpha=0.82, edgecolors="white", s=50)

    if show_trendline and len(x) >= 3:
        trend = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.poly1d(trend)(x_line), color=COLORS["primary"], linewidth=2, label="Trend")

    ax.set_title(f"{y_column} vs {x_column}", pad=10)
    ax.set_xlabel(x_column, labelpad=8)
    ax.set_ylabel(y_column, labelpad=8)
    apply_axis_format(ax, x_column, axis="x")
    apply_axis_format(ax, y_column, axis="y")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def _build_boxplot_figure(df: pd.DataFrame, group_column: str, y_column: str) -> plt.Figure:
    """Build a boxplot figure for the workspace."""
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

    rotation = 25 if len(labels) > 3 or max((len(label) for label in labels), default=0) > 10 else 0
    fig, ax = plt.subplots(figsize=(9, CHART_HEIGHT))
    ax.boxplot(
        grouped,
        tick_labels=labels,
        widths=0.52,
        patch_artist=True,
        boxprops={"facecolor": COLORS["accent"], "color": COLORS["primary"], "linewidth": 1.1},
        medianprops={"color": COLORS["primary"], "linewidth": 2},
        whiskerprops={"color": COLORS["muted"]},
        capprops={"color": COLORS["muted"]},
    )
    ax.set_title(f"{y_column} by {group_column}", pad=10)
    ax.set_xlabel(group_column, labelpad=10)
    ax.set_ylabel(y_column, labelpad=10)
    ax.tick_params(axis="x", labelsize=9, rotation=rotation, pad=4)
    if rotation:
        for label in ax.get_xticklabels():
            label.set_ha("right")
    apply_axis_format(ax, y_column, axis="y")
    _style_axes(ax)
    fig.subplots_adjust(bottom=0.22 if rotation else 0.14)
    fig.tight_layout()
    return fig


def _build_bar_figure(df: pd.DataFrame, x_column: str, *, top_n: int) -> plt.Figure:
    """Build a horizontal bar chart for a categorical distribution."""
    _apply_chart_style()
    value_counts = df[x_column].value_counts(dropna=False).head(top_n)
    labels = [str(label) for label in value_counts.index[::-1]]
    values = value_counts.values[::-1]

    fig, ax = plt.subplots(figsize=(9, max(3.8, 0.42 * len(labels) + 2.2)))
    bars = ax.barh(labels, values, color=COLORS["secondary"], edgecolor="white", height=0.68)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            format_number(value),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_title(f"{x_column} Distribution", pad=10)
    ax.set_xlabel("Count", labelpad=8)
    ax.set_xlim(0, max(values) * 1.14)
    apply_axis_format(ax, None, axis="x")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def _init_chart_state(package: dict[str, Any]) -> None:
    """Initialize chart control session state from smart recommendations."""
    if "chart_state" in st.session_state:
        return

    recommended = package["recommended_charts"]
    st.session_state["chart_state"] = spec_to_chart_state(recommended[0]) if recommended else {
        "chart_type": "scatter",
        "x_column": None,
        "y_column": None,
        "group_column": None,
    }


def _render_sidebar(package: dict[str, Any]) -> str:
    """Render sidebar navigation and dataset overview."""
    overview = package["overview"]
    quality = package["report_insights"]["quality_score"]

    with st.sidebar:
        st.markdown("### Navigation")
        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "Overview"
        page = st.radio(
            "Workspace section",
            NAV_PAGES,
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Dataset")
        st.caption(package["csv_path"])
        st.write(f"**Rows:** {overview['row_count']:,}")
        st.write(f"**Columns:** {overview['column_count']}")
        st.write(f"**Quality:** {quality['score']}/100 ({quality['rating']})")

        with st.expander("Advanced options"):
            st.caption("Environment")
            if openai_key_available():
                st.success("OPENAI_API_KEY detected")
            else:
                st.info("AI summary uses deterministic fallback unless OPENAI_API_KEY is set.")
            st.caption("Uploads are stored temporarily under outputs/uploads/.")

        st.markdown("---")
        st.caption("Local prototype · CLI pipeline unchanged")

    return page


def _render_overview_page(package: dict[str, Any]) -> None:
    """Render the overview dashboard."""
    render_kpi_dashboard(package)

    with st.container(border=True):
        st.subheader("Analyst Snapshot")
        findings = package["report_insights"].get("important_findings") or package["report_insights"].get("key_insights", [])
        display_findings = [item for item in findings if not str(item).startswith("Sample Size Caution")][:4]
        if display_findings:
            for finding in display_findings:
                st.markdown(f"- {finding}")
        else:
            st.caption("No major findings flagged in the initial review.")

    recommended = package["recommended_charts"]
    if recommended:
        st.subheader("Recommended Views")
        st.caption("These charts come from the same deterministic planner used in PDF reports.")
        for index, spec in enumerate(recommended[:4]):
            label = spec.get("type", "chart")
            if spec["type"] == "scatter":
                title = f"{spec['y']} vs {spec['x']}"
            elif spec["type"] == "boxplot":
                title = f"{spec['y']} by {spec['x']}"
            else:
                title = f"{spec.get('column', 'category')} distribution"
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{label.title()}** · {title}")
            if cols[1].button("Use", key=f"use_rec_{index}"):
                st.session_state["chart_state"] = spec_to_chart_state(spec)
                st.session_state["nav_page"] = "Chart Builder"
                st.rerun()


def _render_explore_page(package: dict[str, Any]) -> None:
    """Render searchable dataset exploration tools."""
    df = package["df"]
    column_info = package["column_info"]

    st.subheader("Column Summary")
    st.dataframe(build_column_summary(column_info, df), use_container_width=True, hide_index=True)

    st.subheader("Data Preview")
    controls = st.columns([2, 1, 1, 1])
    search_text = controls[0].text_input("Search rows", placeholder="Filter across visible columns")
    row_limit = controls[1].selectbox("Rows", [10, 25, 50, 100], index=0)
    sort_column = controls[2].selectbox("Sort by", ["None", *df.columns.tolist()])
    sort_ascending = controls[3].toggle("Ascending", value=True)

    selected_columns = st.multiselect(
        "Columns to display",
        options=list(df.columns),
        default=list(df.columns),
    )

    preview = filter_preview_dataframe(
        df,
        search_text=search_text,
        selected_columns=selected_columns,
        row_limit=row_limit,
        sort_column=None if sort_column == "None" else sort_column,
        sort_ascending=sort_ascending,
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(preview):,} of {len(df):,} rows")


def _render_chart_page(package: dict[str, Any]) -> None:
    """Render the interactive chart builder."""
    df = package["df"]
    options = package["chart_options"]
    _init_chart_state(package)
    state = st.session_state["chart_state"]

    left, right = st.columns([1, 1.8], gap="medium")

    with left:
        st.subheader("Chart Controls")
        supported = [chart for chart in CHART_TYPES if chart in options["supported_chart_types"]] or list(CHART_TYPES)
        chart_type = st.selectbox(
            "Chart type",
            supported,
            index=supported.index(state["chart_type"]) if state.get("chart_type") in supported else 0,
        )

        x_column = state.get("x_column")
        y_column = state.get("y_column")
        group_column = state.get("group_column")
        color_column: str | None = None
        show_trendline = True
        top_n = 8

        numeric_cols = options["available_y_columns"]
        group_cols = options["available_group_columns"]
        bar_cols = [column for column in options["available_x_columns"] if column in group_cols]

        if chart_type == "scatter":
            x_column = st.selectbox("X column", numeric_cols, index=_safe_index(numeric_cols, state.get("x_column")))
            y_options = [column for column in numeric_cols if column != x_column]
            y_column = st.selectbox("Y column", y_options or numeric_cols, index=_safe_index(y_options or numeric_cols, state.get("y_column")))
            color_column = st.selectbox("Color by (optional)", ["None", *group_cols])
            color_column = None if color_column == "None" else color_column
            show_trendline = st.toggle("Show trendline", value=True)
        elif chart_type == "boxplot":
            group_column = st.selectbox("Group by", group_cols, index=_safe_index(group_cols, state.get("group_column")))
            y_column = st.selectbox("Y column", numeric_cols, index=_safe_index(numeric_cols, state.get("y_column")))
        else:
            x_column = st.selectbox("X column", bar_cols or group_cols, index=_safe_index(bar_cols or group_cols, state.get("x_column")))
            top_n = st.slider("Top categories", min_value=3, max_value=12, value=8)

        st.session_state["chart_state"] = {
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "group_column": group_column,
        }

    validation_error = _validate_chart_selection(df, chart_type, x_column, y_column, group_column)

    with right:
        st.subheader("Chart Preview")
        if validation_error:
            st.info(validation_error)
        else:
            if chart_type == "scatter":
                figure = _build_scatter_figure(
                    df,
                    x_column or "",
                    y_column or "",
                    show_trendline=show_trendline,
                    color_column=color_column,
                )
            elif chart_type == "boxplot":
                figure = _build_boxplot_figure(df, group_column or "", y_column or "")
            else:
                figure = _build_bar_figure(df, x_column or "", top_n=top_n)

            st.pyplot(figure, use_container_width=True, clear_figure=True)
            plt.close(figure)
            st.markdown(
                f"**Insight:** {_chart_caption(df, chart_type, x_column, y_column, group_column)}"
            )


def _render_export_page(package: dict[str, Any]) -> None:
    """Render the PDF export workflow."""
    st.subheader("Generate PDF Report")
    st.caption("Uses the same backend pipeline as `python main.py your_file.csv`.")

    with st.expander("Report configuration", expanded=True):
        report_title = st.text_input("Report title (optional)", placeholder="Business Intelligence Report")
        analyst_name = st.text_input("Analyst name (optional)", placeholder="Prepared by")
        use_ai_summary = st.checkbox(
            "Include AI-assisted executive summary",
            value=False,
            help="Requires OPENAI_API_KEY in the environment.",
        )
        if use_ai_summary and not openai_key_available():
            st.warning("OPENAI_API_KEY is not set. The report will use the deterministic fallback summary.")

    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting report pipeline...")
        try:
            progress.progress(25, text="Analyzing dataset and generating charts...")
            report_path = run_report(
                package["csv_path"],
                use_ai_summary=use_ai_summary,
                report_title=report_title or None,
                analyst_name=analyst_name or None,
            )
            progress.progress(100, text="Report complete.")
        except Exception as exc:
            progress.empty()
            st.error(f"Report generation failed: {exc}")
        else:
            st.success("PDF report generated successfully.")
            st.markdown(f"**Saved to:** `{report_path}`")
            st.code(str(report_path))


def _safe_index(options: list[str], value: str | None) -> int:
    """Return a safe selectbox index for a stored chart state value."""
    if value in options:
        return options.index(value)
    return 0


def main() -> None:
    """Run the Streamlit analyst workspace."""
    setup_logging()
    ensure_output_dirs()
    st.set_page_config(
        page_title="AI CSV Reporter",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_app_styles()
    render_app_header()

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV to begin analysis.")
        return

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    try:
        package = load_and_analyze_dataset(file_hash, uploaded_file.name, file_bytes)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    package["recommended_charts"] = get_recommended_charts(package["df"])
    package["chart_options"] = get_chart_column_options(package["df"])

    for warning in package["warnings"]:
        st.warning(warning)

    page = _render_sidebar(package)

    if page == "Overview":
        _render_overview_page(package)
    elif page == "Explore Data":
        _render_explore_page(package)
    elif page == "Chart Builder":
        _render_chart_page(package)
    else:
        _render_export_page(package)


if __name__ == "__main__":
    main()
