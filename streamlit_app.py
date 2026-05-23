"""Polished Streamlit analyst workspace for interactive CSV exploration."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.column_utils import assess_grouped_comparison
from src.config import ensure_output_dirs
from src.logger import setup_logging
from src.main import run_report
from src.plotly_charts import build_interactive_chart, get_chart_metadata, numeric_series_for_chart as _numeric_series_for_chart
from src.streamlit_helpers import (
    align_chart_state_to_dataframe,
    apply_pending_session_updates,
    build_column_summary,
    enrich_analysis_package,
    file_fingerprint,
    filter_preview_dataframe,
    get_sample_csv_bytes,
    inject_app_styles,
    load_and_analyze_dataset,
    load_sample_dataset,
    openai_key_available,
    queue_recommended_chart,
    render_app_header,
    render_dataset_warnings,
    render_kpi_dashboard,
    reset_dataset_session,
    spec_to_chart_state,
)

CHART_TYPES = ("scatter", "boxplot", "bar")
NAV_PAGES = ("Overview", "Explore Data", "Chart Builder", "Export Report")


def _display_label(column_name: str) -> str:
    """Format a column name for chart titles and axis labels."""
    label = str(column_name).strip().replace("_", " ")
    return label if label else "Value"


def _chart_spec_title(spec: dict[str, Any]) -> str:
    """Readable title for a recommended chart spec."""
    chart_type = spec.get("type", "chart")
    if chart_type == "scatter":
        return f"{_display_label(spec['y'])} vs {_display_label(spec['x'])}"
    if chart_type == "boxplot":
        return f"{_display_label(spec['y'])} by {_display_label(spec['x'])}"
    column = spec.get("column", "category")
    return f"{_display_label(column)} distribution"


def _chart_data_preview(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    group_column: str | None,
    *,
    limit: int = 12,
) -> pd.DataFrame:
    """Return rows used for the active chart configuration."""
    columns: list[str] = []
    if chart_type == "scatter":
        columns = [c for c in (x_column, y_column) if c]
    elif chart_type == "boxplot":
        columns = [c for c in (group_column, y_column) if c]
    else:
        columns = [x_column] if x_column else []
    columns = list(dict.fromkeys([c for c in columns if c in df.columns]))
    if not columns:
        return pd.DataFrame()
    return df[columns].dropna(how="all").head(limit)


def _validate_chart_selection(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    group_column: str | None,
    options: dict[str, Any],
) -> str | None:
    """Return an error message when the selected chart configuration is invalid."""
    if chart_type == "scatter":
        if not x_column or not y_column:
            return "Select both an X column and a Y column."
        if x_column == y_column:
            return "Choose two different numeric columns for a scatterplot."
        if x_column not in options["available_y_columns"] or y_column not in options["available_y_columns"]:
            return "Selected columns must be numeric or mostly numeric for a scatterplot."
        if _numeric_series_for_chart(df, x_column).notna().sum() < 2 or _numeric_series_for_chart(df, y_column).notna().sum() < 2:
            return "Not enough numeric values to plot a scatter chart."
        return None

    if chart_type == "boxplot":
        if not group_column or not y_column:
            return "Select a Group by column and a Y column."
        if group_column not in df.columns or y_column not in df.columns:
            return "Selected columns are not available in this dataset."
        if assess_grouped_comparison(df, group_column, y_column) is None:
            return "This grouping does not have enough data for a reliable boxplot."
        return None

    if chart_type == "bar":
        if not x_column:
            return "Select an X column for the bar chart."
        if x_column not in df.columns:
            return "Selected column is not available in this dataset."
        if df[x_column].nunique(dropna=True) > 30:
            return "This column has too many categories for a compact bar chart. Try a lower-cardinality field."
        if df[x_column].dropna().empty:
            return "Selected column has no values to chart."
        return None

    return "Unsupported chart type."


def _init_chart_state(package: dict[str, Any]) -> None:
    """Initialize chart control session state from smart recommendations."""
    df = package["df"]
    if "chart_state" not in st.session_state:
        recommended = package["recommended_charts"]
        if recommended:
            st.session_state["chart_state"] = spec_to_chart_state(recommended[0], df)
        else:
            options = package["chart_options"]
            numeric = options["available_y_columns"]
            st.session_state["chart_state"] = {
                "chart_type": "scatter" if len(numeric) >= 2 else "bar",
                "x_column": numeric[0] if numeric else None,
                "y_column": numeric[1] if len(numeric) > 1 else None,
                "group_column": None,
            }
    st.session_state["chart_state"] = align_chart_state_to_dataframe(st.session_state["chart_state"], df)


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
        st.caption(package.get("source_label", package["csv_path"]))
        if package.get("source_label") and package.get("source_label") != package["csv_path"]:
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
            title = _chart_spec_title(spec)
            label = spec.get("type", "chart").title()
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{label}** · {title}")
            cols[1].button("Use", key=f"use_rec_{index}", on_click=queue_recommended_chart, args=(spec, package["df"]))
    else:
        st.info("No automatic chart recommendations for this dataset. Open **Chart Builder** to explore columns manually.")


def _render_explore_page(package: dict[str, Any]) -> None:
    """Render searchable dataset exploration tools."""
    df = package["df"]
    column_info = package["column_info"]

    st.subheader("Column Summary")
    summary = build_column_summary(column_info, df)
    if summary.empty:
        st.info("No columns available to summarize.")
    else:
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Data Preview")
    if df.empty:
        st.warning("This dataset has no rows to preview.")
        return

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
    if not selected_columns:
        st.warning("Select at least one column to preview data.")
        return

    preview = filter_preview_dataframe(
        df,
        search_text=search_text,
        selected_columns=selected_columns,
        row_limit=row_limit,
        sort_column=None if sort_column == "None" else sort_column,
        sort_ascending=sort_ascending,
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)
    if preview.empty:
        st.caption("No rows match the current filters.")
    else:
        st.caption(f"Showing {len(preview):,} of {len(df):,} rows")


def _render_chart_page(package: dict[str, Any]) -> None:
    """Render the interactive Plotly chart builder."""
    df = package["df"]
    options = package["chart_options"]
    recommended = package.get("recommended_charts", [])
    _init_chart_state(package)
    state = st.session_state["chart_state"]

    st.subheader("Chart Builder")
    st.caption("Interactive exploration powered by Plotly. PDF reports still use the matplotlib export pipeline.")

    controls_col, workspace_col = st.columns([1, 2.2], gap="large")

    with controls_col:
        with st.container(border=True):
            st.markdown("**Chart controls**")
            supported = [c for c in CHART_TYPES if c in options["supported_chart_types"]] or list(CHART_TYPES)
            chart_type = st.selectbox(
                "Chart type",
                supported,
                index=_safe_index(supported, state.get("chart_type")),
                key="chart_type_select",
            )

            x_column = state.get("x_column")
            y_column = state.get("y_column")
            group_column = state.get("group_column")
            color_column: str | None = None
            show_trendline = True
            top_n = 8
            sort_desc = True

            numeric_cols = options["available_y_columns"]
            group_cols = options["available_group_columns"]
            bar_cols = options.get("bar_columns") or options["available_group_columns"]
            color_cols = options.get("color_columns", group_cols)

            if chart_type == "scatter":
                if not numeric_cols:
                    st.warning("No numeric columns detected. Pick numeric fields or switch chart type.")
                x_column = st.selectbox(
                    "X column",
                    numeric_cols or list(df.columns),
                    index=_safe_index(numeric_cols or list(df.columns), state.get("x_column")),
                    key="scatter_x_select",
                )
                y_options = [c for c in (numeric_cols or list(df.columns)) if c != x_column]
                y_column = st.selectbox(
                    "Y column",
                    y_options or numeric_cols or list(df.columns),
                    index=_safe_index(y_options or numeric_cols or list(df.columns), state.get("y_column")),
                    key="scatter_y_select",
                )
                color_choices = ["None", *color_cols]
                color_default = state.get("color_column") or "None"
                if color_default not in color_choices:
                    color_default = "None"
                color_pick = st.selectbox(
                    "Color by (optional)",
                    color_choices,
                    index=_safe_index(color_choices, color_default),
                    key="scatter_color_select",
                )
                color_column = None if color_pick == "None" else color_pick
                show_trendline = st.toggle("Show trendline", value=True, key="scatter_trend_toggle")
            elif chart_type == "boxplot":
                group_column = st.selectbox(
                    "Group by",
                    group_cols or list(df.columns),
                    index=_safe_index(group_cols or list(df.columns), state.get("group_column")),
                    key="box_group_select",
                )
                y_column = st.selectbox(
                    "Y column",
                    numeric_cols or list(df.columns),
                    index=_safe_index(numeric_cols or list(df.columns), state.get("y_column")),
                    key="box_y_select",
                )
            else:
                x_column = st.selectbox(
                    "Category column",
                    bar_cols or list(df.columns),
                    index=_safe_index(bar_cols or list(df.columns), state.get("x_column")),
                    key="bar_x_select",
                )
                top_n = st.slider("Top categories", min_value=3, max_value=12, value=8, key="bar_top_n")
                sort_desc = st.toggle("Highest count first", value=True, key="bar_sort_desc")

            st.session_state["chart_state"] = align_chart_state_to_dataframe(
                {
                    "chart_type": chart_type,
                    "x_column": x_column,
                    "y_column": y_column,
                    "group_column": group_column,
                    "color_column": color_column,
                },
                df,
            )

    validation_error = _validate_chart_selection(df, chart_type, x_column, y_column, group_column, options)

    with workspace_col:
        chart_title = _chart_spec_title(
            {
                "type": chart_type,
                "x": x_column or group_column,
                "y": y_column,
                "column": x_column,
            }
        )
        st.markdown(f"### {chart_title}")
        st.caption("Zoom, pan, and hover on the chart. Export the full PDF report from the Export Report section.")

        if validation_error:
            st.warning(validation_error)
            return

        try:
            metadata = get_chart_metadata(
                df,
                recommended,
                chart_type,
                x_column=x_column,
                y_column=y_column,
                group_column=group_column,
            )
        except Exception as exc:
            st.warning(f"Could not build chart metadata: {exc}")
            metadata = {"insight": "", "reason": "", "is_recommended": "Custom exploration"}

        preview_tab, insight_tab, data_tab = st.tabs(["Preview", "Insights", "Data"])

        with preview_tab:
            try:
                fig = build_interactive_chart(
                    df,
                    chart_type,
                    x_column=x_column,
                    y_column=y_column,
                    group_column=group_column,
                    color_column=color_column,
                    show_trendline=show_trendline,
                    top_n=top_n,
                    sort_desc=sort_desc,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render this chart. Try different columns or chart type. ({exc})")

        with insight_tab:
            st.markdown(f"**Insight:** {metadata['insight']}")
            with st.container(border=True):
                st.markdown("**Why this chart was recommended**")
                st.write(metadata["reason"])
                st.caption(f"Planner match: {metadata['is_recommended']}")

        with data_tab:
            preview_rows = _chart_data_preview(
                df,
                chart_type,
                x_column,
                y_column,
                group_column,
                limit=15,
            )
            if preview_rows.empty:
                st.info("No rows available for the selected chart columns.")
            else:
                st.dataframe(preview_rows, use_container_width=True, hide_index=True)
                st.caption(f"Showing up to {len(preview_rows)} rows used in this chart view.")


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
        progress = st.progress(0, text="Preparing report...")
        try:
            progress.progress(35, text="Running analysis, charts, and PDF export...")
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
            progress.empty()
            st.success("PDF report generated successfully.")
            st.markdown(f"**Saved to:** `{report_path}`")
            st.caption("Open the Export Report section again or check outputs/reports/ for the latest file.")


def _safe_index(options: list[str], value: str | None) -> int:
    """Return a safe selectbox index for a stored chart state value."""
    if value in options:
        return options.index(value)
    return 0


def _render_welcome_screen() -> None:
    """Show the landing state before a dataset is loaded."""
    st.info("Upload a CSV or try the bundled sample dataset to begin.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Upload your own file**")
        st.caption("Use the file uploader above. Messy headers, missing values, and mixed types are handled safely.")
    with col2:
        st.markdown("**Demo without uploading**")
        if get_sample_csv_bytes() is None:
            st.warning("Sample file not found at data/sample_data.csv")
        elif st.button("Try sample dataset", type="primary", use_container_width=True):
            st.session_state["use_sample_dataset"] = True
            st.rerun()


def _load_uploaded_package(uploaded_file: Any) -> dict[str, Any] | None:
    """Load and enrich a user-uploaded CSV package."""
    file_bytes = uploaded_file.getvalue()
    file_hash = file_fingerprint(file_bytes)
    reset_dataset_session(file_hash)
    try:
        with st.spinner("Analyzing uploaded dataset..."):
            package = load_and_analyze_dataset(file_hash, uploaded_file.name, file_bytes)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return None
    except Exception as exc:
        st.error(f"Could not analyze this file: {exc}")
        return None
    package["source_label"] = f"Uploaded · {uploaded_file.name}"
    return enrich_analysis_package(package, package["df"])


def _render_dataset_workspace(package: dict[str, Any]) -> None:
    """Render the main workspace after a dataset is loaded."""
    render_dataset_warnings(package["warnings"])
    apply_pending_session_updates()

    page = _render_sidebar(package)

    if page == "Overview":
        _render_overview_page(package)
    elif page == "Explore Data":
        _render_explore_page(package)
    elif page == "Chart Builder":
        _render_chart_page(package)
    else:
        _render_export_page(package)


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

    upload_col, demo_col = st.columns([2.2, 1])
    with upload_col:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
    with demo_col:
        st.markdown("**Quick start**")
        if get_sample_csv_bytes() is None:
            st.caption("Sample file unavailable.")
        elif st.button("Try sample dataset", use_container_width=True):
            st.session_state["use_sample_dataset"] = True
            st.rerun()

    package: dict[str, Any] | None = None

    if uploaded_file is not None:
        st.session_state.pop("use_sample_dataset", None)
        package = _load_uploaded_package(uploaded_file)
    elif st.session_state.get("use_sample_dataset"):
        with st.spinner("Loading sample dataset..."):
            package = load_sample_dataset()
        if package is None:
            st.error("Sample dataset could not be loaded.")
            st.session_state.pop("use_sample_dataset", None)
        else:
            reset_dataset_session(file_fingerprint(get_sample_csv_bytes() or b""))
    else:
        _render_welcome_screen()
        return

    if package is None:
        return

    _render_dataset_workspace(package)


if __name__ == "__main__":
    main()
