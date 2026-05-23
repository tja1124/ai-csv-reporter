"""Polished Streamlit analyst workspace for interactive CSV exploration."""

from __future__ import annotations

import hashlib
import html
from typing import Any

import pandas as pd
import streamlit as st

from src.column_utils import assess_grouped_comparison
from src.config import ensure_output_dirs
from src.logger import setup_logging
from src.main import run_report
from src.chart_workspace import (
    SHELF_LABELS,
    add_chart_to_workspace,
    assign_column_to_active_shelf,
    chart_state_to_plot_kwargs,
    clear_chart_shelf,
    display_title_for_spec,
    group_columns_for_shelf,
    init_workspace_session,
    load_saved_chart_into_builder,
    move_saved_chart,
    remove_saved_chart,
    shelf_display_value,
    shelf_is_active,
    shelf_slot_labels,
)
from src.plotly_charts import (
    PALETTE_NAMES,
    POINT_COLOR_NAMES,
    build_interactive_chart,
    get_chart_metadata,
    normalize_palette_name,
    normalize_point_color_name,
    numeric_series_for_chart as _numeric_series_for_chart,
)
from src.streamlit_helpers import (
    SHELF_CHIP_MAX_LEN,
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
    truncate_column_label,
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


def _show_group_palette(chart_type: str, state: dict[str, Any]) -> bool:
    """Whether the group color palette control should appear."""
    if chart_type == "boxplot":
        return True
    if chart_type == "scatter":
        return bool(state.get("color_column"))
    return False


def _show_point_color(chart_type: str, state: dict[str, Any]) -> bool:
    """Whether the ungrouped point color control should appear."""
    if chart_type == "bar":
        return True
    if chart_type == "scatter":
        return not state.get("color_column")
    return False


def _chart_preview_key(state: dict[str, Any]) -> str:
    """Return a stable key for st.plotly_chart that changes whenever chart config changes.

    When Streamlit sees a new key it destroys the previous component and mounts a
    fresh one, guaranteeing a re-render even when the figure JSON is structurally
    similar to the previous frame.
    """
    sig = "|".join(
        str(state.get(k, ""))
        for k in (
            "chart_type", "x_column", "y_column", "group_column",
            "color_column", "color_palette", "point_color",
            "show_trendline", "top_n", "sort_desc",
        )
    )
    return "chart_preview_" + hashlib.md5(sig.encode()).hexdigest()[:10]


def _sync_adv_widgets_if_needed(state: dict[str, Any], chart_type: str) -> None:
    """Push chart_state values into advanced-dropdown widget keys when an external
    change (pill click, shelf clear, gallery load, recommended-chart load) is detected.

    Streamlit widget keys hold their OWN last-selected value independently of
    chart_state.  After a pill click the widget key still holds a stale value; when
    the selectbox renders it returns that stale value and silently overwrites the pill
    assignment.

    We detect external changes via _ext_change_count (incremented by every callback
    that mutates chart_state from outside the widget tree).  When the counter
    advances we force-write only the widget keys that belong to the CURRENT chart
    type, so each selectbox renders — and returns — the correct value.

    Scoping to the current chart type prevents writing a numeric column name into an
    adv_bar_x key whose option list only contains categorical columns, which would
    raise a StreamlitAPIException when the bar-chart expander next renders.

    Widget-driven changes do NOT increment the counter, so user edits via the
    advanced dropdowns are never clobbered.
    """
    current = state.get("_ext_change_count", 0)
    if current == st.session_state.get("_adv_sync_count", -1):
        return  # no external change since last sync

    x_col = state.get("x_column")
    y_col = state.get("y_column")
    group_col = state.get("group_column")
    color_col = state.get("color_column")

    if chart_type == "scatter":
        if x_col is not None:
            st.session_state["adv_scatter_x"] = x_col
        if y_col is not None:
            st.session_state["adv_scatter_y"] = y_col
        # color selectbox uses the string "None" as its sentinel value
        st.session_state["adv_scatter_color"] = color_col if color_col is not None else "None"
    elif chart_type == "boxplot":
        if group_col is not None:
            st.session_state["adv_box_group"] = group_col
        if y_col is not None:
            st.session_state["adv_box_y"] = y_col
    elif chart_type == "bar":
        if x_col is not None:
            st.session_state["adv_bar_x"] = x_col

    st.session_state["_adv_sync_count"] = current


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
                "color_column": None,
                "show_trendline": True,
                "top_n": 8,
                "sort_desc": True,
                "title_override": "",
                "analyst_note": "",
                "include_in_pdf": False,
                "color_palette": "Default",
                "point_color": "Blue",
                # Start at 0; _adv_sync_count defaults to -1 so the first
                # render always syncs the advanced dropdowns.
                "_ext_change_count": 0,
            }
    init_workspace_session(package)


def _render_column_pills(df: pd.DataFrame, chart_options: dict[str, Any]) -> None:
    """Render clickable column pills grouped by detected type."""
    st.markdown('<p class="shelf-target-label">Assign to shelf</p>', unsafe_allow_html=True)
    st.radio(
        "Active shelf",
        options=list(SHELF_LABELS.keys()),
        format_func=lambda key: SHELF_LABELS[key],
        horizontal=True,
        key="active_shelf",
        label_visibility="collapsed",
    )
    st.markdown('<div class="workspace-pills">', unsafe_allow_html=True)

    for group_name, columns in group_columns_for_shelf(chart_options).items():
        st.markdown(f'<p class="pill-group-title">{html.escape(group_name)}</p>', unsafe_allow_html=True)
        pill_cols = st.columns(2, gap="small")
        for index, column in enumerate(columns):
            with pill_cols[index % 2]:
                label = truncate_column_label(column)
                st.button(
                    label,
                    key=f"pill_{group_name}_{index}",
                    width="stretch",
                    help=str(column),
                    on_click=assign_column_to_active_shelf,
                    args=(column, df),
                )

    st.markdown("</div>", unsafe_allow_html=True)


def _render_shelf_row(
    shelf: str,
    label: str,
    value: str | None,
    df: pd.DataFrame,
    *,
    enabled: bool = True,
) -> None:
    """Render a horizontal variable row: label, value chip, clear control."""
    if not enabled:
        st.markdown(
            f'<div class="var-row--disabled">{html.escape(label)} — not used for this chart</div>',
            unsafe_allow_html=True,
        )
        return

    if value:
        display = truncate_column_label(value, SHELF_CHIP_MAX_LEN)
        chip_html = (
            f'<span class="var-chip" title="{html.escape(str(value))}">{html.escape(display)}</span>'
        )
    else:
        chip_html = '<span class="var-chip var-chip--empty">None selected</span>'

    row = st.columns([0.88, 2.22, 0.34], gap="small", vertical_alignment="center")
    row[0].markdown(f'<div class="var-label">{html.escape(label)}</div>', unsafe_allow_html=True)
    row[1].markdown(f'<div class="var-chip-wrap">{chip_html}</div>', unsafe_allow_html=True)
    with row[2]:
        st.markdown('<div class="shelf-clear-slot">', unsafe_allow_html=True)
        st.button(
            "×",
            key=f"clear_shelf_{shelf}",
            disabled=not value,
            help="Clear assignment",
            on_click=clear_chart_shelf,
            args=(shelf, df),
        )


def _render_chart_shelves(df: pd.DataFrame, state: dict[str, Any], options: dict[str, Any]) -> None:
    """Render shelf rows and advanced fallback controls; write final state to session_state."""
    chart_type = state.get("chart_type", "scatter")
    labels = shelf_slot_labels(chart_type)
    supported = [c for c in CHART_TYPES if c in options["supported_chart_types"]] or list(CHART_TYPES)
    palette_default = normalize_palette_name(state.get("color_palette"))
    point_default = normalize_point_color_name(state.get("point_color"))

    st.markdown('<div class="workspace-shelves">', unsafe_allow_html=True)
    st.markdown('<div class="chart-control-bar">', unsafe_allow_html=True)

    chart_type = st.selectbox(
        labels["chart_type"],
        supported,
        index=_safe_index(supported, chart_type),
        key="workspace_chart_type",
    )
    state["chart_type"] = chart_type

    show_trendline = state.get("show_trendline", True)
    top_n = state.get("top_n", 8)
    sort_desc = state.get("sort_desc", True)

    show_group = _show_group_palette(chart_type, state)
    show_point = _show_point_color(chart_type, state)
    show_trend = chart_type == "scatter" and show_point

    if show_group or show_point or show_trend:
        ctrl_cols = st.columns(2, gap="small")
        slot = 0
        if show_group:
            with ctrl_cols[slot % 2]:
                state["color_palette"] = st.selectbox(
                    "Group palette",
                    list(PALETTE_NAMES),
                    index=_safe_index(list(PALETTE_NAMES), palette_default),
                    key="workspace_color_palette",
                    help="Colors for grouped series.",
                )
            slot += 1
        if show_point:
            with ctrl_cols[slot % 2]:
                state["point_color"] = st.selectbox(
                    "Point color",
                    list(POINT_COLOR_NAMES),
                    index=_safe_index(list(POINT_COLOR_NAMES), point_default),
                    key="workspace_point_color",
                    help="Color for ungrouped points or bars.",
                )
            slot += 1
        if show_trend:
            with ctrl_cols[slot % 2]:
                show_trendline = st.toggle("Trendline", value=show_trendline, key="workspace_trend_toggle")
    else:
        state["point_color"] = point_default

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="var-assignments">', unsafe_allow_html=True)
    for shelf in ("x", "y", "group_color"):
        active = shelf_is_active(chart_type, shelf)
        _render_shelf_row(
            shelf,
            labels[shelf],
            shelf_display_value(state, shelf) if active else None,
            df,
            enabled=active,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Sync advanced dropdown widget keys to chart_state if a pill/clear/load
    # callback changed the state since the last render pass.  Pass the current
    # chart_type so only the keys for that chart type are updated — prevents
    # writing a numeric column into the bar chart's categorical dropdown key.
    _sync_adv_widgets_if_needed(state, chart_type)

    with st.expander("More options", expanded=False):
        numeric_cols = options["available_y_columns"]
        group_cols = options["available_group_columns"]
        bar_cols = options.get("bar_columns") or options["available_group_columns"]
        color_cols = options.get("color_columns", group_cols)

        if chart_type == "scatter":
            if not numeric_cols:
                st.warning("No numeric columns detected.")
            state["x_column"] = st.selectbox(
                "X column",
                numeric_cols or list(df.columns),
                index=_safe_index(numeric_cols or list(df.columns), state.get("x_column")),
                key="adv_scatter_x",
            )
            y_options = [c for c in (numeric_cols or list(df.columns)) if c != state.get("x_column")]
            state["y_column"] = st.selectbox(
                "Y column",
                y_options or numeric_cols or list(df.columns),
                index=_safe_index(y_options or numeric_cols or list(df.columns), state.get("y_column")),
                key="adv_scatter_y",
            )
            color_choices = ["None", *color_cols]
            color_default = state.get("color_column") or "None"
            if color_default not in color_choices:
                color_default = "None"
            color_pick = st.selectbox(
                "Color by",
                color_choices,
                index=_safe_index(color_choices, color_default),
                key="adv_scatter_color",
            )
            state["color_column"] = None if color_pick == "None" else color_pick
        elif chart_type == "boxplot":
            state["group_column"] = st.selectbox(
                "Group by",
                group_cols or list(df.columns),
                index=_safe_index(group_cols or list(df.columns), state.get("group_column")),
                key="adv_box_group",
            )
            state["y_column"] = st.selectbox(
                "Y column",
                numeric_cols or list(df.columns),
                index=_safe_index(numeric_cols or list(df.columns), state.get("y_column")),
                key="adv_box_y",
            )
        else:
            state["x_column"] = st.selectbox(
                "Category column",
                bar_cols or list(df.columns),
                index=_safe_index(bar_cols or list(df.columns), state.get("x_column")),
                key="adv_bar_x",
            )
            top_n = st.slider("Top categories", 3, 12, value=state.get("top_n", 8), key="adv_bar_top_n")
            sort_desc = st.toggle("Highest count first", value=state.get("sort_desc", True), key="adv_bar_sort")

    if not _show_point_color(chart_type, state):
        state["point_color"] = point_default
    if not _show_group_palette(chart_type, state):
        state["color_palette"] = palette_default

    color_palette = normalize_palette_name(state.get("color_palette"))
    point_color = normalize_point_color_name(state.get("point_color"))

    state["show_trendline"] = show_trendline
    state["top_n"] = top_n
    state["sort_desc"] = sort_desc
    state["color_palette"] = color_palette
    state["point_color"] = point_color
    st.session_state["chart_state"] = align_chart_state_to_dataframe(state, df)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_save_export_section(
    state: dict[str, Any],
    df: pd.DataFrame,
    *,
    validation_error: str | None,
) -> None:
    """Report prep placeholders and add-to-gallery action."""
    _render_report_prep_controls(state)
    if validation_error:
        st.button(
            "Add chart to workspace",
            type="primary",
            disabled=True,
            help=validation_error,
            width="stretch",
        )
    else:
        if st.button("Add chart to workspace", type="primary", width="stretch"):
            add_chart_to_workspace(df)
            st.toast("Chart added to workspace gallery.")
            st.rerun()


def _render_report_prep_controls(state: dict[str, Any]) -> None:
    """Placeholder report-prep fields (not wired to custom PDF export yet)."""
    state["include_in_pdf"] = st.checkbox(
        "Include in PDF",
        value=state.get("include_in_pdf", False),
        disabled=True,
        help="Coming soon: export selected workspace charts into the PDF report.",
    )
    state["title_override"] = st.text_input(
        "Chart title override",
        value=state.get("title_override", ""),
        placeholder="Optional title for saved charts",
        key="workspace_title_override",
    )
    state["analyst_note"] = st.text_area(
        "Analyst note",
        value=state.get("analyst_note", ""),
        placeholder="Notes stored with saved charts (PDF wiring later)",
        height=68,
        key="workspace_analyst_note",
    )


def _render_saved_chart_gallery(
    package: dict[str, Any],
) -> None:
    """Render saved workspace charts below the builder."""
    df = package["df"]
    recommended = package.get("recommended_charts", [])
    saved = st.session_state.get("saved_charts", [])

    st.markdown("---")
    st.subheader("Saved chart gallery")
    st.caption("Charts saved here stay in your session until you load a new dataset.")

    if not saved:
        st.info("No saved charts yet. Configure the workspace and use **Add chart to workspace**.")
        return

    for index, spec in enumerate(saved):
        title = display_title_for_spec(spec, _chart_spec_title)
        with st.container(border=True):
            header = st.columns([3, 1, 1, 1, 1, 1])
            header[0].markdown(f"**{title}**")
            header[1].button(
                "Load",
                key=f"gallery_load_{spec['id']}",
                on_click=load_saved_chart_into_builder,
                args=(spec["id"], df),
            )
            header[2].button(
                "Up",
                key=f"gallery_up_{spec['id']}",
                disabled=index == 0,
                on_click=move_saved_chart,
                args=(spec["id"], -1),
            )
            header[3].button(
                "Down",
                key=f"gallery_down_{spec['id']}",
                disabled=index == len(saved) - 1,
                on_click=move_saved_chart,
                args=(spec["id"], 1),
            )
            header[4].button(
                "Remove",
                key=f"gallery_remove_{spec['id']}",
                on_click=remove_saved_chart,
                args=(spec["id"],),
            )
            if spec.get("analyst_note"):
                header[5].caption("Has note")

            plot_kwargs = chart_state_to_plot_kwargs(spec)
            chart_type = spec.get("chart_type", "scatter")
            err = _validate_chart_selection(
                df,
                chart_type,
                plot_kwargs["x_column"],
                plot_kwargs["y_column"],
                plot_kwargs["group_column"],
                package["chart_options"],
            )
            if err:
                st.warning(err)
                continue
            try:
                fig = build_interactive_chart(df, chart_type, **plot_kwargs)
                st.plotly_chart(fig, width="stretch", key=f"gallery_plot_{spec['id']}")
            except Exception as exc:
                st.warning(f"Could not render saved chart: {exc}")
                continue

            try:
                meta = get_chart_metadata(
                    df,
                    recommended,
                    chart_type,
                    x_column=plot_kwargs["x_column"],
                    y_column=plot_kwargs["y_column"],
                    group_column=plot_kwargs["group_column"],
                )
                st.caption(meta["insight"])
            except Exception:
                pass
            if spec.get("analyst_note"):
                st.markdown(f"*Note:* {spec['analyst_note']}")
            if spec.get("include_in_pdf"):
                st.caption("Marked for PDF (export not wired yet)")


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
        st.info("No automatic chart recommendations for this dataset. Open **Chart Builder** (Visual Analytics Workspace) to explore columns manually.")


def _render_explore_page(package: dict[str, Any]) -> None:
    """Render searchable dataset exploration tools."""
    df = package["df"]
    column_info = package["column_info"]

    st.subheader("Column Summary")
    summary = build_column_summary(column_info, df)
    if summary.empty:
        st.info("No columns available to summarize.")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)

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
    st.dataframe(preview, width="stretch", hide_index=True)
    if preview.empty:
        st.caption("No rows match the current filters.")
    else:
        st.caption(f"Showing {len(preview):,} of {len(df):,} rows")


def _render_chart_page(package: dict[str, Any]) -> None:
    """Render the JMP-lite interactive chart workspace."""
    df = package["df"]
    options = package["chart_options"]
    recommended = package.get("recommended_charts", [])
    _init_chart_state(package)
    state = st.session_state["chart_state"]

    st.subheader("Visual Analytics Workspace")
    st.markdown(
        '<p class="workspace-hint">Assign columns on the left, preview on the right, '
        "and save views to the gallery below.</p>",
        unsafe_allow_html=True,
    )

    left_panel, right_panel = st.columns([31, 69], gap="medium")

    with left_panel:
        st.markdown('<div class="workspace-left">', unsafe_allow_html=True)

        st.markdown('<p class="workspace-panel-title">Dataset columns</p>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-panel-box">', unsafe_allow_html=True)
        _render_column_pills(df, options)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<p class="workspace-panel-title">Chart setup</p>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-panel-box">', unsafe_allow_html=True)
        _render_chart_shelves(df, state, options)
        # _render_chart_shelves writes the definitive aligned state back to
        # session_state at its own end; re-read once here so save/export and
        # the right panel both see the same single source of truth.
        state = st.session_state["chart_state"]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<p class="workspace-panel-title">Save & export</p>', unsafe_allow_html=True)
        st.markdown('<div class="workspace-panel-box">', unsafe_allow_html=True)
        validation_error = _validate_chart_selection(
            df,
            state.get("chart_type", "scatter"),
            state.get("x_column"),
            state.get("y_column"),
            state.get("group_column"),
            options,
        )
        _render_save_export_section(state, df, validation_error=validation_error)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Single authoritative read after the entire left panel has settled.
    state = st.session_state["chart_state"]
    chart_type = state.get("chart_type", "scatter")
    color_palette = normalize_palette_name(state.get("color_palette", "Default"))
    point_color = normalize_point_color_name(state.get("point_color", "Blue"))
    show_trendline = state.get("show_trendline", True)
    top_n = state.get("top_n", 8)
    sort_desc = state.get("sort_desc", True)
    x_column = state.get("x_column")
    y_column = state.get("y_column")
    group_column = state.get("group_column")
    color_column = state.get("color_column")
    validation_error = _validate_chart_selection(df, chart_type, x_column, y_column, group_column, options)

    with right_panel:
        st.markdown('<div class="workspace-main">', unsafe_allow_html=True)
        chart_title = display_title_for_spec({**state, "chart_type": chart_type}, _chart_spec_title)
        st.markdown(f"### {chart_title}")
        if _show_point_color(chart_type, state) and not color_column:
            color_note = f"Point color: **{point_color}**"
        elif _show_group_palette(chart_type, state):
            color_note = f"Group palette: **{color_palette}**"
        else:
            color_note = "Configure variables on the left to preview"
        st.caption(f"{color_note} · Zoom, pan, hover to inspect.")

        if validation_error:
            st.warning(validation_error)
        else:
            metadata: dict[str, str] = {
                "insight": "",
                "reason": "",
                "is_recommended": "Custom exploration",
            }
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

            preview_tab, insight_tab, data_tab = st.tabs(["Preview", "Insights", "Data"])

            with preview_tab:
                st.markdown('<div class="preview-chart-wrap">', unsafe_allow_html=True)
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
                        color_palette=color_palette,
                        point_color=point_color,
                    )
                    st.plotly_chart(fig, width="stretch", key=_chart_preview_key(state))
                except Exception as exc:
                    st.warning(f"Could not render this chart. Try different columns or chart type. ({exc})")
                st.markdown("</div>", unsafe_allow_html=True)

            with insight_tab:
                st.markdown(f"**Insight:** {metadata['insight']}")
                with st.container(border=True):
                    st.markdown("**Why this chart was recommended**")
                    st.write(metadata["reason"])
                    st.caption(f"Planner match: {metadata['is_recommended']}")
                if state.get("analyst_note"):
                    st.markdown(f"**Analyst note:** {state['analyst_note']}")

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
                    st.dataframe(preview_rows, width="stretch", hide_index=True)
                    st.caption(f"Showing up to {len(preview_rows)} rows used in this chart view.")

        st.markdown("</div>", unsafe_allow_html=True)

    _render_saved_chart_gallery(package)


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

    if st.button("Generate PDF Report", type="primary", width="stretch"):
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
        elif st.button("Try sample dataset", type="primary", width="stretch"):
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
        elif st.button("Try sample dataset", width="stretch"):
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
