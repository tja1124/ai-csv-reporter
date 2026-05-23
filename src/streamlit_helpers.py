"""Shared helpers for the Streamlit analyst workspace."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.analyzer import (
    analyze_columns,
    get_categorical_summary,
    get_dataset_overview,
    get_missing_values,
)
from src.config import PROJECT_ROOT, UPLOADS_DIR, ensure_output_dirs
from src.data_loader import load_csv, sanitize_dataframe_columns, validate_dataframe
from src.format_utils import format_number, is_currency_like_column
from src.insight_generator import generate_report_insights

PILL_LABEL_MAX_LEN = 18
SHELF_CHIP_MAX_LEN = 28

UPLOAD_DIR = UPLOADS_DIR
MAX_STORED_UPLOADS = 12
SAMPLE_CSV_PATH = PROJECT_ROOT / "data" / "sample_data.csv"

APP_CSS = """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1.25rem; max-width: 1280px; }
    .app-header {
        background: linear-gradient(135deg, #2E5090 0%, #5B8DBE 100%);
        color: white; padding: 1.1rem 1.25rem; border-radius: 10px;
        margin-bottom: 0.75rem; box-shadow: 0 2px 8px rgba(46, 80, 144, 0.15);
    }
    .app-header h1 { color: white !important; margin: 0; font-size: 1.55rem; }
    .app-header p { color: rgba(255,255,255,0.92); margin: 0.25rem 0 0 0; font-size: 0.92rem; }
    .kpi-card {
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 0.85rem 1rem; min-height: 96px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .kpi-label { color: #64748B; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { color: #2E5090; font-size: 1.45rem; font-weight: 700; line-height: 1.2; margin-top: 0.15rem; }
    .kpi-help { color: #64748B; font-size: 0.82rem; margin-top: 0.35rem; }
    .section-card {
        background: white; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 1rem 1.1rem; margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .workspace-panel-title {
        color: #1E293B; font-size: 0.88rem; font-weight: 700;
        margin: 0.35rem 0 0.3rem 0; letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .workspace-panel-box {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 0.75rem 0.85rem 0.8rem 0.85rem; margin-bottom: 0.65rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .workspace-left {
        padding-right: 0.35rem;
    }
    div.workspace-left div[data-testid="stVerticalBlock"] > div { gap: 0.2rem !important; }
    div.workspace-left div[data-testid="stRadio"] { margin-bottom: 0.15rem; }
    div.workspace-left div[data-testid="stExpander"] { margin-top: 0.15rem; margin-bottom: 0.1rem; }
    div.workspace-left div[data-testid="stSelectbox"] { margin-bottom: 0.1rem; }
    .workspace-main {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        padding: 0.7rem 0.85rem 0.75rem 0.85rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        min-width: 0;
    }
    .workspace-main h3 { margin: 0 0 0.15rem 0; font-size: 1.12rem; line-height: 1.25; }
    .workspace-main div[data-testid="stCaptionContainer"] { margin-bottom: 0.25rem; }
    .workspace-main [data-baseweb="tab-list"] {
        gap: 0.12rem; margin-bottom: 0; min-height: 2.1rem;
    }
    .workspace-main [data-baseweb="tab-panel"] { padding-top: 0.1rem; padding-bottom: 0; }
    .preview-chart-wrap {
        background: #FAFBFC; border: 1px solid #EEF2F6; border-radius: 10px;
        padding: 0.15rem 0.1rem 0.05rem 0.1rem; margin-top: 0.1rem;
        overflow: hidden;
    }
    .preview-chart-wrap div[data-testid="stPlotlyChart"] { margin: 0 !important; padding: 0 !important; }
    .workspace-hint { color: #64748B; font-size: 0.8rem; margin: 0 0 0.45rem 0; }
    .shelf-target-label {
        color: #64748B; font-size: 0.72rem; font-weight: 600;
        margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.05em;
    }
    div.workspace-pills { margin-top: 0.1rem; }
    div.workspace-pills div[data-testid="column"] {
        padding: 0 0.18rem 0.38rem 0.18rem; min-width: 0 !important;
    }
    div.workspace-pills div[data-testid="stButton"] > button {
        min-height: 2.35rem !important; max-height: 2.35rem !important; height: 2.35rem !important;
        padding: 0 0.55rem !important; font-size: 0.78rem !important;
        line-height: 2.35rem !important; white-space: nowrap !important;
        overflow: hidden !important; text-overflow: ellipsis !important;
        word-break: keep-all !important; overflow-wrap: normal !important;
        display: block !important; width: 100% !important; max-width: 100% !important;
        border-radius: 8px !important; border: 1px solid #E2E8F0 !important;
        background: #F8FAFC !important; color: #1E293B !important;
    }
    div.workspace-pills div[data-testid="stButton"] > button:hover {
        border-color: #94A3B8 !important; background: #EFF6FF !important;
    }
    .pill-group-title {
        color: #64748B; font-size: 0.72rem; font-weight: 600;
        margin: 0.55rem 0 0.28rem 0; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .pill-group-title:first-of-type { margin-top: 0.15rem; }
    div.chart-control-bar { margin: 0 0 0.35rem 0; }
    div.chart-control-bar div[data-testid="stSelectbox"] label,
    div.chart-control-bar div[data-testid="stToggle"] label {
        font-size: 0.76rem !important; margin-bottom: 0.1rem !important;
    }
    div.var-assignments { margin: 0.15rem 0 0.25rem 0; }
    div.var-assignments div[data-testid="column"] { min-width: 0; }
    .var-label {
        color: #64748B; font-size: 0.76rem; font-weight: 600;
        line-height: 2rem; white-space: nowrap;
    }
    .var-chip-wrap { min-width: 0; display: flex; align-items: center; height: 2rem; }
    .var-chip {
        display: inline-block; max-width: 100%; padding: 0.28rem 0.55rem;
        background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 999px;
        color: #1E3A5F; font-size: 0.8rem; font-weight: 600; line-height: 1.2;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .var-chip--empty {
        background: #F8FAFC; border: 1px dashed #CBD5E1; color: #94A3B8;
        font-style: italic; font-weight: 500;
    }
    .var-row--disabled { color: #94A3B8; font-size: 0.78rem; margin: 0.2rem 0; }
    div.shelf-clear-slot div[data-testid="stButton"] {
        display: flex; justify-content: flex-end;
    }
    div.shelf-clear-slot {
        display: flex; align-items: center; justify-content: flex-end;
        height: 2rem; min-width: 2.4rem;
    }
    div.shelf-clear-slot div[data-testid="stButton"] {
        width: 2.25rem !important; min-width: 2.25rem !important; flex: 0 0 2.25rem !important;
    }
    div.shelf-clear-slot div[data-testid="stButton"] > button {
        min-width: 2.25rem !important; max-width: 2.25rem !important; width: 2.25rem !important;
        min-height: 2rem !important; max-height: 2rem !important; height: 2rem !important;
        padding: 0 !important; margin: 0 !important;
        font-size: 1.2rem !important; line-height: 1 !important;
        white-space: nowrap !important; flex-shrink: 0 !important;
        border-radius: 8px !important; overflow: hidden !important;
        text-overflow: clip !important;
    }
    div.shelf-clear-slot div[data-testid="stButton"] > button p {
        white-space: nowrap !important; line-height: 1 !important; margin: 0 !important;
    }
    div.workspace-shelves details {
        margin-top: 0.35rem; border: none;
    }
    div.workspace-shelves details summary {
        font-size: 0.78rem !important; color: #64748B !important;
        padding: 0.2rem 0 !important; font-weight: 500 !important;
    }
    div.workspace-shelves details summary span {
        font-size: 0.78rem !important;
    }
    div.workspace-shelves hr {
        margin: 0.45rem 0 0.4rem 0; border: none; border-top: 1px solid #EEF2F6;
    }
    div[data-testid="stSidebar"] { background-color: #F8FAFC; }
</style>
"""


def inject_app_styles() -> None:
    """Apply lightweight global styling."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def truncate_column_label(name: str, max_len: int = PILL_LABEL_MAX_LEN) -> str:
    """Shorten long column names for pills and shelf cards."""
    text = str(name).strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}…"


def openai_key_available() -> bool:
    """Return True when an OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def cleanup_old_uploads(upload_dir: Path = UPLOAD_DIR, keep: int = MAX_STORED_UPLOADS) -> None:
    """Remove older uploaded CSV files to keep local/Cloud storage tidy."""
    ensure_output_dirs()
    uploads = sorted(upload_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale_file in uploads[keep:]:
        stale_file.unlink(missing_ok=True)


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded CSV for reuse by the existing backend."""
    ensure_output_dirs()
    destination = UPLOAD_DIR / Path(uploaded_file.name).name
    destination.write_bytes(uploaded_file.getbuffer())
    cleanup_old_uploads()
    return destination


def get_sample_csv_bytes() -> bytes | None:
    """Return bundled sample CSV bytes for demo mode, or None if missing."""
    if not SAMPLE_CSV_PATH.is_file():
        return None
    return SAMPLE_CSV_PATH.read_bytes()


def enrich_analysis_package(package: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Attach chart recommendations and UI column options to an analysis package."""
    from src.chart_generator import get_recommended_charts

    package["df"] = df
    try:
        package["recommended_charts"] = get_recommended_charts(df)
    except Exception:
        package["recommended_charts"] = []
    package["chart_options"] = get_ui_chart_column_options(df)
    return package


def reset_dataset_session(file_hash: str) -> None:
    """Clear chart/navigation session keys when a new dataset is loaded."""
    previous = st.session_state.get("dataset_hash")
    if previous == file_hash:
        return
    st.session_state["dataset_hash"] = file_hash
    for key in (
        "chart_state",
        "pending_chart_state",
        "pending_nav_page",
        "saved_charts",
        "active_shelf",
        # Reset the advanced-dropdown sync counter so the first render on the new
        # dataset always syncs the widget keys — prevents stale column names from
        # the previous file appearing in the new file's dropdowns (which would raise
        # a StreamlitAPIException if the old name is not in the new options list).
        "_adv_sync_count",
    ):
        st.session_state.pop(key, None)


def render_dataset_warnings(warnings: list[str]) -> None:
    """Show validation warnings once per run."""
    if not warnings:
        return
    with st.expander(f"Data notes ({len(warnings)})", expanded=len(warnings) <= 2):
        for warning in warnings:
            st.warning(warning)


def file_fingerprint(file_bytes: bytes) -> str:
    """Build a stable hash for Streamlit cache keys."""
    return hashlib.sha256(file_bytes).hexdigest()


def load_sample_dataset() -> dict[str, Any] | None:
    """Load and analyze the bundled sample dataset for demo mode."""
    sample_bytes = get_sample_csv_bytes()
    if sample_bytes is None:
        return None
    file_hash = file_fingerprint(sample_bytes)
    package = load_and_analyze_dataset(file_hash, SAMPLE_CSV_PATH.name, sample_bytes)
    package["source_label"] = "Sample dataset"
    return enrich_analysis_package(package, package["df"])


@st.cache_data(show_spinner="Analyzing dataset...")
def load_and_analyze_dataset(file_hash: str, file_name: str, file_bytes: bytes) -> dict[str, Any]:
    """Load, validate, and analyze an uploaded CSV through the existing backend."""
    ensure_output_dirs()
    csv_path = UPLOAD_DIR / Path(file_name).name
    csv_path.write_bytes(file_bytes)
    cleanup_old_uploads()

    df = load_csv(csv_path)
    sanitize_warnings: list[str] = []
    df, sanitize_warnings = sanitize_dataframe_columns(df)
    warnings = sanitize_warnings + validate_dataframe(df)
    overview = get_dataset_overview(df)
    missing_values = get_missing_values(df)
    categorical_summary = get_categorical_summary(df)
    column_info = analyze_columns(df)
    report_insights = generate_report_insights(
        df=df,
        overview=overview,
        missing_values=missing_values,
        categorical_summary=categorical_summary,
        validation_warnings=warnings,
    )

    return {
        "df": df,
        "csv_path": str(csv_path),
        "warnings": warnings,
        "overview": overview,
        "missing_values": missing_values,
        "column_info": column_info,
        "report_insights": report_insights,
        "source_label": file_name,
    }


def missing_percentage(missing_values: dict[str, Any], overview: dict[str, Any]) -> float:
    """Calculate overall missing-value percentage."""
    total_cells = max(overview["row_count"] * overview["column_count"], 1)
    return round((missing_values["total_missing"] / total_cells) * 100, 1)


def quality_trend_label(score: int, rating: str) -> str:
    """Return a compact quality indicator for KPI cards."""
    if score >= 90:
        return f"▲ {rating}"
    if score >= 75:
        return f"● {rating}"
    return f"▼ {rating}"


def sample_size_label(row_count: int) -> str:
    """Return a sample-size caution label when appropriate."""
    if row_count < 30:
        return "Small sample — directional"
    if row_count < 100:
        return "Moderate sample"
    return "Adequate sample"


def render_kpi_dashboard(package: dict[str, Any]) -> None:
    """Render polished KPI cards."""
    overview = package["overview"]
    missing_values = package["missing_values"]
    quality = package["report_insights"]["quality_score"]
    missing_pct = missing_percentage(missing_values, overview)
    chart_count = len(package.get("recommended_charts", []))

    cards = [
        ("Rows", f"{overview['row_count']:,}", sample_size_label(overview["row_count"])),
        ("Columns", str(overview["column_count"]), "Fields in dataset"),
        ("Quality", f"{quality['score']}/100", quality_trend_label(quality["score"], quality["rating"])),
        ("Missing", f"{missing_pct}%", "Lower is better"),
        ("Charts", str(chart_count), "Auto-recommended views"),
    ]

    columns = st.columns(len(cards))
    for column, (label, value, help_text) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-help">{help_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_column_summary(column_info: dict[str, dict[str, Any]], df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact column summary table for exploration."""
    rows: list[dict[str, str]] = []
    missing_values = get_missing_values(df)
    ui_options = get_ui_chart_column_options(df)

    for column, info in column_info.items():
        if column not in df.columns:
            continue
        missing = missing_values["columns_with_missing"].get(column, {}).get("percentage", 0.0)
        if column in ui_options["available_y_columns"]:
            role = "numeric"
        elif column in ui_options["datetime_columns"]:
            role = "datetime"
        else:
            role = info["category"]
        rows.append(
            {
                "Column": column,
                "Type": role,
                "Unique": str(info["unique_values"]),
                "Missing %": f"{missing:.1f}%",
                "Format": "Currency-like" if is_currency_like_column(column) else "Standard",
            }
        )
    return pd.DataFrame(rows)


def filter_preview_dataframe(
    df: pd.DataFrame,
    *,
    search_text: str,
    selected_columns: list[str],
    row_limit: int,
    sort_column: str | None,
    sort_ascending: bool,
) -> pd.DataFrame:
    """Return a filtered, sortable preview dataframe."""
    preview = df.copy()

    if selected_columns:
        preview = preview[selected_columns]

    if search_text.strip():
        mask = preview.astype(str).apply(
            lambda row: row.str.contains(search_text.strip(), case=False, na=False).any(),
            axis=1,
        )
        preview = preview[mask]

    if sort_column and sort_column in preview.columns:
        preview = preview.sort_values(by=sort_column, ascending=sort_ascending, na_position="last")

    return preview.head(row_limit)


def render_app_header() -> None:
    """Render the top application header."""
    st.markdown(
        """
        <div class="app-header">
            <h1>AI CSV Reporter · Analyst Workspace</h1>
            <p>Explore datasets, build focused charts, and export the same PDF analytics brief used by the CLI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_pending_session_updates() -> None:
    """Apply navigation/chart updates queued before widgets render (avoids StreamlitAPIException)."""
    if "pending_chart_state" in st.session_state:
        st.session_state["chart_state"] = st.session_state.pop("pending_chart_state")
    if "pending_nav_page" in st.session_state:
        st.session_state["nav_page"] = st.session_state.pop("pending_nav_page")


def queue_recommended_chart(spec: dict[str, Any], df: pd.DataFrame) -> None:
    """Queue a recommended chart for the next rerun without mutating widget-bound session keys."""
    new_state = spec_to_chart_state(spec, df)
    # Bump the external-change counter so _sync_adv_widgets_if_needed fires on the
    # next render, forcing the advanced dropdowns to display the new column values.
    # Reading the current counter (rather than always writing 1) handles the edge
    # case where the counter is already 0 after the very first render pass.
    new_state["_ext_change_count"] = (
        st.session_state.get("chart_state", {}).get("_ext_change_count", 0) + 1
    )
    st.session_state["pending_chart_state"] = new_state
    st.session_state["pending_nav_page"] = "Chart Builder"


def _series_numeric_ratio(series: pd.Series) -> float:
    """Return the share of non-null values that parse as numeric."""
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    coerced = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return float(coerced.notna().sum() / len(non_null))


def _is_ui_numeric_column(df: pd.DataFrame, column: str, *, min_unique: int = 2) -> bool:
    """Detect numeric columns for the UI, including mostly-numeric object columns."""
    if column not in df.columns:
        return False
    series = df[column]
    if series.isnull().all():
        return False
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) >= min_unique
    if pd.api.types.is_bool_dtype(series):
        return series.nunique(dropna=True) >= min_unique
    return _series_numeric_ratio(series) >= 0.6 and series.nunique(dropna=True) >= min_unique


def _is_ui_datetime_column(series: pd.Series) -> bool:
    """Detect datetime columns, including common string date formats."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        parsed = pd.to_datetime(series, errors="coerce", utc=False)
        return parsed.notna().sum() / max(series.notna().sum(), 1) >= 0.6
    return False


def get_ui_chart_column_options(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build chart dropdown options from the loaded dataframe (broader than PDF auto-chart rules).

    Uses actual column names in the dataframe so every usable field can appear in Chart Builder.
    """
    all_columns = [str(column) for column in df.columns]
    numeric_cols: list[str] = []
    datetime_cols: list[str] = []
    categorical_cols: list[str] = []

    for column in all_columns:
        if column not in df.columns:
            continue
        series = df[column]
        if _is_ui_datetime_column(series):
            datetime_cols.append(column)
            continue
        if _is_ui_numeric_column(df, column):
            numeric_cols.append(column)
        elif not series.isnull().all():
            categorical_cols.append(column)

    group_cols = [
        column
        for column in categorical_cols
        if 2 <= df[column].nunique(dropna=True) <= 50
    ]
    bar_cols = [
        column
        for column in categorical_cols
        if 1 <= df[column].nunique(dropna=True) <= 30
    ]

    supported: list[str] = []
    if len(numeric_cols) >= 2:
        supported.append("scatter")
    if numeric_cols and group_cols:
        supported.append("boxplot")
    if bar_cols or categorical_cols:
        supported.append("bar")

    x_columns = list(dict.fromkeys([*numeric_cols, *group_cols, *bar_cols, *categorical_cols]))
    color_cols = group_cols or [column for column in categorical_cols if df[column].nunique(dropna=True) <= 20]

    return {
        "all_columns": all_columns,
        "available_x_columns": x_columns or all_columns,
        "available_y_columns": numeric_cols or all_columns,
        "available_group_columns": group_cols or categorical_cols[:25] or all_columns,
        "color_columns": color_cols,
        "datetime_columns": datetime_cols,
        "categorical_columns": categorical_cols,
        "bar_columns": bar_cols or categorical_cols[:25] or all_columns,
        "supported_chart_types": supported or ["bar"],
    }


def align_chart_state_to_dataframe(state: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Ensure chart state columns exist in the current dataframe."""
    all_columns = list(df.columns)
    aligned = dict(state)
    for key in ("x_column", "y_column", "group_column", "color_column"):
        value = aligned.get(key)
        if value and value not in all_columns:
            aligned[key] = None
    if aligned.get("chart_type") == "scatter":
        if aligned.get("x_column") not in all_columns or aligned.get("y_column") not in all_columns:
            numeric = get_ui_chart_column_options(df)["available_y_columns"]
            aligned["x_column"] = numeric[0] if numeric else None
            aligned["y_column"] = numeric[1] if len(numeric) > 1 else aligned.get("x_column")
    return aligned


def spec_to_chart_state(spec: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Convert an auto-selected chart spec into Streamlit chart control values."""
    chart_type = spec["type"]
    state = {
        "chart_type": chart_type,
        "x_column": spec.get("x") or spec.get("column"),
        "y_column": spec.get("y"),
        "group_column": spec.get("x") if chart_type == "boxplot" else None,
        "color_column": None,
        "show_trendline": True,
        "top_n": 8,
        "sort_desc": True,
    }
    if df is not None:
        return align_chart_state_to_dataframe(state, df)
    return state
