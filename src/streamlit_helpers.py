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
from src.config import OUTPUTS_DIR
from src.data_loader import load_csv, validate_dataframe
from src.format_utils import format_number, is_currency_like_column
from src.insight_generator import generate_report_insights

UPLOAD_DIR = OUTPUTS_DIR / "uploads"
MAX_STORED_UPLOADS = 12

APP_CSS = """
<style>
    .block-container { padding-top: 1.1rem; padding-bottom: 1.5rem; max-width: 1180px; }
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
    div[data-testid="stSidebar"] { background-color: #F8FAFC; }
</style>
"""


def inject_app_styles() -> None:
    """Apply lightweight global styling."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def openai_key_available() -> bool:
    """Return True when an OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def cleanup_old_uploads(upload_dir: Path = UPLOAD_DIR, keep: int = MAX_STORED_UPLOADS) -> None:
    """Remove older uploaded CSV files to keep local/Cloud storage tidy."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    uploads = sorted(upload_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale_file in uploads[keep:]:
        stale_file.unlink(missing_ok=True)


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded CSV for reuse by the existing backend."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / Path(uploaded_file.name).name
    destination.write_bytes(uploaded_file.getbuffer())
    cleanup_old_uploads()
    return destination


def file_fingerprint(file_bytes: bytes) -> str:
    """Build a stable hash for Streamlit cache keys."""
    return hashlib.sha256(file_bytes).hexdigest()


@st.cache_data(show_spinner="Analyzing dataset...")
def load_and_analyze_dataset(file_hash: str, file_name: str, file_bytes: bytes) -> dict[str, Any]:
    """Load, validate, and analyze an uploaded CSV through the existing backend."""
    csv_path = UPLOAD_DIR / Path(file_name).name
    csv_path.write_bytes(file_bytes)
    cleanup_old_uploads()

    df = load_csv(csv_path)
    warnings = validate_dataframe(df)
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

    for column, info in column_info.items():
        missing = missing_values["columns_with_missing"].get(column, {}).get("percentage", 0.0)
        rows.append(
            {
                "Column": column,
                "Type": info["category"],
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


def spec_to_chart_state(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert an auto-selected chart spec into Streamlit chart control values."""
    chart_type = spec["type"]
    state = {
        "chart_type": chart_type,
        "x_column": spec.get("x") or spec.get("column"),
        "y_column": spec.get("y"),
        "group_column": spec.get("x") if chart_type == "boxplot" else None,
    }
    return state
