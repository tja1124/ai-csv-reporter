"""JMP-lite chart workspace helpers for the Streamlit analyst UI."""

from __future__ import annotations

import uuid
from typing import Any, Callable

import pandas as pd
import streamlit as st

from src.streamlit_helpers import align_chart_state_to_dataframe

SHELF_LABELS = {"x": "X", "y": "Y", "group_color": "Group/Color"}


def group_columns_for_shelf(chart_options: dict[str, Any]) -> dict[str, list[str]]:
    """Group dataframe columns into shelf pill categories."""
    all_columns = chart_options.get("all_columns", [])
    numeric = set(chart_options.get("available_y_columns", []))
    categorical = set(chart_options.get("categorical_columns", []))
    datetime_cols = set(chart_options.get("datetime_columns", []))
    grouped: dict[str, list[str]] = {
        "Numeric": [],
        "Categorical": [],
        "Date/Time": [],
        "Other": [],
    }
    for column in all_columns:
        if column in datetime_cols:
            grouped["Date/Time"].append(column)
        elif column in numeric:
            grouped["Numeric"].append(column)
        elif column in categorical:
            grouped["Categorical"].append(column)
        else:
            grouped["Other"].append(column)
    return {label: cols for label, cols in grouped.items() if cols}


def shelf_slot_labels(chart_type: str) -> dict[str, str]:
    """Return human-readable shelf titles for the active chart type."""
    if chart_type == "boxplot":
        return {
            "x": "X variable",
            "y": "Y variable",
            "group_color": "Group by",
            "chart_type": "Chart type",
        }
    if chart_type == "bar":
        return {
            "x": "Category (X)",
            "y": "Y variable",
            "group_color": "Group/Color",
            "chart_type": "Chart type",
        }
    return {
        "x": "X variable",
        "y": "Y variable",
        "group_color": "Group/Color",
        "chart_type": "Chart type",
    }


def init_workspace_session(package: dict[str, Any]) -> None:
    """Ensure workspace session keys exist alongside chart_state."""
    if "saved_charts" not in st.session_state:
        st.session_state["saved_charts"] = []
    if "active_shelf" not in st.session_state:
        st.session_state["active_shelf"] = "x"
    state = st.session_state.setdefault("chart_state", {})
    state.setdefault("show_trendline", True)
    state.setdefault("top_n", 8)
    state.setdefault("sort_desc", True)
    state.setdefault("color_column", None)
    state.setdefault("title_override", "")
    state.setdefault("analyst_note", "")
    state.setdefault("include_in_pdf", False)
    state.setdefault("color_palette", "Default")
    state.setdefault("point_color", "Blue")
    # Version counter: incremented by callbacks (pills, clear, load) so the
    # advanced-dropdown sync logic can detect that chart_state changed externally.
    state.setdefault("_ext_change_count", 0)
    st.session_state["chart_state"] = align_chart_state_to_dataframe(state, package["df"])


def _read_chart_state() -> dict[str, Any]:
    return dict(st.session_state.get("chart_state", {}))


def _write_chart_state(state: dict[str, Any], df: pd.DataFrame) -> None:
    st.session_state["chart_state"] = align_chart_state_to_dataframe(state, df)


def assign_column_to_active_shelf(column: str, df: pd.DataFrame) -> None:
    """Assign a clicked column pill to the currently selected shelf."""
    shelf = st.session_state.get("active_shelf", "x")
    state = _read_chart_state()
    chart_type = state.get("chart_type", "scatter")

    if shelf == "x":
        state["x_column"] = column
    elif shelf == "y":
        state["y_column"] = column
    elif shelf == "group_color":
        if chart_type == "scatter":
            state["color_column"] = column
        elif chart_type == "boxplot":
            state["group_column"] = column
        else:
            state["color_column"] = column

    # Signal that chart_state changed via an external callback so the advanced
    # dropdown widgets can be force-synced on the next render pass.
    state["_ext_change_count"] = state.get("_ext_change_count", 0) + 1
    _write_chart_state(state, df)


def clear_chart_shelf(shelf: str, df: pd.DataFrame) -> None:
    """Clear a shelf assignment."""
    state = _read_chart_state()
    chart_type = state.get("chart_type", "scatter")

    if shelf == "x":
        state["x_column"] = None
    elif shelf == "y":
        state["y_column"] = None
    elif shelf == "group_color":
        if chart_type == "scatter":
            state["color_column"] = None
        elif chart_type == "boxplot":
            state["group_column"] = None
        else:
            state["color_column"] = None

    state["_ext_change_count"] = state.get("_ext_change_count", 0) + 1
    _write_chart_state(state, df)


def chart_state_to_plot_kwargs(state: dict[str, Any]) -> dict[str, Any]:
    """Extract Plotly builder kwargs from a chart state or saved spec."""
    chart_type = state.get("chart_type", "scatter")
    return {
        "x_column": state.get("x_column"),
        "y_column": state.get("y_column"),
        "group_column": state.get("group_column"),
        "color_column": state.get("color_column"),
        "show_trendline": state.get("show_trendline", True),
        "top_n": state.get("top_n", 8),
        "sort_desc": state.get("sort_desc", True),
        "color_palette": state.get("color_palette", "Default"),
        "point_color": state.get("point_color", "Blue"),
    }


def build_saved_chart_entry(state: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the current builder configuration for the gallery."""
    entry = dict(state)
    entry["id"] = uuid.uuid4().hex[:10]
    entry.setdefault("title_override", "")
    entry.setdefault("analyst_note", "")
    entry.setdefault("include_in_pdf", False)
    return entry


def add_chart_to_workspace(df: pd.DataFrame) -> None:
    """Append the current chart configuration to the saved gallery."""
    state = align_chart_state_to_dataframe(_read_chart_state(), df)
    st.session_state["saved_charts"].append(build_saved_chart_entry(state))


def remove_saved_chart(chart_id: str) -> None:
    """Remove a chart from the workspace gallery."""
    st.session_state["saved_charts"] = [
        chart for chart in st.session_state.get("saved_charts", []) if chart.get("id") != chart_id
    ]


def move_saved_chart(chart_id: str, direction: int) -> None:
    """Move a saved chart up (-1) or down (+1) in the gallery."""
    charts: list[dict[str, Any]] = list(st.session_state.get("saved_charts", []))
    index = next((i for i, chart in enumerate(charts) if chart.get("id") == chart_id), None)
    if index is None:
        return
    target = index + direction
    if 0 <= target < len(charts):
        charts[index], charts[target] = charts[target], charts[index]
        st.session_state["saved_charts"] = charts


def load_saved_chart_into_builder(chart_id: str, df: pd.DataFrame) -> None:
    """Copy a saved chart spec back into the active builder."""
    for chart in st.session_state.get("saved_charts", []):
        if chart.get("id") == chart_id:
            new_state = align_chart_state_to_dataframe(dict(chart), df)
            # Preserve and bump the counter so advanced dropdowns re-sync.
            new_state["_ext_change_count"] = (
                st.session_state.get("chart_state", {}).get("_ext_change_count", 0) + 1
            )
            st.session_state["chart_state"] = new_state
            return


def display_title_for_spec(spec: dict[str, Any], title_fn: Callable[[dict[str, Any]], str]) -> str:
    """Return override title or default chart title."""
    override = (spec.get("title_override") or "").strip()
    if override:
        return override
    planner_spec = {
        "type": spec.get("chart_type", "scatter"),
        "x": spec.get("x_column") or spec.get("group_column"),
        "y": spec.get("y_column"),
        "column": spec.get("x_column"),
    }
    return title_fn(planner_spec)


def shelf_display_value(state: dict[str, Any], shelf: str) -> str | None:
    """Return the column name shown on a shelf card."""
    chart_type = state.get("chart_type", "scatter")
    if shelf == "x":
        return state.get("x_column")
    if shelf == "y":
        return state.get("y_column")
    if shelf == "group_color":
        if chart_type == "boxplot":
            return state.get("group_column")
        return state.get("color_column")
    return None


def shelf_is_active(chart_type: str, shelf: str) -> bool:
    """Whether a shelf slot is used for the current chart type."""
    if chart_type == "bar":
        return shelf == "x"
    if chart_type == "boxplot":
        return shelf in {"y", "group_color"}
    return True
