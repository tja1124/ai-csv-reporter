"""Generate business-oriented charts from dataset analysis."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.column_utils import (
    assess_grouped_comparison,
    get_groupable_categorical_columns,
    is_id_like_column,
    is_small_sample,
)
from src.config import (
    CATEGORICAL_TOP_N,
    COLORS,
    FIGURE_BOXPLOT_HEIGHT,
    FIGURE_COMPACT_HEIGHT,
    FIGURE_DPI,
    FIGURE_HEIGHT,
    FIGURE_SAVE_PAD,
    FIGURE_WIDTH,
    HISTOGRAM_BINS,
    MAX_BAR_CHARTS,
    MAX_BOXPLOTS,
    MAX_HISTOGRAMS,
    MAX_SCATTER_PLOTS,
    MAX_TOTAL_CHARTS,
    MAX_UNIQUE_FOR_BAR_CHART,
    MIN_NUMERIC_VARIATION,
    MIN_ROWS_FOR_HISTOGRAM,
    MIN_SCATTER_CORRELATION,
    MIN_UNIQUE_FOR_HISTOGRAM,
)
from src.format_utils import (
    apply_axis_format,
    apply_figure_margins,
    format_correlation,
    format_number,
    format_value_range,
)
from src.logger import get_logger

logger = get_logger()


def _sanitize_filename(name: str) -> str:
    """Convert a column name into a safe filename."""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def _apply_chart_style() -> None:
    """Apply a consistent, presentation-ready visual style."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.titleweight": "600",
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.labelweight": "500",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "savefig.facecolor": COLORS["background"],
            "savefig.edgecolor": COLORS["background"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": FIGURE_SAVE_PAD,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    """Apply finishing touches to a single chart axis."""
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["muted"], length=0)


def _save_chart(fig: plt.Figure, chart_file: Path, *, bottom: float = 0.14) -> None:
    """Save a chart with tight cropping and balanced margins."""
    apply_figure_margins(fig, bottom=bottom)
    try:
        fig.tight_layout(pad=0.45)
    except Exception:
        pass
    fig.savefig(
        chart_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        pad_inches=FIGURE_SAVE_PAD,
    )
    plt.close(fig)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return usable numeric columns."""
    columns: list[str] = []
    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()
        if series.empty or is_id_like_column(column, df[column], len(df), for_numeric=True):
            continue
        if series.nunique() < MIN_NUMERIC_VARIATION:
            continue
        columns.append(column)
    return columns


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return low-cardinality categorical columns suitable for grouping."""
    return get_groupable_categorical_columns(df)


def _bar_chart_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns suitable for distribution bar charts."""
    columns: list[str] = []
    for column in get_groupable_categorical_columns(df):
        if df[column].nunique(dropna=True) <= MAX_UNIQUE_FOR_BAR_CHART:
            columns.append(column)
    return columns


def _correlation_pairs(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict[str, Any]]:
    """Rank numeric column pairs by absolute correlation."""
    if len(numeric_cols) < 2:
        return []

    correlation = df[numeric_cols].corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []

    for index, col_x in enumerate(numeric_cols):
        for col_y in numeric_cols[index + 1 :]:
            value = correlation.loc[col_x, col_y]
            if pd.isna(value):
                continue
            pairs.append(
                {
                    "x": col_x,
                    "y": col_y,
                    "correlation": float(value),
                    "abs_correlation": abs(float(value)),
                }
            )

    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return pairs


def _chart_title(spec: dict[str, Any]) -> str:
    """Build a readable chart title from a chart spec."""
    chart_type = spec["type"]
    if chart_type == "scatter":
        return f"{spec['y']} vs {spec['x']}"
    if chart_type == "boxplot":
        return f"{spec['y']} by {spec['x']}"
    if chart_type == "bar":
        return f"{spec['column']} Distribution"
    if chart_type == "histogram":
        return f"Distribution of {spec['column']}"
    return "Chart"


def _chart_reason(spec: dict[str, Any]) -> str:
    """Explain why a chart was selected."""
    chart_type = spec["type"]
    if chart_type == "scatter":
        corr = spec.get("correlation", 0)
        return (
            f"Selected because {spec['y']} and {spec['x']} show the strongest numeric "
            f"relationship (r={format_correlation(corr)})."
        )
    if chart_type == "boxplot":
        return (
            f"Selected to compare {spec['y']} across {spec['x']} categories with meaningful spread."
        )
    if chart_type == "bar":
        return f"Selected to summarize how records are distributed across {spec['column']} categories."
    if chart_type == "histogram":
        return f"Selected because the dataset is large enough for a useful distribution view of {spec['column']}."
    return "Selected by chart planning rules."


def _chart_insight_label(df: pd.DataFrame, spec: dict[str, Any]) -> str:
    """Generate a short insight line for chart captions."""
    chart_type = spec["type"]

    if chart_type == "scatter":
        corr = float(spec.get("correlation", 0))
        direction = "rises" if corr > 0 else "falls"
        y_col = spec["y"]
        series = df[y_col].dropna()
        range_text = format_value_range(y_col, series.min(), series.max())
        return (
            f"{y_col} {direction} with {spec['x']} (r={format_correlation(corr)}); "
            f"{range_text}."
        )

    if chart_type == "boxplot":
        grouped = df.groupby(spec["x"], observed=True)[spec["y"]].mean(numeric_only=True)
        top_group = grouped.idxmax()
        top_value = grouped.max()
        y_col = spec["y"]
        series = df[y_col].dropna()
        range_text = format_value_range(y_col, series.min(), series.max())
        suffix = " Small groups — directional only." if spec.get("directional") else ""
        return (
            f"Highest average {y_col} in '{top_group}' "
            f"({format_number(top_value, y_col)}) by {spec['x']}; "
            f"{range_text}.{suffix}"
        )

    if chart_type == "bar":
        counts = df[spec["column"]].value_counts(dropna=False)
        top_value = counts.index[0]
        share = counts.iloc[0] / max(len(df), 1)
        return f"'{top_value}' leads {spec['column']} ({share:.0%} of records)."

    if chart_type == "histogram":
        series = df[spec["column"]].dropna()
        if series.mean() > series.median():
            skew = "right-skewed"
        elif series.mean() < series.median():
            skew = "left-skewed"
        else:
            skew = "balanced"
        column = spec["column"]
        return (
            f"{column} distribution is {skew}; "
            f"mean {format_number(series.mean(), column)}, "
            f"median {format_number(series.median(), column)}."
        )

    return "Review the strongest pattern in the selected fields."


def _chart_caption(df: pd.DataFrame, spec: dict[str, Any]) -> str:
    """Generate a short deterministic caption (same as insight label for PDF use)."""
    return _chart_insight_label(df, spec)


def _chart_layout(spec: dict[str, Any]) -> str:
    """Return layout hint for PDF placement."""
    return "hero" if spec["type"] == "scatter" else "compact"


def _build_column_options(df: pd.DataFrame) -> dict[str, Any]:
    """Build dropdown-ready column lists for a future graph builder UI."""
    numeric_cols = _numeric_columns(df)
    group_cols = _categorical_columns(df)
    bar_cols = _bar_chart_columns(df)

    supported: list[str] = []
    if len(numeric_cols) >= 2:
        supported.append("scatter")
    if numeric_cols and group_cols:
        supported.append("boxplot")
    if bar_cols:
        supported.append("bar")
    if len(df) >= MIN_ROWS_FOR_HISTOGRAM:
        histogram_ready = any(
            df[column].nunique(dropna=True) >= MIN_UNIQUE_FOR_HISTOGRAM for column in numeric_cols
        )
        if histogram_ready:
            supported.append("histogram")

    x_columns = list(dict.fromkeys([*numeric_cols, *group_cols, *bar_cols]))
    return {
        "available_x_columns": x_columns,
        "available_y_columns": numeric_cols,
        "available_group_columns": group_cols,
        "supported_chart_types": supported,
    }


def _build_chart_metadata(
    df: pd.DataFrame,
    spec: dict[str, Any],
    chart_path: Path,
    column_options: dict[str, Any],
) -> dict[str, Any]:
    """
    Build UI-ready chart metadata for PDF rendering and future graph builders.

    Fields are designed to map directly to future dropdowns:
    chart type, X variable, Y variable, and grouping variable.
    """
    chart_type = spec["type"]
    insight = _chart_insight_label(df, spec)
    metadata: dict[str, Any] = {
        "chart_type": chart_type,
        "x_column": spec.get("x") or spec.get("column"),
        "y_column": spec.get("y"),
        "group_column": spec.get("x") if chart_type == "boxplot" else None,
        "available_x_columns": column_options["available_x_columns"],
        "available_y_columns": column_options["available_y_columns"],
        "available_group_columns": column_options["available_group_columns"],
        "supported_chart_types": column_options["supported_chart_types"],
        "title": _chart_title(spec),
        "caption": insight,
        "insight_label": insight,
        "reason_selected": _chart_reason(spec),
        "caution": None,
        "path": chart_path,
        "layout": _chart_layout(spec),
    }

    if spec.get("directional"):
        metadata["caution"] = "Small group sizes — interpret as directional only."
    elif is_small_sample(len(df)):
        metadata["caution"] = "Small sample size — interpret patterns as directional."

    return metadata


def plan_charts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Choose a focused set of high-value charts for the dataset."""
    plan: list[dict[str, Any]] = []
    numeric_cols = _numeric_columns(df)
    group_cols = _categorical_columns(df)
    bar_cols = _bar_chart_columns(df)
    pairs = _correlation_pairs(df, numeric_cols)
    used_keys: set[str] = set()

    scatter_count = 0
    for pair in pairs:
        if scatter_count >= MAX_SCATTER_PLOTS:
            break
        if len(pairs) == 1 or pair["abs_correlation"] >= MIN_SCATTER_CORRELATION:
            plan.append({"type": "scatter", **pair})
            scatter_count += 1

    boxplot_pairs: list[tuple[float, dict[str, Any]]] = []
    for group_col in group_cols:
        for numeric_col in numeric_cols:
            assessment = assess_grouped_comparison(df, group_col, numeric_col)
            if assessment is None:
                continue
            spread = df.groupby(group_col, observed=True)[numeric_col].std(numeric_only=True).mean(skipna=True)
            if pd.isna(spread) or spread == 0:
                continue
            boxplot_pairs.append(
                (
                    float(spread),
                    {
                        "type": "boxplot",
                        "x": group_col,
                        "y": numeric_col,
                        "directional": assessment["directional"],
                    },
                )
            )

    boxplot_pairs.sort(key=lambda item: item[0], reverse=True)
    for _, spec in boxplot_pairs[:MAX_BOXPLOTS]:
        plan.append(spec)

    used_group_axes = {item["x"] for item in plan if item["type"] == "boxplot"}
    for column in bar_cols:
        if len([item for item in plan if item["type"] == "bar"]) >= MAX_BAR_CHARTS:
            break
        if column in used_group_axes:
            continue
        plan.append({"type": "bar", "column": column})

    if len(df) >= MIN_ROWS_FOR_HISTOGRAM:
        scatter_axes = {col for item in plan if item["type"] == "scatter" for col in (item["x"], item["y"])}
        histogram_candidates = [
            column
            for column in numeric_cols
            if column not in scatter_axes and df[column].nunique(dropna=True) >= MIN_UNIQUE_FOR_HISTOGRAM
        ]
        histogram_candidates.sort(key=lambda column: float(df[column].std(skipna=True)), reverse=True)
        for column in histogram_candidates[:MAX_HISTOGRAMS]:
            if len(plan) >= MAX_TOTAL_CHARTS:
                break
            plan.append({"type": "histogram", "column": column})

    final_plan = plan[:MAX_TOTAL_CHARTS]
    logger.info("Chart plan selected: %s", [item["type"] for item in final_plan])
    return final_plan


def _create_scatter_chart(df: pd.DataFrame, spec: dict[str, Any], output_path: Path) -> Path:
    """Create a scatterplot with an optional trend line."""
    col_x, col_y = spec["x"], spec["y"]
    data = df[[col_x, col_y]].dropna()
    x, y = data[col_x], data[col_y]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    ax.scatter(x, y, color=COLORS["secondary"], alpha=0.78, edgecolors="white", s=52)

    if len(x) >= 3:
        trend = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.poly1d(trend)(x_line), color=COLORS["primary"], linewidth=2, label="Trend")

    ax.set_title(_chart_title(spec), pad=12)
    ax.set_xlabel(col_x, labelpad=8)
    ax.set_ylabel(col_y, labelpad=8)
    apply_axis_format(ax, col_x, axis="x")
    apply_axis_format(ax, col_y, axis="y")
    if len(x) >= 3:
        ax.legend(frameon=False, loc="best")
    _style_axes(ax)

    chart_file = output_path / f"scatter_{_sanitize_filename(col_x)}--{_sanitize_filename(col_y)}.png"
    _save_chart(fig, chart_file, bottom=0.14)
    return chart_file


def _create_boxplot_chart(df: pd.DataFrame, spec: dict[str, Any], output_path: Path) -> Path:
    """Create a boxplot comparing a numeric metric across categories."""
    group_col, numeric_col = spec["x"], spec["y"]
    grouped = [
        group[numeric_col].dropna().values
        for _, group in df.groupby(group_col, observed=True)
        if not group[numeric_col].dropna().empty
    ]
    labels = [
        str(label)
        for label, group in df.groupby(group_col, observed=True)
        if not group[numeric_col].dropna().empty
    ]

    label_count = len(labels)
    longest_label = max((len(label) for label in labels), default=0)
    rotation = 0
    if label_count > 4 or longest_label > 10:
        rotation = 30
    elif label_count > 2:
        rotation = 15

    fig_height = max(FIGURE_BOXPLOT_HEIGHT, 3.4 + 0.08 * label_count)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, fig_height))
    ax.boxplot(
        grouped,
        tick_labels=labels,
        widths=0.52,
        patch_artist=True,
        boxprops={"facecolor": COLORS["accent"], "color": COLORS["primary"], "linewidth": 1.2},
        medianprops={"color": COLORS["primary"], "linewidth": 2.0},
        whiskerprops={"color": COLORS["muted"], "linewidth": 1.2},
        capprops={"color": COLORS["muted"], "linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.55},
    )
    ax.set_title(_chart_title(spec), pad=10, fontsize=13)
    ax.set_xlabel(group_col, labelpad=10, fontsize=10)
    ax.set_ylabel(numeric_col, labelpad=10, fontsize=10)
    ax.tick_params(axis="x", labelsize=9, pad=4, rotation=rotation)
    if rotation:
        for label in ax.get_xticklabels():
            label.set_ha("right")
    apply_axis_format(ax, numeric_col, axis="y")
    ax.margins(y=0.10, x=0.04)
    _style_axes(ax)

    bottom_margin = 0.24 if rotation else 0.16
    chart_file = output_path / f"boxplot_{_sanitize_filename(numeric_col)}_by_{_sanitize_filename(group_col)}.png"
    _save_chart(fig, chart_file, bottom=bottom_margin)
    return chart_file


def _create_bar_chart(df: pd.DataFrame, column: str, output_path: Path) -> Path:
    """Create a horizontal bar chart for a categorical distribution."""
    value_counts = df[column].value_counts(dropna=False).head(CATEGORICAL_TOP_N)
    labels = [str(label) for label in value_counts.index[::-1]]
    values = value_counts.values[::-1]
    chart_height = max(FIGURE_COMPACT_HEIGHT, 0.42 * len(labels) + 1.8)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, chart_height))
    bars = ax.barh(labels, values, color=COLORS["secondary"], edgecolor="white", height=0.65)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            format_number(value),
            va="center",
            ha="left",
            fontsize=9,
            color=COLORS["text"],
        )

    ax.set_title(f"{column} Distribution", pad=12)
    ax.set_xlabel("Count", labelpad=8)
    ax.set_xlim(0, max(values) * 1.12)
    apply_axis_format(ax, None, axis="x")
    _style_axes(ax)

    chart_file = output_path / f"bar_{_sanitize_filename(column)}.png"
    _save_chart(fig, chart_file, bottom=0.14)
    return chart_file


def _create_histogram_chart(df: pd.DataFrame, column: str, output_path: Path) -> Path:
    """Create a histogram for large datasets where distribution views are useful."""
    series = df[column].dropna()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_COMPACT_HEIGHT))
    ax.hist(series, bins=HISTOGRAM_BINS, color=COLORS["secondary"], edgecolor=COLORS["background"], linewidth=0.8, alpha=0.9)
    mean_label = format_number(series.mean(), column)
    ax.axvline(series.mean(), color=COLORS["primary"], linestyle="--", linewidth=1.8, label=f"Mean {mean_label}")
    ax.set_title(f"Distribution of {column}", pad=12)
    ax.set_xlabel(column, labelpad=8)
    ax.set_ylabel("Frequency", labelpad=8)
    apply_axis_format(ax, column, axis="x")
    ax.legend(frameon=False)
    _style_axes(ax)

    chart_file = output_path / f"histogram_{_sanitize_filename(column)}.png"
    _save_chart(fig, chart_file, bottom=0.14)
    return chart_file


def generate_charts(df: pd.DataFrame, output_dir: str | Path) -> list[dict[str, Any]]:
    """
    Generate charts and return UI-ready metadata for each selected visualization.

    Returns:
        List of chart metadata dictionaries including file path, caption, and layout hints.
    """
    _apply_chart_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chart_results: list[dict[str, Any]] = []
    column_options = _build_column_options(df)
    for spec in plan_charts(df):
        if spec["type"] == "scatter":
            chart_file = _create_scatter_chart(df, spec, output_path)
        elif spec["type"] == "boxplot":
            chart_file = _create_boxplot_chart(df, spec, output_path)
        elif spec["type"] == "bar":
            chart_file = _create_bar_chart(df, spec["column"], output_path)
        elif spec["type"] == "histogram":
            chart_file = _create_histogram_chart(df, spec["column"], output_path)
        else:
            continue

        metadata = _build_chart_metadata(df, spec, chart_file, column_options)
        chart_results.append(metadata)
        logger.info("Created %s chart: %s", spec["type"], chart_file.name)

    return chart_results


def get_chart_column_options(df: pd.DataFrame) -> dict[str, Any]:
    """Return dropdown-ready column lists for interactive chart builders."""
    return _build_column_options(df)


def get_recommended_charts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the auto-selected chart plan used by the PDF pipeline."""
    return plan_charts(df)


def chart_reason_for_spec(spec: dict[str, Any]) -> str:
    """Public wrapper for chart recommendation rationale (Streamlit metadata panel)."""
    return _chart_reason(spec)


def chart_insight_for_spec(df: pd.DataFrame, spec: dict[str, Any]) -> str:
    """Public wrapper for deterministic chart insight text."""
    return _chart_insight_label(df, spec)


def build_spec_from_ui(
    df: pd.DataFrame,
    chart_type: str,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    group_column: str | None = None,
) -> dict[str, Any]:
    """Build a chart spec dict from Streamlit control selections."""
    if chart_type == "scatter":
        corr = 0.0
        if x_column and y_column and x_column in df.columns and y_column in df.columns:
            x_vals = pd.to_numeric(df[x_column], errors="coerce")
            y_vals = pd.to_numeric(df[y_column], errors="coerce")
            valid = x_vals.notna() & y_vals.notna()
            if valid.sum() >= 2:
                corr = float(x_vals[valid].corr(y_vals[valid]))
        return {"type": "scatter", "x": x_column, "y": y_column, "correlation": corr}
    if chart_type == "boxplot":
        return {"type": "boxplot", "x": group_column, "y": y_column}
    return {"type": "bar", "column": x_column}


def find_matching_recommended_spec(
    recommended: list[dict[str, Any]],
    chart_type: str,
    *,
    x_column: str | None = None,
    y_column: str | None = None,
    group_column: str | None = None,
) -> dict[str, Any] | None:
    """Return a planner spec that matches the current UI chart selection, if any."""
    for spec in recommended:
        if spec.get("type") != chart_type:
            continue
        if chart_type == "scatter" and spec.get("x") == x_column and spec.get("y") == y_column:
            return spec
        if chart_type == "boxplot" and spec.get("x") == group_column and spec.get("y") == y_column:
            return spec
        if chart_type == "bar" and spec.get("column") == x_column:
            return spec
    return None


def get_chart_skip_reasons(df: pd.DataFrame) -> list[str]:
    """Explain why certain visualizations were not selected."""
    reasons: list[str] = []
    plan = plan_charts(df)
    planned_types = {item["type"] for item in plan}
    numeric_cols = _numeric_columns(df)

    if len(numeric_cols) >= 2 and "scatter" not in planned_types:
        reasons.append("No scatterplots selected: numeric relationships were weak or already covered.")
    if numeric_cols and _categorical_columns(df) and "boxplot" not in planned_types:
        reasons.append("No boxplots selected: category groupings lacked enough numeric spread.")
    if _bar_chart_columns(df) and "bar" not in planned_types:
        reasons.append("No bar charts selected: categorical distributions were redundant or unsuitable.")
    if len(df) < MIN_ROWS_FOR_HISTOGRAM:
        reasons.append(f"Histograms skipped: dataset has {len(df)} rows (minimum {MIN_ROWS_FOR_HISTOGRAM} recommended).")
    elif numeric_cols and "histogram" not in planned_types:
        reasons.append("Histograms skipped: relationship and comparison charts were prioritized.")

    if not plan:
        reasons.append("No charts met quality thresholds for this dataset.")
    return reasons
