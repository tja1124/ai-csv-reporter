"""Generate charts from dataset analysis."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_TOP_N,
    COLORS,
    FIGURE_DPI,
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    HISTOGRAM_BINS,
    ID_LIKE_NAME_KEYWORDS,
    ID_LIKE_UNIQUENESS_RATIO,
    MAX_UNIQUE_FOR_BAR_CHART,
    MIN_UNIQUE_FOR_HISTOGRAM,
)
from src.logger import get_logger

logger = get_logger()


def _sanitize_filename(name: str) -> str:
    """Convert a column name into a safe filename."""
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def _apply_chart_style() -> None:
    """Apply a consistent, clean visual style to all charts."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "axes.titleweight": "600",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.labelweight": "500",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "savefig.facecolor": COLORS["background"],
            "savefig.edgecolor": COLORS["background"],
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    """Apply finishing touches to a single chart axis."""
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["muted"], length=0)


def _save_chart(fig: plt.Figure, chart_file: Path) -> None:
    """Save a chart with consistent export settings."""
    fig.savefig(chart_file, dpi=FIGURE_DPI)
    plt.close(fig)


def _is_id_like_column(column_name: str, series: pd.Series, row_count: int, *, for_histogram: bool = False) -> bool:
    """
    Detect columns that behave like identifiers rather than analytical fields.

    Numeric histograms use stricter rules so small datasets with unique values
    (e.g. age, salary) are still charted. Categorical charts also skip columns
    where nearly every value is unique, since bar charts would not be readable.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(column_name).lower())
    if any(keyword in normalized_name for keyword in ID_LIKE_NAME_KEYWORDS):
        return True

    non_null_count = series.dropna().shape[0]
    if non_null_count == 0 or row_count <= 1:
        return False

    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / non_null_count

    if for_histogram:
        # For numeric fields, only treat very large near-unique columns as IDs.
        return unique_ratio >= ID_LIKE_UNIQUENESS_RATIO and unique_count >= 50

    # For categorical fields, near-unique columns produce unreadable bar charts.
    return unique_ratio >= ID_LIKE_UNIQUENESS_RATIO


def _histogram_skip_reason(column: str, series: pd.Series, row_count: int) -> str | None:
    """Return a skip reason for a numeric histogram, or None if the column is chartable."""
    cleaned = series.dropna()
    unique_count = cleaned.nunique()

    if cleaned.empty:
        return "column is all null"
    if _is_id_like_column(column, series, row_count, for_histogram=True):
        return "column appears to be an ID"
    if unique_count < MIN_UNIQUE_FOR_HISTOGRAM:
        return f"only {unique_count} unique value(s)"
    return None


def _bar_chart_skip_reason(column: str, series: pd.Series, row_count: int) -> str | None:
    """Return a skip reason for a categorical bar chart, or None if the column is chartable."""
    unique_count = series.nunique(dropna=True)

    if series.isnull().all():
        return "column is all null"
    if _is_id_like_column(column, series, row_count):
        return "column appears to be an ID"
    if unique_count >= MAX_UNIQUE_FOR_BAR_CHART:
        return f"{unique_count} unique values (max {MAX_UNIQUE_FOR_BAR_CHART})"
    if unique_count == 0:
        return "no usable values"
    return None


def get_chart_skip_reasons(df: pd.DataFrame) -> list[str]:
    """
    Explain why certain columns were excluded from chart generation.

    Returns:
        Plain-English reasons for skipped histograms and bar charts.
    """
    reasons: list[str] = []

    numeric_df = df.select_dtypes(include="number")
    for column in numeric_df.columns:
        reason = _histogram_skip_reason(column, numeric_df[column], len(df))
        if reason:
            reasons.append(f"Histogram skipped for '{column}': {reason}.")

    categorical_df = df.select_dtypes(exclude="number")
    for column in categorical_df.columns:
        reason = _bar_chart_skip_reason(column, categorical_df[column], len(df))
        if reason:
            reasons.append(f"Bar chart skipped for '{column}': {reason}.")

    if len(numeric_df.columns) < 2:
        reasons.append(
            f"Correlation heatmap skipped: found {len(numeric_df.columns)} numeric column(s), need at least 2."
        )

    return reasons


def _select_histogram_columns(df: pd.DataFrame) -> list[str]:
    """
    Choose numeric columns suitable for histograms.

    Decision rules:
    - Must be numeric
    - Must have enough unique values to show a distribution
    - Must not look like an ID column
    - Must not be entirely null
    """
    selected: list[str] = []
    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        reason = _histogram_skip_reason(column, numeric_df[column], len(df))
        if reason:
            logger.info("Skipping histogram for '%s': %s", column, reason)
            continue

        selected.append(column)

    return selected


def _select_bar_chart_columns(df: pd.DataFrame) -> list[str]:
    """
    Choose categorical columns suitable for bar charts.

    Decision rules:
    - Must be non-numeric
    - Must have fewer than 15 unique values
    - Must not look like an ID column
    - Must not be entirely null
    """
    selected: list[str] = []
    categorical_df = df.select_dtypes(exclude="number")

    for column in categorical_df.columns:
        reason = _bar_chart_skip_reason(column, categorical_df[column], len(df))
        if reason:
            logger.info("Skipping bar chart for '%s': %s", column, reason)
            continue

        selected.append(column)

    return selected


def generate_numeric_histograms(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """
    Create histogram charts for suitable numeric columns.

    Args:
        df: The DataFrame to chart.
        output_dir: Directory where chart images will be saved.

    Returns:
        A list of paths to the generated chart files.
    """
    _apply_chart_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chart_paths: list[Path] = []
    columns = _select_histogram_columns(df)
    logger.info("Generating histograms for columns: %s", columns or "none")

    for column in columns:
        series = df[column].dropna()

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        ax.hist(
            series,
            bins=HISTOGRAM_BINS,
            color=COLORS["secondary"],
            edgecolor=COLORS["background"],
            linewidth=0.8,
            alpha=0.9,
        )

        mean_value = series.mean()
        ax.axvline(
            mean_value,
            color=COLORS["primary"],
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_value:,.2g}",
        )

        ax.set_title(f"Distribution of {column}", pad=16)
        ax.set_xlabel(column, labelpad=10)
        ax.set_ylabel("Frequency", labelpad=10)
        ax.legend(frameon=False, loc="upper right")
        _style_axes(ax)

        chart_file = output_path / f"histogram_{_sanitize_filename(column)}.png"
        _save_chart(fig, chart_file)
        chart_paths.append(chart_file)
        logger.info("Created histogram: %s", chart_file.name)

    return chart_paths


def generate_categorical_bar_charts(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """
    Create horizontal bar charts for suitable categorical columns.

    Args:
        df: The DataFrame to chart.
        output_dir: Directory where chart images will be saved.

    Returns:
        A list of paths to the generated chart files.
    """
    _apply_chart_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chart_paths: list[Path] = []
    columns = _select_bar_chart_columns(df)
    logger.info("Generating bar charts for columns: %s", columns or "none")

    for column in columns:
        value_counts = df[column].value_counts(dropna=False).head(CATEGORICAL_TOP_N)
        if value_counts.empty:
            continue

        labels = [str(label) for label in value_counts.index[::-1]]
        values = value_counts.values[::-1]
        chart_height = max(FIGURE_HEIGHT, 0.45 * len(labels) + 2)

        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, chart_height))
        bars = ax.barh(
            labels,
            values,
            color=COLORS["secondary"],
            edgecolor=COLORS["background"],
            height=0.65,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,}",
                va="center",
                ha="left",
                fontsize=9,
                color=COLORS["text"],
            )

        ax.set_title(f"Top Values in {column}", pad=16)
        ax.set_xlabel("Count", labelpad=10)
        ax.set_ylabel("")
        ax.set_xlim(0, max(values) * 1.12)
        _style_axes(ax)

        chart_file = output_path / f"bar_{_sanitize_filename(column)}.png"
        _save_chart(fig, chart_file)
        chart_paths.append(chart_file)
        logger.info("Created bar chart: %s", chart_file.name)

    return chart_paths


def generate_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> Path | None:
    """
    Create a correlation heatmap when at least two numeric columns exist.

    Args:
        df: The DataFrame to chart.
        output_dir: Directory where the chart image will be saved.

    Returns:
        Path to the heatmap file, or None if fewer than two numeric columns exist.
    """
    _apply_chart_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df.columns) < 2:
        logger.info(
            "Skipping correlation heatmap: found %d numeric column(s), need at least 2",
            len(numeric_df.columns),
        )
        return None

    correlation = numeric_df.corr(numeric_only=True)
    size = max(6, len(correlation.columns) * 0.9)

    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation.columns)
    ax.set_title("Numeric Column Correlations", pad=16)

    for row in range(correlation.shape[0]):
        for col in range(correlation.shape[1]):
            value = correlation.iloc[row, col]
            if not np.isnan(value):
                text_color = "white" if abs(value) > 0.55 else COLORS["text"]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="500",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=COLORS["muted"])
    cbar.set_label("Correlation", color=COLORS["muted"], fontsize=9)
    ax.grid(False)
    _style_axes(ax)

    chart_file = output_path / "correlation_heatmap.png"
    _save_chart(fig, chart_file)
    logger.info("Created correlation heatmap: %s", chart_file.name)

    return chart_file
