"""Deterministic report intelligence: quality scores, insights, and summaries."""

from typing import Any

import pandas as pd

from src.chart_generator import get_chart_skip_reasons
from src.config import (
    CORRELATION_MIN_ABS,
    DOMINANT_CATEGORY_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
)
from src.logger import get_logger

logger = get_logger()


def _quality_rating(score: int) -> str:
    """Map a numeric quality score to a readable label."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def _count_usable_columns(df: pd.DataFrame) -> int:
    """Count columns that contain at least one non-null value."""
    return sum(1 for column in df.columns if not df[column].isnull().all())


def calculate_dataset_quality_score(
    df: pd.DataFrame,
    validation_warnings: list[str],
) -> dict[str, Any]:
    """
    Calculate a simple dataset quality score from 0 to 100.

    Deductions are based on missing values, duplicate rows, all-null columns,
    duplicate column names, and the share of usable columns.

    Returns:
        Score, rating label, and bullet-point deduction explanations.
    """
    score = 100
    deductions: list[str] = []

    total_cells = max(len(df) * len(df.columns), 1)
    missing_pct = (df.isnull().sum().sum() / total_cells) * 100
    if missing_pct > 0:
        missing_deduction = min(round(missing_pct * 0.6), 30)
        score -= missing_deduction
        deductions.append(
            f"Missing values cover {missing_pct:.1f}% of cells (-{missing_deduction} points)."
        )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        duplicate_ratio = duplicate_rows / max(len(df), 1)
        duplicate_deduction = min(round(duplicate_ratio * 100 * 0.2), 20)
        score -= duplicate_deduction
        deductions.append(
            f"{duplicate_rows:,} duplicate row(s) detected (-{duplicate_deduction} points)."
        )

    null_columns = [col for col in df.columns if df[col].isnull().all()]
    if null_columns:
        null_deduction = min(len(null_columns) * 10, 20)
        score -= null_deduction
        deductions.append(
            f"{len(null_columns)} all-null column(s) found (-{null_deduction} points)."
        )

    if df.columns.duplicated().any():
        score -= 10
        deductions.append("Duplicate column names detected (-10 points).")

    usable_columns = _count_usable_columns(df)
    usable_ratio = usable_columns / max(len(df.columns), 1)
    if usable_ratio < 0.75:
        usability_deduction = min(round((0.75 - usable_ratio) * 40), 20)
        score -= usability_deduction
        deductions.append(
            f"Only {usable_columns} of {len(df.columns)} columns are usable "
            f"(-{usability_deduction} points)."
        )

    if any("missing headers" in warning.lower() for warning in validation_warnings):
        score -= 5
        deductions.append("Possible missing header row detected (-5 points).")

    final_score = max(0, min(score, 100))
    rating = _quality_rating(final_score)

    if not deductions:
        deductions.append("No major quality issues detected.")

    logger.info("Dataset quality score calculated: %d (%s)", final_score, rating)

    return {
        "score": final_score,
        "rating": rating,
        "deductions": deductions,
    }


def detect_outliers(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """
    Detect possible outliers in numeric columns using the IQR method.

    Outliers are flagged but never removed from the dataset.

    Returns:
        A mapping of column names to outlier counts and percentages.
    """
    summary: dict[str, dict[str, float | int]] = {}
    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper_bound = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count == 0:
            continue

        summary[column] = {
            "count": outlier_count,
            "percentage": round((outlier_count / len(series)) * 100, 2),
        }

    logger.info("Outlier detection complete for %d column(s)", len(summary))
    return summary


def get_correlation_highlights(df: pd.DataFrame) -> dict[str, Any]:
    """
    Identify the strongest numeric correlations above the configured threshold.

    Self-correlations are ignored. Only pairs with absolute correlation >= 0.5
    are considered notable.

    Returns:
        Strongest positive pair, strongest negative pair, and all notable pairs.
    """
    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df.columns) < 2:
        return {
            "strongest_positive": None,
            "strongest_negative": None,
            "notable_pairs": [],
        }

    correlation = numeric_df.corr(numeric_only=True)
    columns = list(correlation.columns)
    notable_pairs: list[dict[str, Any]] = []

    for index, col_a in enumerate(columns):
        for col_b in columns[index + 1 :]:
            value = correlation.loc[col_a, col_b]
            if pd.isna(value) or abs(value) < CORRELATION_MIN_ABS:
                continue
            notable_pairs.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": round(float(value), 3),
                }
            )

    notable_pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)

    positive_pairs = [pair for pair in notable_pairs if pair["correlation"] > 0]
    negative_pairs = [pair for pair in notable_pairs if pair["correlation"] < 0]

    result = {
        "strongest_positive": positive_pairs[0] if positive_pairs else None,
        "strongest_negative": negative_pairs[0] if negative_pairs else None,
        "notable_pairs": notable_pairs,
    }
    logger.info("Found %d notable correlation pair(s)", len(notable_pairs))
    return result


def generate_key_insights(
    df: pd.DataFrame,
    overview: dict[str, Any],
    missing_values: dict[str, Any],
    categorical_summary: dict[str, dict[str, Any]],
    outlier_summary: dict[str, dict[str, float | int]],
    correlation_highlights: dict[str, Any],
    chart_skip_reasons: list[str],
) -> list[str]:
    """
    Generate plain-English insights from deterministic analysis results.

    Returns:
        A list of readable insight sentences for the PDF report.
    """
    insights: list[str] = []

    row_count = overview["row_count"]
    column_count = overview["column_count"]
    insights.append(
        f"The dataset contains {row_count:,} rows and {column_count} columns."
    )

    if missing_values["columns_with_missing"]:
        top_missing = max(
            missing_values["columns_with_missing"].items(),
            key=lambda item: item[1]["percentage"],
        )
        column_name, info = top_missing
        insights.append(
            f"'{column_name}' has the most missing values "
            f"({info['count']:,} cells, {info['percentage']}% of rows)."
        )
    else:
        insights.append("No missing values were found in any column.")

    strongest_positive = correlation_highlights["strongest_positive"]
    if strongest_positive:
        insights.append(
            f"Strongest positive correlation: {strongest_positive['column_a']} and "
            f"{strongest_positive['column_b']} (r={strongest_positive['correlation']:.2f})."
        )

    strongest_negative = correlation_highlights["strongest_negative"]
    if strongest_negative:
        insights.append(
            f"Strongest negative correlation: {strongest_negative['column_a']} and "
            f"{strongest_negative['column_b']} (r={strongest_negative['correlation']:.2f})."
        )

    for column, stats in categorical_summary.items():
        if not stats["top_values"]:
            continue
        top_value, top_count = next(iter(stats["top_values"].items()))
        share = top_count / max(row_count, 1)
        if share >= DOMINANT_CATEGORY_THRESHOLD:
            insights.append(
                f"'{column}' is dominated by '{top_value}' "
                f"({top_count:,} rows, {share:.0%} of the dataset)."
            )

    for column, stats in outlier_summary.items():
        if stats["percentage"] >= 5:
            insights.append(
                f"'{column}' may contain outliers "
                f"({stats['count']:,} values, {stats['percentage']}% of non-null rows)."
            )

    if chart_skip_reasons:
        insights.append(
            f"{len(chart_skip_reasons)} chart(s) were skipped due to column suitability rules."
        )

    logger.info("Generated %d key insight(s)", len(insights))
    return insights


def build_data_warnings(
    validation_warnings: list[str],
    quality_score: dict[str, Any],
) -> list[str]:
    """
    Combine validation warnings and quality deductions into report warnings.

    Returns:
        A list of warning messages for the PDF report.
    """
    warnings = list(validation_warnings)

    for deduction in quality_score["deductions"]:
        if deduction != "No major quality issues detected.":
            warnings.append(deduction)

    if not warnings:
        warnings.append("No data warnings detected.")

    return warnings


def generate_report_insights(
    df: pd.DataFrame,
    overview: dict[str, Any],
    missing_values: dict[str, Any],
    categorical_summary: dict[str, dict[str, Any]],
    validation_warnings: list[str],
) -> dict[str, Any]:
    """
    Run all Phase 2 intelligence functions and return a single insights bundle.

    Returns:
        Quality score, key insights, warnings, outliers, and correlation highlights.
    """
    logger.info("Generating report intelligence")

    quality_score = calculate_dataset_quality_score(df, validation_warnings)
    outlier_summary = detect_outliers(df)
    correlation_highlights = get_correlation_highlights(df)
    chart_skip_reasons = get_chart_skip_reasons(df)

    key_insights = generate_key_insights(
        df=df,
        overview=overview,
        missing_values=missing_values,
        categorical_summary=categorical_summary,
        outlier_summary=outlier_summary,
        correlation_highlights=correlation_highlights,
        chart_skip_reasons=chart_skip_reasons,
    )

    data_warnings = build_data_warnings(validation_warnings, quality_score)

    logger.info("Report intelligence generation complete")

    return {
        "quality_score": quality_score,
        "key_insights": key_insights,
        "data_warnings": data_warnings,
        "outlier_summary": outlier_summary,
        "correlation_highlights": correlation_highlights,
        "chart_skip_reasons": chart_skip_reasons,
    }
