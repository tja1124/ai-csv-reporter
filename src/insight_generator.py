"""Deterministic report intelligence: quality scores, insights, and summaries."""

from typing import Any

import pandas as pd

from src.chart_generator import get_chart_skip_reasons
from src.column_utils import (
    assess_grouped_comparison,
    get_groupable_categorical_columns,
    is_groupable_categorical,
    is_id_like_column,
    is_small_sample,
    sample_size_caution,
)
from src.config import (
    CORRELATION_MIN_ABS,
    DOMINANT_CATEGORY_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER,
    SCATTER_CORRELATION_NOTE,
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


def _directional_prefix(row_count: int) -> str:
    """Prefix insight text when the sample is too small for strong claims."""
    if is_small_sample(row_count):
        return "In this small sample, "
    return ""


def _describe_correlation(pair: dict[str, Any], row_count: int) -> str:
    """Turn a correlation pair into a cautious business-oriented insight."""
    col_a = pair["column_a"]
    col_b = pair["column_b"]
    value = pair["correlation"]
    prefix = _directional_prefix(row_count)

    if value >= SCATTER_CORRELATION_NOTE:
        return (
            f"{prefix}{col_b} appears to rise with {col_a}, which may indicate "
            f"that these metrics move together in practice."
        )
    if value <= -SCATTER_CORRELATION_NOTE:
        return (
            f"{prefix}{col_b} appears to fall as {col_a} increases, suggesting "
            f"these metrics may move in opposite directions."
        )
    return (
        f"{prefix}{col_a} and {col_b} show an early positive pattern that "
        f"would need more records to confirm."
    )


def _describe_dominant_category(column: str, top_value: str, share: float, row_count: int) -> str:
    """Describe a heavily skewed categorical field."""
    prefix = _directional_prefix(row_count)
    return (
        f"{prefix}'{column}' is concentrated in '{top_value}' ({share:.0%} of records), "
        f"which may limit balanced comparisons across categories."
    )


def _describe_outliers(column: str, row_count: int) -> str:
    """Describe a column with notable outlier activity."""
    prefix = _directional_prefix(row_count)
    return (
        f"{prefix}'{column}' includes values that sit outside the typical range and "
        f"may deserve a closer review before drawing conclusions."
    )


def calculate_dataset_quality_score(
    df: pd.DataFrame,
    validation_warnings: list[str],
) -> dict[str, Any]:
    """Calculate a simple dataset quality score from 0 to 100."""
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
    """Detect possible outliers in numeric columns using the IQR method."""
    summary: dict[str, dict[str, float | int]] = {}
    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        if is_id_like_column(column, numeric_df[column], len(df), for_numeric=True):
            continue

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
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        if outlier_count == 0:
            continue

        summary[column] = {
            "count": outlier_count,
            "percentage": round((outlier_count / len(series)) * 100, 2),
        }

    logger.info("Outlier detection complete for %d column(s)", len(summary))
    return summary


def get_correlation_highlights(df: pd.DataFrame) -> dict[str, Any]:
    """Identify the strongest numeric correlations above the configured threshold."""
    numeric_df = df.select_dtypes(include="number")
    numeric_cols = [
        column
        for column in numeric_df.columns
        if not is_id_like_column(column, numeric_df[column], len(df), for_numeric=True)
    ]

    if len(numeric_cols) < 2:
        return {
            "strongest_positive": None,
            "strongest_negative": None,
            "notable_pairs": [],
        }

    correlation = numeric_df[numeric_cols].corr(numeric_only=True)
    notable_pairs: list[dict[str, Any]] = []

    for index, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[index + 1 :]:
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


def _describe_group_spread(df: pd.DataFrame, row_count: int) -> list[str]:
    """Describe meaningful differences in numeric metrics across safe categories."""
    insights: list[str] = []
    numeric_cols = [
        column
        for column in df.select_dtypes(include="number").columns
        if not is_id_like_column(column, df[column], len(df), for_numeric=True)
    ]

    best_spread = 0.0
    best_message = ""

    for group_col in get_groupable_categorical_columns(df):
        for numeric_col in numeric_cols:
            assessment = assess_grouped_comparison(df, group_col, numeric_col)
            if assessment is None:
                continue

            grouped = df.groupby(group_col, observed=True)[numeric_col].mean(numeric_only=True)
            if grouped.empty or grouped.nunique() < 2:
                continue

            spread = float(grouped.max() - grouped.min())
            if spread <= best_spread:
                continue

            best_spread = spread
            top_group = grouped.idxmax()
            prefix = _directional_prefix(row_count)
            qualifier = (
                " This comparison is directional because some groups are small."
                if assessment["directional"] or is_small_sample(row_count)
                else ""
            )
            best_message = (
                f"{prefix}{numeric_col} appears highest in '{top_group}' when viewed "
                f"across '{group_col}', which may point to category-level differences "
                f"worth exploring.{qualifier}"
            )

    if best_message:
        insights.append(best_message)

    return insights


def generate_key_insights(
    df: pd.DataFrame,
    overview: dict[str, Any],
    missing_values: dict[str, Any],
    categorical_summary: dict[str, dict[str, Any]],
    outlier_summary: dict[str, dict[str, float | int]],
    correlation_highlights: dict[str, Any],
    chart_skip_reasons: list[str],
) -> list[str]:
    """Generate narrative insights focused on patterns, not raw repetition."""
    insights: list[str] = []
    row_count = overview["row_count"]

    numeric_df = df.select_dtypes(include="number")
    numeric_cols = [
        column
        for column in numeric_df.columns
        if not is_id_like_column(column, numeric_df[column], len(df), for_numeric=True)
    ]

    if len(numeric_cols) >= 2:
        correlation = numeric_df[numeric_cols].corr(numeric_only=True)
        top_pair = None
        top_value = 0.0
        for index, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[index + 1 :]:
                value = correlation.loc[col_a, col_b]
                if pd.isna(value):
                    continue
                if abs(value) > abs(top_value):
                    top_value = float(value)
                    top_pair = {
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": round(float(value), 3),
                    }
        if top_pair:
            insights.append(_describe_correlation(top_pair, row_count))

    for column, stats in categorical_summary.items():
        if not is_groupable_categorical(df, column):
            continue
        if not stats["top_values"]:
            continue
        top_value, top_count = next(iter(stats["top_values"].items()))
        share = top_count / max(row_count, 1)
        if share >= DOMINANT_CATEGORY_THRESHOLD:
            insights.append(_describe_dominant_category(column, top_value, share, row_count))

    if missing_values["columns_with_missing"]:
        top_missing = max(
            missing_values["columns_with_missing"].items(),
            key=lambda item: item[1]["percentage"],
        )
        column_name, info = top_missing
        if info["percentage"] >= 10:
            insights.append(
                f"Missing data is most visible in '{column_name}', which could "
                f"reduce confidence in analyses that rely on that field."
            )

    for column, stats in outlier_summary.items():
        if stats["percentage"] >= 5:
            insights.append(_describe_outliers(column, row_count))

    insights.extend(_describe_group_spread(df, row_count))

    if chart_skip_reasons and len(insights) < 4:
        insights.append(
            "Some comparisons were intentionally omitted where they would not "
            "add reliable business insight."
        )

    logger.info("Generated %d key insight(s)", len(insights))
    return insights[:6]


def generate_important_findings(
    key_insights: list[str],
    quality_score: dict[str, Any],
    row_count: int,
) -> list[str]:
    """Return the top findings for the front section of the report."""
    findings: list[str] = []

    caution = sample_size_caution(row_count)
    if caution:
        findings.append(caution)

    findings.extend(key_insights[:4])

    if quality_score["rating"] in {"Fair", "Poor"} and quality_score["deductions"]:
        findings.append(
            f"Data quality is rated {quality_score['rating']}, so conclusions should "
            f"be validated before action."
        )

    return findings[:5]


def generate_executive_brief(
    csv_name: str,
    overview: dict[str, Any],
    quality_score: dict[str, Any],
    important_findings: list[str],
) -> dict[str, str]:
    """Build a concise deterministic executive brief when AI is not used."""
    lead = next(
        (finding for finding in important_findings if not finding.startswith("Sample Size Caution")),
        "The dataset shows stable structure with no major anomalies in the initial review.",
    )
    overview_text = (
        f"This analytics brief summarizes '{csv_name}' ({overview['row_count']:,} records, "
        f"{overview['column_count']} fields) with an overall data quality rating of "
        f"{quality_score['rating']} ({quality_score['score']}/100). {lead}"
    )
    return {"overview": overview_text}


def build_data_warnings(
    validation_warnings: list[str],
    quality_score: dict[str, Any],
    row_count: int,
) -> list[str]:
    """Combine validation warnings and quality deductions into report warnings."""
    warnings = list(validation_warnings)

    caution = sample_size_caution(row_count)
    if caution:
        warnings.insert(0, caution)

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
    """Run intelligence functions and return a single insights bundle."""
    logger.info("Generating report intelligence")

    row_count = overview["row_count"]
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
    important_findings = generate_important_findings(key_insights, quality_score, row_count)
    data_warnings = build_data_warnings(validation_warnings, quality_score, row_count)

    logger.info("Report intelligence generation complete")

    return {
        "quality_score": quality_score,
        "key_insights": key_insights,
        "important_findings": important_findings,
        "data_warnings": data_warnings,
        "outlier_summary": outlier_summary,
        "correlation_highlights": correlation_highlights,
        "chart_skip_reasons": chart_skip_reasons,
        "sample_size_caution": sample_size_caution(row_count),
    }
