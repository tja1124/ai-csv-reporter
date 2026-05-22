"""Shared column classification helpers for charts and insights."""

import re

import pandas as pd

from src.config import (
    ID_LIKE_NAME_KEYWORDS,
    ID_LIKE_UNIQUENESS_RATIO,
    MAX_CATEGORIES_FOR_BOXPLOT,
    MIN_ROWS_PER_GROUP,
    SMALL_SAMPLE_THRESHOLD,
)


def is_id_like_column(
    column_name: str,
    series: pd.Series,
    row_count: int,
    *,
    for_numeric: bool = False,
) -> bool:
    """
    Detect identifier-like columns that should not drive grouped analysis.

    Columns such as name, uuid, or near-unique text fields are excluded.
    """
    normalized_name = re.sub(r"[^a-z0-9]", "", str(column_name).lower())
    if normalized_name == "name" or normalized_name.endswith("name"):
        return True
    if any(keyword in normalized_name for keyword in ID_LIKE_NAME_KEYWORDS):
        return True

    non_null_count = series.dropna().shape[0]
    if non_null_count == 0 or row_count <= 1:
        return False

    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / non_null_count

    if for_numeric:
        return unique_ratio >= ID_LIKE_UNIQUENESS_RATIO and unique_count >= 50

    return unique_ratio >= ID_LIKE_UNIQUENESS_RATIO


def is_groupable_categorical(df: pd.DataFrame, column: str) -> bool:
    """Return True when a categorical column is safe for grouped comparisons."""
    if column not in df.columns:
        return False

    series = df[column]
    if series.isnull().all():
        return False
    if is_id_like_column(column, series, len(df)):
        return False

    unique_count = series.nunique(dropna=True)
    return 2 <= unique_count <= MAX_CATEGORIES_FOR_BOXPLOT


def get_groupable_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical columns suitable for grouped comparisons."""
    return [
        column
        for column in df.select_dtypes(exclude="number").columns
        if is_groupable_categorical(df, column)
    ]


def assess_grouped_comparison(
    df: pd.DataFrame,
    group_col: str,
    numeric_col: str,
    *,
    min_per_group: int = MIN_ROWS_PER_GROUP,
) -> dict[str, bool] | None:
    """
    Check whether a grouped numeric comparison is viable.

    Returns:
        None if the comparison should be skipped, otherwise a dict with:
        - usable: whether the comparison can be shown
        - directional: whether group sizes are too small for strong claims
    """
    if not is_groupable_categorical(df, group_col):
        return None

    counts = df.groupby(group_col, observed=True)[numeric_col].count()
    counts = counts[counts > 0]
    if len(counts) < 2:
        return None

    min_count = int(counts.min())
    if min_count < 2:
        return None

    return {
        "usable": True,
        "directional": min_count < min_per_group,
    }


def sample_size_caution(row_count: int) -> str | None:
    """Return a standard caution sentence for small datasets."""
    if row_count < SMALL_SAMPLE_THRESHOLD:
        return (
            "Sample Size Caution: This dataset contains fewer than 30 records, "
            "so all findings should be treated as directional rather than definitive."
        )
    return None


def is_small_sample(row_count: int) -> bool:
    """Return True when the dataset is too small for strong claims."""
    return row_count < SMALL_SAMPLE_THRESHOLD
