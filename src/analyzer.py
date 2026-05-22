"""Dataset analysis functions."""

from typing import Any

import pandas as pd


def get_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return high-level summary information about the dataset.

    Args:
        df: The DataFrame to summarize.

    Returns:
        A dictionary with row count, column count, column names, and memory usage.
    """
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }


def analyze_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Classify each column by data type category.

    Args:
        df: The DataFrame to analyze.

    Returns:
        A dictionary mapping column names to type information.
    """
    column_info: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        dtype = str(df[column].dtype)
        if pd.api.types.is_numeric_dtype(df[column]):
            category = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            category = "datetime"
        else:
            category = "categorical"

        column_info[column] = {
            "dtype": dtype,
            "category": category,
            "unique_values": int(df[column].nunique(dropna=True)),
        }

    return column_info


def get_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate missing value counts and percentages for each column.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A dictionary with total missing values and per-column breakdown.
    """
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df) * 100).round(2)

    columns_with_missing = {
        column: {
            "count": int(missing_counts[column]),
            "percentage": float(missing_percentages[column]),
        }
        for column in df.columns
        if missing_counts[column] > 0
    }

    return {
        "total_missing": int(missing_counts.sum()),
        "columns_with_missing": columns_with_missing,
    }


def get_numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """
    Compute descriptive statistics for numeric columns.

    Args:
        df: The DataFrame to summarize.

    Returns:
        A dictionary mapping numeric column names to summary statistics.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}

    summary: dict[str, dict[str, float | None]] = {}
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        summary[column] = {
            "count": float(series.count()),
            "mean": float(series.mean()) if not series.empty else None,
            "std": float(series.std()) if not series.empty else None,
            "min": float(series.min()) if not series.empty else None,
            "25%": float(series.quantile(0.25)) if not series.empty else None,
            "50%": float(series.median()) if not series.empty else None,
            "75%": float(series.quantile(0.75)) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
        }

    return summary


def get_categorical_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Compute summary statistics for categorical (non-numeric) columns.

    Args:
        df: The DataFrame to summarize.

    Returns:
        A dictionary mapping categorical column names to top value counts.
    """
    categorical_df = df.select_dtypes(exclude="number")
    if categorical_df.empty:
        return {}

    summary: dict[str, dict[str, Any]] = {}
    for column in categorical_df.columns:
        value_counts = categorical_df[column].value_counts(dropna=False).head(10)
        summary[column] = {
            "unique_values": int(categorical_df[column].nunique(dropna=True)),
            "top_values": {str(key): int(value) for key, value in value_counts.items()},
        }

    return summary
