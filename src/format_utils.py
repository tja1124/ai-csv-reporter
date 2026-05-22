"""Number formatting utilities for reports, charts, and insights."""

import math
import re
from typing import Any

CURRENCY_KEYWORDS = (
    "salary",
    "revenue",
    "cost",
    "price",
    "amount",
    "income",
    "sales",
    "profit",
    "dollar",
    "usd",
    "fee",
    "budget",
    "wage",
    "pay",
    "spend",
    "spending",
    "payment",
)

PERCENT_KEYWORDS = (
    "percent",
    "percentage",
    "pct",
    "rate",
    "ratio",
    "share",
    "margin",
)


def _normalize_column_name(column_name: str) -> str:
    """Normalize a column name for keyword matching."""
    return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def is_currency_like_column(column_name: str | None) -> bool:
    """Return True when a column name suggests currency-like values."""
    if not column_name:
        return False
    normalized = _normalize_column_name(column_name)
    return any(keyword in normalized for keyword in CURRENCY_KEYWORDS)


def is_percent_column(column_name: str | None) -> bool:
    """Return True when a column name suggests percentage values."""
    if not column_name:
        return False
    normalized = _normalize_column_name(column_name)
    return any(keyword in normalized for keyword in PERCENT_KEYWORDS)


def _to_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def format_number(value: Any, column_name: str | None = None) -> str:
    """
    Format a numeric value for human-readable reports.

    Avoids scientific notation except for extremely small non-zero values.
    """
    number = _to_float(value)
    if number is None:
        return "N/A"

    if column_name and is_percent_column(column_name):
        if abs(number) <= 1:
            return f"{number:.1%}"
        return f"{number:,.1f}%"

    if column_name and is_currency_like_column(column_name):
        if abs(number - round(number)) < 1e-9 or abs(number) >= 100:
            return f"{number:,.0f}"
        return f"{number:,.2f}"

    abs_number = abs(number)
    if abs_number != 0 and abs_number < 1e-6:
        return f"{number:.2e}"

    if abs(number - round(number)) < 1e-9 or abs_number >= 1000:
        return f"{number:,.0f}"

    if abs_number >= 1:
        formatted = f"{number:,.2f}".rstrip("0").rstrip(".")
        return formatted

    formatted = f"{number:,.4f}".rstrip("0").rstrip(".")
    return formatted or "0"


def format_correlation(value: Any) -> str:
    """Format a correlation coefficient."""
    number = _to_float(value)
    if number is None:
        return "N/A"
    return f"{number:.2f}"


def format_value_range(column_name: str, minimum: Any, maximum: Any) -> str:
    """Format a readable min-max range for a numeric column."""
    low = format_number(minimum, column_name)
    high = format_number(maximum, column_name)
    return f"{column_name} ranges from {low} to {high}"


def apply_axis_format(ax: Any, column_name: str | None, *, axis: str = "y") -> None:
    """Apply readable tick formatting to a matplotlib axis."""
    from matplotlib.ticker import FuncFormatter

    target_axis = ax.yaxis if axis == "y" else ax.xaxis
    target_axis.set_major_formatter(
        FuncFormatter(lambda value, _position: format_number(value, column_name))
    )


def apply_figure_margins(fig: Any, *, bottom: float = 0.14, top: float = 0.90) -> None:
    """Apply balanced figure margins before export."""
    fig.subplots_adjust(bottom=bottom, top=top, left=0.10, right=0.97)
