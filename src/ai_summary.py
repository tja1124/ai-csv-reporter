"""Optional OpenAI executive summary based on safe aggregated report metadata."""

import json
import os
import re
from typing import Any

from src.config import OPENAI_MODEL
from src.logger import get_logger

logger = get_logger()


def build_summary_context(
    csv_name: str,
    overview: dict[str, Any],
    column_info: dict[str, dict[str, Any]],
    missing_values: dict[str, Any],
    numeric_summary: dict[str, dict[str, float | None]],
    categorical_summary: dict[str, dict[str, Any]],
    report_insights: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a safe, aggregated context object for executive summary generation.

    Only summary statistics and report metadata are included. Raw CSV rows
    are never added to this context.

    Returns:
        A dictionary of aggregated metadata suitable for AI or fallback summaries.
    """
    column_summaries = []
    for column, info in column_info.items():
        summary: dict[str, Any] = {
            "name": column,
            "category": info["category"],
            "dtype": info["dtype"],
            "unique_values": info["unique_values"],
        }
        if column in numeric_summary:
            stats = numeric_summary[column]
            summary["numeric_stats"] = {
                key: value for key, value in stats.items() if value is not None
            }
        if column in categorical_summary:
            cat_stats = categorical_summary[column]
            summary["top_values"] = cat_stats["top_values"]
        if column in missing_values["columns_with_missing"]:
            summary["missing"] = missing_values["columns_with_missing"][column]
        column_summaries.append(summary)

    context = {
        "source_file": csv_name,
        "overview": {
            "row_count": overview["row_count"],
            "column_count": overview["column_count"],
            "memory_usage_mb": overview["memory_usage_mb"],
            "columns": overview["columns"],
        },
        "column_summaries": column_summaries,
        "quality_score": report_insights["quality_score"],
        "key_insights": report_insights["key_insights"],
        "data_warnings": report_insights["data_warnings"],
        "outlier_summary": report_insights["outlier_summary"],
        "correlation_highlights": report_insights["correlation_highlights"],
        "chart_skip_reasons": report_insights.get("chart_skip_reasons", []),
    }

    logger.info("Built executive summary context for %s", csv_name)
    return context


def get_fallback_summary(context: dict[str, Any]) -> dict[str, Any]:
    """
    Build a deterministic executive summary when AI is unavailable.

    Returns:
        Executive summary with overview paragraph, takeaways, and optional caution.
    """
    overview = context["overview"]
    quality = context["quality_score"]
    row_count = overview["row_count"]
    column_count = overview["column_count"]

    overview_text = (
        f"This report analyzes '{context['source_file']}', a dataset with "
        f"{row_count:,} rows and {column_count} columns. "
        f"The dataset quality score is {quality['score']}/100 ({quality['rating']})."
    )

    takeaways = context["key_insights"][:5]
    if not takeaways:
        takeaways = [
            f"The dataset contains {row_count:,} rows across {column_count} columns.",
            f"Overall data quality is rated {quality['rating']}.",
        ]

    caution = None
    warnings = [
        warning
        for warning in context["data_warnings"]
        if warning != "No data warnings detected."
    ]
    if warnings:
        caution = warnings[0]
    elif context["chart_skip_reasons"]:
        caution = (
            "Some columns were excluded from charts because they were not suitable "
            "for visualization."
        )

    logger.info("Generated deterministic fallback executive summary")

    return {
        "title": "Executive Summary",
        "overview": overview_text,
        "takeaways": takeaways,
        "caution": caution,
        "source": "fallback",
    }


def _build_prompt(context: dict[str, Any]) -> str:
    """Create the OpenAI prompt from safe aggregated metadata."""
    context_json = json.dumps(context, indent=2, default=str)

    return f"""You are a professional data analyst writing an executive summary for a CSV analytics report.

Use ONLY the aggregated metadata below. Do not invent facts, rows, or statistics that are not supported by the data.

Respond with valid JSON in this exact shape:
{{
  "overview": "one short professional paragraph",
  "takeaways": ["3 to 5 concise bullet points"],
  "caution": "one caution or limitation sentence, or null if none is relevant"
}}

Requirements:
- Professional business tone
- No unsupported claims
- Reference only provided metrics and insights
- Keep the overview to 2-4 sentences
- Provide 3-5 takeaway bullets

Aggregated report metadata:
{context_json}
"""


def _parse_ai_response(content: str) -> dict[str, Any] | None:
    """Parse the JSON response from OpenAI into a summary dictionary."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Some models wrap JSON in markdown fences; try extracting it.
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    overview = str(parsed.get("overview", "")).strip()
    takeaways = parsed.get("takeaways", [])
    caution = parsed.get("caution")

    if not overview or not isinstance(takeaways, list):
        return None

    cleaned_takeaways = [str(item).strip() for item in takeaways if str(item).strip()][:5]
    if len(cleaned_takeaways) < 3:
        return None

    caution_text = None
    if caution and str(caution).strip().lower() not in {"null", "none", ""}:
        caution_text = str(caution).strip()

    return {
        "title": "AI-Assisted Executive Summary",
        "overview": overview,
        "takeaways": cleaned_takeaways,
        "caution": caution_text,
        "source": "ai",
    }


def generate_ai_executive_summary(context: dict[str, Any]) -> dict[str, Any] | None:
    """
    Generate an executive summary using OpenAI from aggregated report metadata.

    Returns:
        AI-generated summary dict, or None if the API key is missing or the
        request fails. Callers should use get_fallback_summary() when None.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info("AI summary skipped: OPENAI_API_KEY is not set")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("AI summary skipped: openai package is not installed")
        return None

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise executive summaries for data reports. "
                        "Respond with JSON only."
                    ),
                },
                {"role": "user", "content": _build_prompt(context)},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            logger.warning("AI summary failed: empty response from OpenAI")
            return None

        summary = _parse_ai_response(content)
        if summary is None:
            logger.warning("AI summary failed: could not parse OpenAI response")
            return None

        logger.info("AI executive summary generated successfully")
        return summary

    except Exception as exc:
        logger.warning("AI summary failed: %s", exc)
        return None
