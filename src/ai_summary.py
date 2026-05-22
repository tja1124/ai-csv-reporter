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
    chart_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a safe, aggregated context object for executive summary generation."""
    column_summaries = []
    for column, info in column_info.items():
        summary: dict[str, Any] = {
            "name": column,
            "category": info["category"],
            "dtype": info["dtype"],
            "unique_values": info["unique_values"],
        }
        if column in numeric_summary:
            summary["numeric_stats"] = {
                key: value for key, value in numeric_summary[column].items() if value is not None
            }
        if column in categorical_summary:
            summary["top_values"] = categorical_summary[column]["top_values"]
        if column in missing_values["columns_with_missing"]:
            summary["missing"] = missing_values["columns_with_missing"][column]
        column_summaries.append(summary)

    context = {
        "source_file": csv_name,
        "overview": {
            "row_count": overview["row_count"],
            "column_count": overview["column_count"],
            "memory_usage_mb": overview["memory_usage_mb"],
        },
        "quality_score": report_insights["quality_score"],
        "sample_size_caution": report_insights.get("sample_size_caution"),
        "correlation_highlights": report_insights["correlation_highlights"],
        "outlier_summary": report_insights["outlier_summary"],
        "chart_highlights": [
            {
                "title": chart["title"],
                "caption": chart["caption"],
                "insight_label": chart.get("insight_label", chart["caption"]),
                "reason_selected": chart["reason_selected"],
            }
            for chart in (chart_metadata or [])
        ],
        "data_warnings": [
            warning
            for warning in report_insights["data_warnings"]
            if not str(warning).startswith("Sample Size Caution")
        ],
    }

    logger.info("Built executive summary context for %s", csv_name)
    return context


def get_fallback_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic executive summary when AI is unavailable."""
    overview = context["overview"]
    quality = context["quality_score"]
    row_count = overview["row_count"]
    chart_highlights = context.get("chart_highlights", [])

    lead_parts: list[str] = []
    if row_count < 30:
        lead_parts.append(
            f"This is an early read of a compact {row_count:,}-record dataset"
        )
    else:
        lead_parts.append(f"This brief reviews a {row_count:,}-record dataset")

    if quality["rating"] in {"Fair", "Poor"}:
        lead_parts.append(f"with {quality['rating'].lower()} data quality ({quality['score']}/100)")
    else:
        lead_parts.append(f"with {quality['rating'].lower()} overall data quality")

    if chart_highlights:
        lead_parts.append(f"and highlights {len(chart_highlights)} visual comparison(s) worth reviewing first")
    overview_text = ", ".join(lead_parts) + "."

    takeaways: list[str] = []
    for chart in chart_highlights[:2]:
        insight = chart.get("insight_label") or chart.get("caption", "")
        takeaways.append(
            f"The strongest visual signal is in {chart['title']}: {insight}"
        )

    correlation = context.get("correlation_highlights", {})
    strongest = correlation.get("strongest_positive") or correlation.get("strongest_negative")
    if strongest and len(takeaways) < 3:
        pair = strongest
        takeaways.append(
            f"The most notable numeric pattern is between {pair['column_a']} and {pair['column_b']}, "
            f"which may be worth validating on a larger sample."
        )

    if context["data_warnings"] and len(takeaways) < 3:
        takeaways.append(f"Data quality note: {context['data_warnings'][0]}")

    if not takeaways:
        takeaways = [
            "No major anomalies stood out in the initial review.",
            "Detailed statistics and charts in this report provide the best next layer of review.",
        ]

    caution = context.get("sample_size_caution")
    if not caution and row_count < 30:
        caution = "Findings should be treated as directional because the sample size is limited."

    logger.info("Generated deterministic fallback executive summary")

    return {
        "title": "Executive Summary",
        "overview": overview_text,
        "takeaways": takeaways[:4],
        "caution": caution,
        "source": "fallback",
    }


def _build_prompt(context: dict[str, Any]) -> str:
    """Create the OpenAI prompt from safe aggregated metadata."""
    context_json = json.dumps(context, indent=2, default=str)

    return f"""You are a senior business analyst writing an executive summary for a CSV analytics brief.

Use ONLY the aggregated metadata below. Do not invent facts.

Respond with valid JSON:
{{
  "overview": "2-3 sentence executive overview",
  "takeaways": ["2 to 4 concise bullets"],
  "caution": "one limitation sentence or null"
}}

Writing rules:
- Sound like a business analyst, not a generic AI assistant
- Focus on what is unusual, useful, or decision-relevant
- Mention sample size limitations naturally when row_count is under 30
- Do not repeat chart captions verbatim
- Do not restate obvious facts like row count and column count unless relevant
- Avoid filler phrases like "this report analyzes" or "the dataset contains"
- Use cautious language: appears, may, suggests, likely
- Do not overclaim causation or certainty

Aggregated metadata:
{context_json}
"""


def _parse_ai_response(content: str) -> dict[str, Any] | None:
    """Parse the JSON response from OpenAI into a summary dictionary."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
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

    cleaned_takeaways = [str(item).strip() for item in takeaways if str(item).strip()][:4]
    if len(cleaned_takeaways) < 2:
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
    """Generate an executive summary using OpenAI from aggregated report metadata."""
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
                        "You write concise, analyst-style executive summaries for data reports. "
                        "Respond with JSON only."
                    ),
                },
                {"role": "user", "content": _build_prompt(context)},
            ],
            temperature=0.25,
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
