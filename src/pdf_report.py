"""Generate PDF reports from analysis results."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.column_utils import get_groupable_categorical_columns, is_id_like_column
from src.config import COLORS, REPORT_THEME
from src.format_utils import format_number

PDF_COLORS = {
    "primary": colors.HexColor(REPORT_THEME["primary"]),
    "secondary": colors.HexColor(COLORS["secondary"]),
    "text": colors.HexColor(COLORS["text"]),
    "muted": colors.HexColor(REPORT_THEME["muted"]),
    "grid": colors.HexColor(REPORT_THEME["divider"]),
    "surface": colors.HexColor(REPORT_THEME["surface"]),
    "background": colors.HexColor(REPORT_THEME["background"]),
    "white": colors.white,
}

APPENDIX_NUMERIC_STATS = ("mean", "std", "min", "50%", "max")


def _build_styles() -> dict[str, ParagraphStyle]:
    """Create reusable paragraph styles for the report."""
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=REPORT_THEME["title_size"],
            leading=REPORT_THEME["title_size"] + 4,
            textColor=PDF_COLORS["primary"],
            spaceAfter=4,
        ),
        "brand": ParagraphStyle(
            "BrandLabel",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=PDF_COLORS["white"],
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=PDF_COLORS["muted"],
            spaceAfter=3,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=REPORT_THEME["section_size"],
            leading=REPORT_THEME["section_size"] + 4,
            textColor=PDF_COLORS["primary"],
            spaceBefore=2,
            spaceAfter=4,
        ),
        "subsection": ParagraphStyle(
            "SubsectionHeading",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=REPORT_THEME["subsection_size"],
            leading=REPORT_THEME["subsection_size"] + 3,
            textColor=PDF_COLORS["text"],
            spaceBefore=1,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=REPORT_THEME["body_size"],
            leading=REPORT_THEME["body_size"] + 4,
            textColor=PDF_COLORS["text"],
            spaceAfter=4,
        ),
        "body_compact": ParagraphStyle(
            "ReportBodyCompact",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=REPORT_THEME["body_size"],
            leading=REPORT_THEME["body_size"] + 3,
            textColor=PDF_COLORS["text"],
            spaceAfter=2,
        ),
        "insight": ParagraphStyle(
            "ChartInsight",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=REPORT_THEME["caption_size"],
            leading=REPORT_THEME["caption_size"] + 3,
            textColor=PDF_COLORS["muted"],
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=PDF_COLORS["muted"],
            alignment=TA_CENTER,
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=REPORT_THEME["kpi_value_size"],
            leading=REPORT_THEME["kpi_value_size"] + 2,
            textColor=PDF_COLORS["primary"],
            alignment=TA_CENTER,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=REPORT_THEME["body_size"],
            leading=REPORT_THEME["body_size"] + 3,
            textColor=PDF_COLORS["text"],
            leftIndent=12,
            spaceAfter=2,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=REPORT_THEME["body_size"],
            leading=REPORT_THEME["body_size"] + 4,
            textColor=PDF_COLORS["text"],
            spaceAfter=2,
        ),
        "appendix_note": ParagraphStyle(
            "AppendixNote",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            textColor=PDF_COLORS["muted"],
            spaceAfter=6,
        ),
    }


def _divider(width: float) -> Table:
    """Render a subtle section divider line."""
    line = Table([[""]], colWidths=[width], rowHeights=[1])
    line.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.75, PDF_COLORS["grid"])]))
    return line


def _shaded_panel(content: list[Any], width: float, *, border: bool = True) -> Table:
    """Wrap flowables in a lightly shaded panel."""
    panel = Table([[content]], colWidths=[width])
    style_commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, -1), PDF_COLORS["surface"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]
    if border:
        style_commands.append(("BOX", (0, 0), (-1, -1), 0.5, PDF_COLORS["grid"]))
    panel.setStyle(TableStyle(style_commands))
    return panel


def _callout_box(text: str, styles: dict[str, ParagraphStyle], width: float) -> Table:
    """Render a highlighted callout box."""
    return _shaded_panel([Paragraph(text, styles["callout"])], width, border=True)


def _mini_section_header(title: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Compact section label without extra divider spacing."""
    return [Paragraph(title, styles["subsection"]), Spacer(1, 0.03 * inch)]


def _build_section_header(title: str, styles: dict[str, ParagraphStyle], width: float) -> list[Any]:
    """Create a section heading with divider."""
    return [
        Spacer(1, 0.04 * inch),
        Paragraph(title, styles["section"]),
        Spacer(1, 0.02 * inch),
        _divider(width),
        Spacer(1, 0.04 * inch),
    ]


def _build_table(
    data: list[list[str]],
    col_widths: list[float] | None = None,
    *,
    compact: bool = False,
) -> Table:
    """Create a styled table for the PDF report."""
    table = Table(data, colWidths=col_widths, repeatRows=1)
    padding = 5 if compact else 6
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PDF_COLORS["surface"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), PDF_COLORS["primary"]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5 if compact else 9),
                ("TEXTCOLOR", (0, 1), (-1, -1), PDF_COLORS["text"]),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5 if compact else 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, 0), 1, PDF_COLORS["secondary"]),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, PDF_COLORS["grid"]),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PDF_COLORS["white"], PDF_COLORS["surface"]]),
            ]
        )
    )
    return table


def _build_bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Render a clean bullet list."""
    return [Paragraph(f"• {item}", styles["bullet"]) for item in items]


def _normalize_text(text: str) -> str:
    """Normalize text for simple duplicate detection."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def _texts_overlap(left: str, right: str) -> bool:
    """Return True when two strings likely describe the same point."""
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return False
    if left_norm in right_norm or right_norm in left_norm:
        return True

    left_words = set(re.findall(r"[a-z0-9]+", left_norm))
    right_words = set(re.findall(r"[a-z0-9]+", right_norm))
    if not left_words or not right_words:
        return False
    overlap = len(left_words & right_words) / min(len(left_words), len(right_words))
    return overlap >= 0.55


def _dedupe_findings(
    findings: list[str],
    executive_summary: dict[str, Any] | None,
) -> list[str]:
    """Remove findings that repeat executive summary content."""
    if not executive_summary:
        return findings

    compare_against = [executive_summary.get("overview", "")]
    compare_against.extend(executive_summary.get("takeaways", []))
    compare_against.extend(executive_summary.get("caution", "") or "")

    filtered = [
        finding
        for finding in findings
        if not any(_texts_overlap(finding, other) for other in compare_against if other)
    ]
    return filtered or findings[:2]


def _build_title_page(
    csv_name: str,
    overview: dict[str, Any],
    report_timestamp: str,
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    """Build a branded title page."""
    header = Table(
        [
            [Paragraph("AI CSV REPORTER", styles["brand"])],
            [Paragraph("Analytics Brief", styles["brand"])],
        ],
        colWidths=[width],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PDF_COLORS["primary"]),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
            ]
        )
    )

    meta_rows = [
        [Paragraph("<b>Dataset</b>", styles["body"]), Paragraph(csv_name, styles["body"])],
        [Paragraph("<b>Generated</b>", styles["body"]), Paragraph(report_timestamp, styles["body"])],
        [
            Paragraph("<b>Scope</b>", styles["body"]),
            Paragraph(
                f"{overview['row_count']:,} records · {overview['column_count']} fields",
                styles["body"],
            ),
        ],
    ]
    meta_table = Table(meta_rows, colWidths=[1.1 * inch, width - 1.1 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return [
        header,
        Spacer(1, 0.45 * inch),
        Paragraph("Business Intelligence Report", styles["title"]),
        Paragraph("Automated analytics brief with visual insights and quality signals.", styles["subtitle"]),
        Spacer(1, 0.12 * inch),
        _divider(width),
        Spacer(1, 0.12 * inch),
        meta_table,
        Spacer(1, 0.16 * inch),
        _shaded_panel(
            [
                Paragraph(
                    "This report highlights the strongest patterns, category comparisons, and data quality "
                    "notes surfaced from the dataset. Use the front summary for decisions; use the appendix "
                    "for validation.",
                    styles["body"],
                )
            ],
            width,
        ),
        PageBreak(),
    ]


def _build_kpi_cards(
    overview: dict[str, Any],
    report_insights: dict[str, Any],
    missing_values: dict[str, Any],
    chart_count: int,
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    """Build KPI-style summary cards at the top of the front matter."""
    total_cells = max(overview["row_count"] * overview["column_count"], 1)
    missing_pct = round((missing_values["total_missing"] / total_cells) * 100, 1)
    quality = report_insights["quality_score"]
    sample_note = "Small sample" if overview["row_count"] < 30 else "Adequate"

    card_width = width / 3
    cards = Table(
        [
            [
                Paragraph("Rows", styles["kpi_label"]),
                Paragraph("Columns", styles["kpi_label"]),
                Paragraph("Quality", styles["kpi_label"]),
            ],
            [
                Paragraph(f"{overview['row_count']:,}", styles["kpi_value"]),
                Paragraph(f"{overview['column_count']}", styles["kpi_value"]),
                Paragraph(f"{quality['score']}/100", styles["kpi_value"]),
            ],
            [
                Paragraph("Missing", styles["kpi_label"]),
                Paragraph("Charts", styles["kpi_label"]),
                Paragraph("Sample", styles["kpi_label"]),
            ],
            [
                Paragraph(f"{missing_pct}%", styles["kpi_value"]),
                Paragraph(str(chart_count), styles["kpi_value"]),
                Paragraph(sample_note, styles["kpi_value"]),
            ],
        ],
        colWidths=[card_width, card_width, card_width],
    )
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PDF_COLORS["surface"]),
                ("BACKGROUND", (0, 2), (-1, 2), PDF_COLORS["surface"]),
                ("BOX", (0, 0), (-1, -1), 0.75, PDF_COLORS["grid"]),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, PDF_COLORS["grid"]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    section: list[Any] = []
    section.extend(_build_section_header("At a Glance", styles, width))
    section.append(cards)
    section.append(Spacer(1, 0.1 * inch))
    return section


def _build_front_matter_summary(
    executive_summary: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    """Build executive summary and takeaways with breathing room."""
    section: list[Any] = []
    section.extend(_mini_section_header(executive_summary.get("title", "Executive Summary"), styles))
    section.append(_callout_box(executive_summary["overview"], styles, width))
    section.append(Spacer(1, 0.08 * inch))

    takeaways = executive_summary.get("takeaways")
    if takeaways:
        section.extend(_mini_section_header("Analyst Takeaways", styles))
        section.extend(_build_bullet_list(takeaways[:4], styles))

    if executive_summary.get("caution"):
        section.append(Spacer(1, 0.05 * inch))
        section.append(
            _callout_box(f"<b>Caution:</b> {executive_summary['caution']}", styles, width)
        )

    section.append(Spacer(1, 0.06 * inch))
    return section


def _build_findings_section(
    report_insights: dict[str, Any],
    executive_summary: dict[str, Any] | None,
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    """Build findings that complement rather than repeat the executive summary."""
    section: list[Any] = []
    section.extend(_build_section_header("Most Important Findings", styles, width))

    findings = report_insights.get("important_findings") or report_insights.get("key_insights", [])
    display_findings = [
        finding
        for finding in findings
        if not str(finding).startswith("Sample Size Caution")
    ]
    display_findings = _dedupe_findings(display_findings, executive_summary)

    if display_findings:
        section.extend(_build_bullet_list(display_findings[:4], styles))
    else:
        section.append(
            Paragraph(
                "No additional findings beyond the executive summary were flagged in this review.",
                styles["body_compact"],
            )
        )
    section.append(Spacer(1, 0.04 * inch))
    return section


def _build_data_quality_section(
    report_insights: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    """Build a concise data quality section for later pages."""
    section: list[Any] = []
    section.extend(_build_section_header("Data Quality Notes", styles, width))

    quality = report_insights["quality_score"]
    section.append(
        Paragraph(
            f"Overall quality score: <b>{quality['score']}/100</b> ({quality['rating']})",
            styles["body_compact"],
        )
    )

    warnings = [
        warning
        for warning in report_insights["data_warnings"]
        if not str(warning).startswith("Sample Size Caution")
    ]
    if warnings:
        section.append(Spacer(1, 0.03 * inch))
        section.extend(_build_bullet_list(warnings[:4], styles))

    chart_skip_reasons = report_insights.get("chart_skip_reasons", [])
    if chart_skip_reasons:
        section.append(Spacer(1, 0.04 * inch))
        section.extend(_mini_section_header("Visualization Notes", styles))
        section.extend(_build_bullet_list(chart_skip_reasons[:3], styles))

    return section


def _scaled_chart_image(chart_path: Path, max_width: float, max_height: float) -> Image:
    """Embed a chart image while preserving aspect ratio within page bounds."""
    image = Image(str(chart_path))
    width_scale = max_width / float(image.imageWidth)
    height_scale = max_height / float(image.imageHeight)
    scale = min(width_scale, height_scale)

    image.drawWidth = float(image.imageWidth) * scale
    image.drawHeight = float(image.imageHeight) * scale
    image.hAlign = "CENTER"
    return image


def _insight_paragraph(chart: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Paragraph:
    """Render a labeled chart insight line."""
    insight = chart.get("insight_label") or chart.get("caption", "")
    return Paragraph(f"<b>Insight:</b> {insight}", styles["insight"])


def _compact_chart_height(chart: dict[str, Any]) -> float:
    """Return a PDF image height tuned to chart type."""
    if chart.get("chart_type") == "boxplot":
        return 3.15 * inch
    if chart.get("chart_type") == "bar":
        return 2.75 * inch
    return 2.85 * inch


def _should_pair_charts(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Only pair simple bar charts; boxplots stack full-width for alignment."""
    return left.get("chart_type") == "bar" and right.get("chart_type") == "bar"


def _build_hero_chart_block(
    chart: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> KeepTogether:
    """Build a large hero chart block for the strongest relationship view."""
    return KeepTogether(
        [
            Paragraph(chart["title"], styles["subsection"]),
            _scaled_chart_image(chart["path"], width, 4.6 * inch),
            _insight_paragraph(chart, styles),
        ]
    )


def _build_stacked_chart_block(
    chart: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> KeepTogether:
    """Build a full-width stacked compact chart block."""
    return KeepTogether(
        [
            Paragraph(chart["title"], styles["subsection"]),
            _scaled_chart_image(chart["path"], width, _compact_chart_height(chart)),
            _insight_paragraph(chart, styles),
        ]
    )


def _build_chart_cell_table(
    chart: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    cell_width: float,
    max_height: float,
) -> Table:
    """Build one chart cell as a nested table (safe inside side-by-side rows)."""
    cell = Table(
        [
            [Paragraph(chart["title"], styles["subsection"])],
            [_scaled_chart_image(chart["path"], cell_width - 6, max_height)],
            [_insight_paragraph(chart, styles)],
        ],
        colWidths=[cell_width],
    )
    cell.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return cell


def _build_chart_section(
    chart_metadata: list[dict[str, Any]],
    styles: dict[str, ParagraphStyle],
    width: float,
) -> list[Any]:
    """Lay out hero scatter first, then stack or pair compact charts efficiently."""
    section: list[Any] = []
    hero_charts = [chart for chart in chart_metadata if chart.get("layout") == "hero"]
    compact_charts = [chart for chart in chart_metadata if chart.get("layout") != "hero"]

    for index, chart in enumerate(hero_charts):
        section.append(_build_hero_chart_block(chart, styles, width))
        if index < len(hero_charts) - 1 or compact_charts:
            section.append(Spacer(1, 0.02 * inch))

    gutter = 0.12 * inch
    half_width = (width - gutter) / 2
    index = 0
    while index < len(compact_charts):
        left = compact_charts[index]
        right = compact_charts[index + 1] if index + 1 < len(compact_charts) else None

        if right and _should_pair_charts(left, right):
            pair_height = min(_compact_chart_height(left), _compact_chart_height(right))
            row = Table(
                [
                    [
                        _build_chart_cell_table(left, styles, half_width, pair_height),
                        _build_chart_cell_table(right, styles, half_width, pair_height),
                    ]
                ],
                colWidths=[half_width, half_width],
                hAlign="LEFT",
            )
            row.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            section.append(row)
            section.append(Spacer(1, 0.02 * inch))
            index += 2
        else:
            section.append(_build_stacked_chart_block(left, styles, width))
            section.append(Spacer(1, 0.02 * inch))
            index += 1

    return section


def _filter_appendix_categorical(
    categorical_summary: dict[str, dict[str, Any]],
    row_count: int,
    df: Any,
) -> dict[str, dict[str, Any]]:
    """Keep only useful, groupable categorical summaries in the appendix."""
    allowed = set(get_groupable_categorical_columns(df))

    filtered: dict[str, dict[str, Any]] = {}
    for column, stats in categorical_summary.items():
        if column not in allowed:
            continue
        if is_id_like_column(column, df[column], row_count):
            continue
        filtered[column] = stats
    return filtered


def _build_numeric_appendix_table(
    numeric_summary: dict[str, dict[str, float | None]],
    width: float,
) -> Table | None:
    """Render numeric summaries as one compact multi-column table."""
    if not numeric_summary:
        return None

    columns = list(numeric_summary.keys())
    header = ["Metric", *columns]
    rows = [header]

    for stat_key in APPENDIX_NUMERIC_STATS:
        label = "Median" if stat_key == "50%" else stat_key.replace("_", " ").title()
        row = [label]
        for column in columns:
            value = numeric_summary[column].get(stat_key)
            row.append("N/A" if value is None else format_number(value, column))
        rows.append(row)

    col_width = max(0.9 * inch, (width - 1.0 * inch) / max(len(columns), 1))
    return _build_table(rows, [1.0 * inch, *[col_width] * len(columns)], compact=True)


def _build_appendix_section(
    overview: dict[str, Any],
    missing_values: dict[str, Any],
    numeric_summary: dict[str, dict[str, float | None]],
    categorical_summary: dict[str, dict[str, Any]],
    styles: dict[str, ParagraphStyle],
    width: float,
    df: Any,
) -> list[Any]:
    """Build a cleaner, optional-feeling statistics appendix."""
    section: list[Any] = []
    section.extend(_build_section_header("Reference Appendix", styles, width))
    section.append(
        Paragraph(
            "Detailed statistics for validation and deeper review. Skim only if you need to "
            "confirm a specific metric or category breakdown.",
            styles["appendix_note"],
        )
    )

    overview_data = [
        ["Metric", "Value"],
        ["Total Rows", f"{overview['row_count']:,}"],
        ["Total Columns", f"{overview['column_count']:,}"],
        ["Memory Usage", f"{overview['memory_usage_mb']} MB"],
    ]
    section.append(_build_table(overview_data, [1.4 * inch, width - 1.4 * inch], compact=True))
    section.append(Spacer(1, 0.08 * inch))

    section.extend(_mini_section_header("Missing Values", styles))
    if missing_values["total_missing"] == 0:
        section.append(Paragraph("No missing values detected.", styles["body_compact"]))
    else:
        missing_data = [["Column", "Missing", "%"]]
        for column, info in missing_values["columns_with_missing"].items():
            missing_data.append([column, f"{info['count']:,}", f"{info['percentage']}%"])
        section.append(_build_table(missing_data, [2.6 * inch, 1.2 * inch, 1.0 * inch], compact=True))
    section.append(Spacer(1, 0.06 * inch))

    section.extend(_mini_section_header("Numeric Summary", styles))
    numeric_table = _build_numeric_appendix_table(numeric_summary, width)
    if numeric_table is None:
        section.append(Paragraph("No numeric columns found.", styles["body_compact"]))
    else:
        section.append(numeric_table)
    section.append(Spacer(1, 0.06 * inch))

    filtered_categorical = _filter_appendix_categorical(
        categorical_summary,
        overview["row_count"],
        df,
    )
    section.extend(_mini_section_header("Categorical Breakdown", styles))
    if not filtered_categorical:
        section.append(Paragraph("No groupable categorical fields to summarize.", styles["body_compact"]))
    else:
        for column, stats in filtered_categorical.items():
            top_values_data = [["Value", "Count"]]
            for value, count in list(stats["top_values"].items())[:6]:
                top_values_data.append([value, f"{count:,}"])
            section.append(Paragraph(column, styles["body_compact"]))
            section.append(_build_table(top_values_data, [2.8 * inch, 1.0 * inch], compact=True))
            section.append(Spacer(1, 0.03 * inch))

    return section


def _draw_page_footer(canvas: Any, doc: Any, generated_at: str) -> None:
    """Draw a subtle footer with timestamp and page numbers."""
    canvas.saveState()
    canvas.setStrokeColor(PDF_COLORS["grid"])
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.65 * inch, doc.pagesize[0] - doc.rightMargin, 0.65 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(PDF_COLORS["muted"])
    canvas.drawString(doc.leftMargin, 0.45 * inch, f"Generated {generated_at}")
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.45 * inch, "Analytics Brief")
    canvas.drawRightString(
        doc.pagesize[0] - doc.rightMargin,
        0.45 * inch,
        f"Page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


def create_pdf_report(
    csv_path: str | Path,
    overview: dict[str, Any],
    missing_values: dict[str, Any],
    numeric_summary: dict[str, dict[str, float | None]],
    categorical_summary: dict[str, dict[str, Any]],
    chart_metadata: list[dict[str, Any]],
    output_path: str | Path,
    report_insights: dict[str, Any] | None = None,
    executive_summary: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
    df: Any | None = None,
) -> Path:
    """Build a PDF analytics brief from analysis results and chart metadata."""
    if df is None:
        raise ValueError("df is required for appendix filtering")
    pdf_path = Path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    report_time = generated_at or datetime.now()
    report_timestamp = report_time.strftime("%B %d, %Y at %H:%M:%S")
    footer_timestamp = report_time.strftime("%Y-%m-%d %H:%M")
    csv_name = Path(csv_path).name
    margin = REPORT_THEME["page_margin_in"] * inch

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=0.7 * inch,
        bottomMargin=0.8 * inch,
    )

    styles = _build_styles()
    story: list[Any] = []
    content_width = doc.width

    story.extend(_build_title_page(csv_name, overview, report_timestamp, styles, content_width))

    if report_insights:
        story.extend(
            _build_kpi_cards(
                overview,
                report_insights,
                missing_values,
                len(chart_metadata),
                styles,
                content_width,
            )
        )

    if executive_summary:
        story.extend(_build_front_matter_summary(executive_summary, styles, content_width))

    if report_insights and executive_summary:
        story.append(PageBreak())
        story.extend(_build_findings_section(report_insights, executive_summary, styles, content_width))

    story.append(PageBreak())
    story.extend(_build_section_header("Key Visual Insights", styles, content_width))

    if not chart_metadata:
        story.append(Paragraph("No charts were generated for this dataset.", styles["body"]))
    else:
        story.extend(_build_chart_section(chart_metadata, styles, content_width))

    if report_insights:
        story.append(Spacer(1, 0.14 * inch))
        story.extend(_build_data_quality_section(report_insights, styles, content_width))

    story.append(Spacer(1, 0.12 * inch))
    story.extend(
        _build_appendix_section(
            overview,
            missing_values,
            numeric_summary,
            categorical_summary,
            styles,
            content_width,
            df=df,
        )
    )

    doc.build(
        story,
        onFirstPage=lambda canvas, doc: _draw_page_footer(canvas, doc, footer_timestamp),
        onLaterPages=lambda canvas, doc: _draw_page_footer(canvas, doc, footer_timestamp),
    )
    return pdf_path
