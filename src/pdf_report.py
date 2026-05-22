"""Generate PDF reports from analysis results."""

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import COLORS

PDF_COLORS = {
    "primary": colors.HexColor(COLORS["primary"]),
    "secondary": colors.HexColor(COLORS["secondary"]),
    "text": colors.HexColor(COLORS["text"]),
    "muted": colors.HexColor(COLORS["muted"]),
    "grid": colors.HexColor(COLORS["grid"]),
    "surface": colors.HexColor(COLORS["surface"]),
    "white": colors.white,
}


def _build_styles() -> dict[str, ParagraphStyle]:
    """Create reusable paragraph styles for the report."""
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=PDF_COLORS["primary"],
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=PDF_COLORS["muted"],
            spaceAfter=4,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=PDF_COLORS["primary"],
            spaceBefore=10,
            spaceAfter=8,
        ),
        "subsection": ParagraphStyle(
            "SubsectionHeading",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=PDF_COLORS["text"],
            spaceBefore=6,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=PDF_COLORS["text"],
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "ChartCaption",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=PDF_COLORS["muted"],
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=PDF_COLORS["text"],
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=4,
        ),
    }


def _build_section_header(title: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Create a section heading with consistent spacing."""
    return [
        Spacer(1, 0.12 * inch),
        Paragraph(title, styles["section"]),
        Spacer(1, 0.06 * inch),
    ]


def _build_table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    """Create a styled table for the PDF report."""
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PDF_COLORS["surface"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), PDF_COLORS["primary"]),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("TEXTCOLOR", (0, 1), (-1, -1), PDF_COLORS["text"]),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("LINEBELOW", (0, 0), (-1, 0), 1, PDF_COLORS["secondary"]),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, PDF_COLORS["grid"]),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PDF_COLORS["white"], PDF_COLORS["surface"]]),
            ]
        )
    )
    return table


def _build_bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> list[Any]:
    """Render a clean bullet list for insights and warnings."""
    return [
        Paragraph(f"• {item}", styles["bullet"])
        for item in items
    ]


def _build_quality_score_block(
    quality_score: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Render the dataset quality score and deduction bullets."""
    blocks: list[Any] = [
        Paragraph(
            f"<b>{quality_score['score']}/100</b> — {quality_score['rating']}",
            styles["body"],
        ),
        Spacer(1, 0.06 * inch),
    ]
    blocks.extend(_build_bullet_list(quality_score["deductions"], styles))
    return blocks


def _build_executive_summary_section(
    executive_summary: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Build the executive summary section near the top of the PDF."""
    section: list[Any] = []
    section.extend(_build_section_header(executive_summary["title"], styles))
    section.append(Paragraph(executive_summary["overview"], styles["body"]))
    section.append(Spacer(1, 0.08 * inch))

    section.append(Paragraph("Key Takeaways", styles["subsection"]))
    section.extend(_build_bullet_list(executive_summary["takeaways"], styles))

    if executive_summary.get("caution"):
        section.append(Spacer(1, 0.08 * inch))
        section.append(Paragraph("Caution", styles["subsection"]))
        section.append(Paragraph(executive_summary["caution"], styles["body"]))

    section.append(Spacer(1, 0.14 * inch))
    return section


def _build_intelligence_section(
    report_insights: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Build the report intelligence section near the top of the PDF."""
    section: list[Any] = []
    section.extend(_build_section_header("Report Intelligence", styles))

    section.append(Paragraph("Dataset Quality Score", styles["subsection"]))
    section.extend(_build_quality_score_block(report_insights["quality_score"], styles))
    section.append(Spacer(1, 0.1 * inch))

    section.append(Paragraph("Key Insights", styles["subsection"]))
    section.extend(_build_bullet_list(report_insights["key_insights"], styles))
    section.append(Spacer(1, 0.1 * inch))

    section.append(Paragraph("Data Warnings", styles["subsection"]))
    section.extend(_build_bullet_list(report_insights["data_warnings"], styles))
    section.append(Spacer(1, 0.1 * inch))

    section.append(Paragraph("Outlier Summary", styles["subsection"]))
    outlier_summary = report_insights["outlier_summary"]
    if not outlier_summary:
        section.append(Paragraph("No notable outliers detected using the IQR method.", styles["body"]))
    else:
        outlier_data = [["Column", "Outlier Count", "Outlier %"]]
        for column, stats in outlier_summary.items():
            outlier_data.append([
                column,
                f"{stats['count']:,}",
                f"{stats['percentage']}%",
            ])
        section.append(_build_table(outlier_data, [2.8 * inch, 1.4 * inch, 1.3 * inch]))
    section.append(Spacer(1, 0.1 * inch))

    section.append(Paragraph("Correlation Highlights", styles["subsection"]))
    correlation = report_insights["correlation_highlights"]
    correlation_items: list[str] = []

    if correlation["strongest_positive"]:
        pair = correlation["strongest_positive"]
        correlation_items.append(
            f"Strongest positive: {pair['column_a']} & {pair['column_b']} "
            f"(r={pair['correlation']:.2f})"
        )
    if correlation["strongest_negative"]:
        pair = correlation["strongest_negative"]
        correlation_items.append(
            f"Strongest negative: {pair['column_a']} & {pair['column_b']} "
            f"(r={pair['correlation']:.2f})"
        )

    notable_pairs = correlation["notable_pairs"]
    if notable_pairs:
        for pair in notable_pairs[:5]:
            correlation_items.append(
                f"{pair['column_a']} & {pair['column_b']}: r={pair['correlation']:.2f}"
            )
    else:
        correlation_items.append(
            f"No correlations with absolute value >= 0.5 were found."
        )

    section.extend(_build_bullet_list(correlation_items, styles))

    chart_skip_reasons = report_insights.get("chart_skip_reasons", [])
    if chart_skip_reasons:
        section.append(Spacer(1, 0.08 * inch))
        section.append(Paragraph("Chart Selection Notes", styles["subsection"]))
        section.extend(_build_bullet_list(chart_skip_reasons, styles))

    section.append(Spacer(1, 0.14 * inch))
    return section


def _build_kpi_row(overview: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    """Create a compact KPI summary row for the report header."""
    kpi_data = [
        [
            Paragraph("Rows", styles["caption"]),
            Paragraph("Columns", styles["caption"]),
            Paragraph("Memory (MB)", styles["caption"]),
        ],
        [
            Paragraph(f"<b>{overview['row_count']:,}</b>", styles["body"]),
            Paragraph(f"<b>{overview['column_count']:,}</b>", styles["body"]),
            Paragraph(f"<b>{overview['memory_usage_mb']}</b>", styles["body"]),
        ],
    ]
    table = Table(kpi_data, colWidths=[2.1 * inch, 2.1 * inch, 2.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PDF_COLORS["surface"]),
                ("BOX", (0, 0), (-1, -1), 0.75, PDF_COLORS["grid"]),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, PDF_COLORS["grid"]),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _format_chart_title(chart_path: Path) -> str:
    """Convert a chart filename into a readable title."""
    name = chart_path.stem
    if name.startswith("histogram_"):
        column = name.replace("histogram_", "").replace("_", " ")
        return f"Distribution: {column.title()}"
    if name.startswith("bar_"):
        column = name.replace("bar_", "").replace("_", " ")
        return f"Top Values: {column.title()}"
    if name == "correlation_heatmap":
        return "Correlation Heatmap"
    return name.replace("_", " ").title()


def _scaled_chart_image(chart_path: Path, max_width: float, max_height: float) -> Image:
    """Embed a chart image while preserving aspect ratio within page bounds."""
    image = Image(str(chart_path))
    width_scale = max_width / float(image.imageWidth)
    height_scale = max_height / float(image.imageHeight)
    scale = min(width_scale, height_scale)

    image.drawWidth = float(image.imageWidth) * scale
    image.drawHeight = float(image.imageHeight) * scale
    return image


def _draw_page_footer(canvas: Any, doc: Any, generated_at: str) -> None:
    """Draw a subtle footer with timestamp and page numbers."""
    canvas.saveState()
    canvas.setStrokeColor(PDF_COLORS["grid"])
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.65 * inch, doc.pagesize[0] - doc.rightMargin, 0.65 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(PDF_COLORS["muted"])
    canvas.drawString(doc.leftMargin, 0.45 * inch, f"Generated {generated_at}")
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.45 * inch, "CSV Analytics Report")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.45 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def create_pdf_report(
    csv_path: str | Path,
    overview: dict[str, Any],
    missing_values: dict[str, Any],
    numeric_summary: dict[str, dict[str, float | None]],
    categorical_summary: dict[str, dict[str, Any]],
    chart_paths: list[Path],
    output_path: str | Path,
    report_insights: dict[str, Any] | None = None,
    executive_summary: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """
    Build a PDF report from analysis results and chart images.

    Args:
        csv_path: Original CSV file path used for the report title.
        overview: Dataset overview from get_dataset_overview().
        missing_values: Missing value summary from get_missing_values().
        numeric_summary: Numeric stats from get_numeric_summary().
        categorical_summary: Categorical stats from get_categorical_summary().
        chart_paths: List of chart image paths to embed.
        output_path: Destination PDF file path.
        report_insights: Phase 2 intelligence bundle from generate_report_insights().
        executive_summary: Optional Phase 3 executive summary for the report.
        generated_at: Optional report generation timestamp.

    Returns:
        Path to the generated PDF file.
    """
    pdf_path = Path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    report_time = generated_at or datetime.now()
    report_timestamp = report_time.strftime("%B %d, %Y at %H:%M:%S")
    footer_timestamp = report_time.strftime("%Y-%m-%d %H:%M")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.95 * inch,
    )

    styles = _build_styles()
    story: list[Any] = []
    csv_name = Path(csv_path).name

    story.append(Paragraph("CSV Analytics Report", styles["title"]))
    story.append(Paragraph(f"Source file: <b>{csv_name}</b>", styles["subtitle"]))
    story.append(Paragraph(f"Generated on {report_timestamp}", styles["subtitle"]))
    story.append(Spacer(1, 0.22 * inch))
    story.append(_build_kpi_row(overview, styles))
    story.append(Spacer(1, 0.28 * inch))

    if executive_summary:
        story.extend(_build_executive_summary_section(executive_summary, styles))

    if report_insights:
        story.extend(_build_intelligence_section(report_insights, styles))

    story.extend(_build_section_header("Dataset Overview", styles))
    overview_data = [
        ["Metric", "Value"],
        ["Total Rows", f"{overview['row_count']:,}"],
        ["Total Columns", f"{overview['column_count']:,}"],
        ["Memory Usage", f"{overview['memory_usage_mb']} MB"],
        ["Column Names", ", ".join(overview["columns"])],
    ]
    story.append(_build_table(overview_data, [1.6 * inch, 4.9 * inch]))
    story.append(Spacer(1, 0.18 * inch))

    story.extend(_build_section_header("Data Quality", styles))
    if missing_values["total_missing"] == 0:
        story.append(Paragraph("No missing values detected in the dataset.", styles["body"]))
    else:
        missing_data = [["Column", "Missing Count", "Missing %"]]
        for column, info in missing_values["columns_with_missing"].items():
            missing_data.append([column, f"{info['count']:,}", f"{info['percentage']}%"])
        story.append(_build_table(missing_data, [2.8 * inch, 1.4 * inch, 1.3 * inch]))
    story.append(Spacer(1, 0.18 * inch))

    story.extend(_build_section_header("Numeric Summary", styles))
    if not numeric_summary:
        story.append(Paragraph("No numeric columns found.", styles["body"]))
    else:
        for column, stats in numeric_summary.items():
            story.append(Paragraph(column, styles["subsection"]))
            stats_data = [["Statistic", "Value"]]
            for stat_name, stat_value in stats.items():
                display_value = "N/A" if stat_value is None else f"{stat_value:,.4g}"
                stats_data.append([stat_name.replace("_", " ").title(), display_value])
            story.append(_build_table(stats_data, [2.2 * inch, 2.2 * inch]))
            story.append(Spacer(1, 0.06 * inch))
    story.append(Spacer(1, 0.1 * inch))

    story.extend(_build_section_header("Categorical Summary", styles))
    if not categorical_summary:
        story.append(Paragraph("No categorical columns found.", styles["body"]))
    else:
        for column, stats in categorical_summary.items():
            story.append(Paragraph(column, styles["subsection"]))
            top_values_data = [["Value", "Count"]]
            for value, count in stats["top_values"].items():
                top_values_data.append([value, f"{count:,}"])
            story.append(_build_table(top_values_data, [3.2 * inch, 1.2 * inch]))
            story.append(Spacer(1, 0.06 * inch))

    story.append(PageBreak())
    story.extend(_build_section_header("Visual Analysis", styles))

    if not chart_paths:
        story.append(Paragraph("No charts were generated for this dataset.", styles["body"]))
    else:
        max_chart_width = doc.width
        max_chart_height = 4.2 * inch
        for index, chart_path in enumerate(chart_paths):
            if not chart_path.exists():
                continue
            story.append(Paragraph(_format_chart_title(chart_path), styles["subsection"]))
            story.append(_scaled_chart_image(chart_path, max_chart_width, max_chart_height))
            story.append(Spacer(1, 0.18 * inch if index < len(chart_paths) - 1 else 0.04 * inch))

    doc.build(
        story,
        onFirstPage=lambda canvas, doc: _draw_page_footer(canvas, doc, footer_timestamp),
        onLaterPages=lambda canvas, doc: _draw_page_footer(canvas, doc, footer_timestamp),
    )
    return pdf_path
