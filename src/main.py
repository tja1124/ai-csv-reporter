"""Command-line entry point for the CSV analytics reporter."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.ai_summary import (
    build_summary_context,
    generate_ai_executive_summary,
    get_fallback_summary,
)
from src.analyzer import (
    analyze_columns,
    get_categorical_summary,
    get_dataset_overview,
    get_missing_values,
    get_numeric_summary,
)
from src.chart_generator import (
    generate_categorical_bar_charts,
    generate_correlation_heatmap,
    generate_numeric_histograms,
)
from src.config import CHARTS_DIR, REPORTS_DIR
from src.data_loader import load_csv, validate_dataframe
from src.insight_generator import generate_report_insights
from src.logger import get_logger, setup_logging
from src.pdf_report import create_pdf_report


def _info(message: str) -> None:
    """Print a user-facing info message and log it."""
    print(f"[INFO] {message}")
    get_logger().info(message)


def _warn(message: str) -> None:
    """Print a user-facing warning message and log it."""
    print(f"[WARN] {message}")
    get_logger().warning(message)


def _success(message: str) -> None:
    """Print a user-facing success message and log it."""
    print(f"[SUCCESS] {message}")
    get_logger().info(message)


def _print_warnings(warnings: list[str]) -> None:
    """Show non-fatal validation warnings in the terminal."""
    for warning in warnings:
        _warn(warning)


def _build_executive_summary(
    csv_name: str,
    overview: dict,
    column_info: dict,
    missing_values: dict,
    numeric_summary: dict,
    categorical_summary: dict,
    report_insights: dict,
) -> dict:
    """Attempt an AI executive summary, falling back to deterministic text."""
    context = build_summary_context(
        csv_name=csv_name,
        overview=overview,
        column_info=column_info,
        missing_values=missing_values,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        report_insights=report_insights,
    )

    ai_summary = generate_ai_executive_summary(context)
    if ai_summary is not None:
        _info("AI executive summary generated")
        return ai_summary

    _info("Using deterministic executive summary (AI unavailable or failed)")
    return get_fallback_summary(context)


def run_report(csv_path: str | Path, *, use_ai_summary: bool = False) -> Path:
    """
    Run the full CSV analysis and PDF report pipeline.

    Args:
        csv_path: Path to the input CSV file.
        use_ai_summary: Whether to include an optional executive summary section.

    Returns:
        Path to the generated PDF report.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file is invalid or empty.
    """
    csv_path = Path(csv_path)

    _info(f"Loading CSV: {csv_path}")
    df = load_csv(csv_path)

    warnings = validate_dataframe(df)
    if warnings:
        _print_warnings(warnings)

    _info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    _info("Analyzing dataset...")
    overview = get_dataset_overview(df)
    column_info = analyze_columns(df)
    missing_values = get_missing_values(df)
    numeric_summary = get_numeric_summary(df)
    categorical_summary = get_categorical_summary(df)

    _info("Generating insights...")
    report_insights = generate_report_insights(
        df=df,
        overview=overview,
        missing_values=missing_values,
        categorical_summary=categorical_summary,
        validation_warnings=warnings,
    )
    quality = report_insights["quality_score"]
    _info(f"Dataset quality score: {quality['score']}/100 ({quality['rating']})")

    executive_summary = None
    if use_ai_summary:
        _info("Generating executive summary...")
        executive_summary = _build_executive_summary(
            csv_name=csv_path.name,
            overview=overview,
            column_info=column_info,
            missing_values=missing_values,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            report_insights=report_insights,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_output_dir = CHARTS_DIR / f"{csv_path.stem}_{timestamp}"
    chart_output_dir.mkdir(parents=True, exist_ok=True)

    _info("Generating charts...")
    chart_paths: list[Path] = []
    chart_paths.extend(generate_numeric_histograms(df, chart_output_dir))
    chart_paths.extend(generate_categorical_bar_charts(df, chart_output_dir))

    heatmap_path = generate_correlation_heatmap(df, chart_output_dir)
    if heatmap_path:
        chart_paths.append(heatmap_path)

    _info(f"Generated {len(chart_paths)} chart(s)")

    report_filename = f"{csv_path.stem}_report_{timestamp}.pdf"
    report_path = REPORTS_DIR / report_filename

    _info("Creating PDF report...")
    create_pdf_report(
        csv_path=csv_path,
        overview=overview,
        missing_values=missing_values,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        chart_paths=chart_paths,
        output_path=report_path,
        report_insights=report_insights,
        executive_summary=executive_summary,
        generated_at=datetime.now(),
    )

    _success(f"Report saved to: {report_path}")
    return report_path


def main() -> None:
    """Parse command-line arguments and run the report pipeline."""
    setup_logging()
    logger = get_logger()

    parser = argparse.ArgumentParser(
        description="Analyze a CSV file and generate a PDF report with charts.",
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file to analyze",
    )
    parser.add_argument(
        "--ai-summary",
        action="store_true",
        help="Include an optional AI-assisted executive summary (requires OPENAI_API_KEY)",
    )

    args = parser.parse_args()

    try:
        run_report(args.csv_file, use_ai_summary=args.ai_summary)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during report generation")
        print(
            f"\n[ERROR] {exc}\nCheck logs/app.log for details.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
