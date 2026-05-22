# AI CSV Reporter

A Python command-line tool that turns CSV files into professional PDF analytics reports. Load a dataset, run automated analysis, generate charts, and export a polished report — with optional AI-assisted executive summaries.

Built as a modular, portfolio-ready project demonstrating data validation, deterministic analytics, visualization, PDF generation, and safe optional AI integration.

## Features

- **CSV loading and validation** — multiple encodings, friendly error messages, non-fatal warnings
- **Dataset analysis** — overview, missing values, numeric and categorical summaries
- **Report intelligence** — quality score (0–100), key insights, outlier detection (IQR), correlation highlights
- **Intelligent charts** — scatterplots, boxplots, and selective bar charts with metadata for future UI builders
- **PDF export** — KPI cards, executive summary, compact chart layouts, and structured appendix
- **Optional AI summary** — executive summary via OpenAI using aggregated metadata only
- **Logging** — detailed run history saved to `logs/app.log`

## Architecture

The pipeline is function-based and modular. Each module has a single responsibility:

```
CSV File
   │
   ▼
data_loader.py      Load and validate
   │
   ▼
analyzer.py         Compute summaries
   │
   ▼
insight_generator.py Quality score, insights, outliers, correlations
   │
   ├── chart_generator.py   Generate charts
   │
   ├── ai_summary.py        Optional executive summary (--ai-summary)
   │
   ▼
pdf_report.py       Export PDF report
```

## Project Structure

```
ai_csv_reporter_clean/
├── data/
│   └── sample_data.csv       Example dataset
├── src/
│   ├── config.py             Paths and settings
│   ├── logger.py             File logging setup
│   ├── data_loader.py        CSV loading and validation
│   ├── analyzer.py           Dataset analysis
│   ├── insight_generator.py  Quality scores and insights
│   ├── ai_summary.py         Optional OpenAI summary
│   ├── column_utils.py       Shared ID/grouping rules
  ├── chart_generator.py    Matplotlib charts + chart metadata
│   ├── pdf_report.py         ReportLab PDF export
│   └── main.py               CLI entry point
├── outputs/                  Generated charts and reports (gitignored)
├── logs/                     Application logs (gitignored)
├── main.py                   Project root entry point
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository and enter the project directory.

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Set your OpenAI API key for AI summaries:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic run

```bash
python main.py data/sample_data.csv
```

### With optional AI executive summary

```bash
python main.py data/sample_data.csv --ai-summary
```

If `OPENAI_API_KEY` is not set or the request fails, the report still completes using a deterministic fallback summary.

## Sample Output

```text
[INFO] Loading CSV: data/sample_data.csv
[INFO] Loaded 10 rows and 5 columns
[INFO] Analyzing dataset...
[INFO] Generating insights...
[INFO] Dataset quality score: 100/100 (Excellent)
[INFO] Generating charts...
[INFO] Generated 5 chart(s)
[INFO] Creating PDF report...
[SUCCESS] Report saved to: outputs/reports/sample_data_report_20260522_120000.pdf
```

**Generated files:**

| Location | Contents |
|---|---|
| `outputs/charts/<name>_<timestamp>/` | PNG chart files |
| `outputs/reports/<name>_report_<timestamp>.pdf` | Full PDF report |
| `logs/app.log` | Detailed run log |

## AI Safety Note

When `--ai-summary` is enabled, OpenAI receives **only aggregated report metadata** — dataset overview, column summaries, quality scores, insights, warnings, outlier counts, correlation highlights, and chart notes.

**Raw CSV rows are never sent to OpenAI.**

## Roadmap

### Interactive analytics (planned)
- [ ] Streamlit or lightweight web UI for upload and report preview
- [ ] Dropdown-based chart builder (X variable, Y variable, grouping variable, chart type)
- [ ] Drag-and-drop style graph builder inspired by JMP
- [ ] Export selected charts to PDF from the UI

### Platform enhancements
- [ ] Batch processing for multiple CSV files
- [ ] Custom PDF report templates
- [ ] Export to HTML in addition to PDF
- [ ] Scheduled report generation

## License

MIT
