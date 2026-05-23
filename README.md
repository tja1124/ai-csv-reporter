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
- **Streamlit workspace** — interactive Plotly charts for exploration; PDF export still uses matplotlib

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
  ├── chart_generator.py    Matplotlib charts + chart metadata (CLI/PDF)
  ├── plotly_charts.py      Plotly charts (Streamlit only)
│   ├── pdf_report.py         ReportLab PDF export
│   └── main.py               CLI entry point
├── streamlit_app.py          Streamlit analyst workspace
├── .streamlit/config.toml    Streamlit theme and server settings
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

### Streamlit analyst workspace

```bash
streamlit run streamlit_app.py
```

The workspace reuses the same backend modules as the CLI:

- Upload and validate CSV files
- View KPI dashboard and recommended charts
- Build interactive scatter, boxplot, and bar charts (Plotly) with Preview / Insights / Data tabs
- See why each chart was recommended and planner-matched insights
- Generate the full PDF report through `run_report()` (matplotlib charts embedded in PDF)

Optional report title and analyst name can be set in the Export Report section.

## Deploying to Streamlit Cloud

This project is structured for Streamlit Cloud deployment without changing the CLI pipeline.

1. Push the repository to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io).
3. Set **Main file path** to `streamlit_app.py`.
4. Add secrets in the Streamlit Cloud dashboard if you want AI summaries:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

**Notes for deployment:**

- Uploaded CSVs are saved under `outputs/uploads/` and old files are cleaned automatically.
- Reports and charts continue to write to `outputs/reports/` and `outputs/charts/`.
- No database or authentication is required for the current prototype.
- The CLI remains the reference implementation and does not require Streamlit.

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

### Interactive analytics (in progress)
- [x] Streamlit analyst workspace with upload, KPI dashboard, and chart builder
- [ ] Dropdown-based chart builder refinements and saved chart recipes
- [ ] Drag-and-drop style graph builder inspired by JMP
- [ ] Export selected charts to PDF from the UI

### Platform enhancements
- [ ] Batch processing for multiple CSV files
- [ ] Custom PDF report templates
- [ ] Export to HTML in addition to PDF
- [ ] Scheduled report generation

## License

MIT
# rebuild trigger
