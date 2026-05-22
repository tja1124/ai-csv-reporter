# AI CSV Reporter

A professional AI-assisted CSV analytics platform built in Python.

Analyze CSV datasets through either:
- a modular command-line analytics pipeline
- or an interactive Streamlit analyst workspace

The system automatically validates datasets, generates insights, creates intelligent visualizations, and exports polished PDF analytics reports with optional AI-assisted executive summaries.

Built as a portfolio-quality project demonstrating:
- data engineering fundamentals
- deterministic analytics
- visualization intelligence
- report generation
- interactive analytics UX
- safe optional AI integration

---

# Features

## Core Analytics Pipeline

- Robust CSV loading and validation
  - Multiple encodings supported
  - Friendly validation errors
  - Non-fatal warnings
  - ID-like column detection
  - Dataset quality scoring

- Dataset analysis
  - Overview metrics
  - Missing value analysis
  - Numeric summaries
  - Categorical summaries
  - Correlation analysis
  - Outlier detection (IQR)

- Deterministic insight generation
  - Business-oriented findings
  - Sample-size-aware wording
  - Data quality warnings
  - Correlation highlights
  - Grouped comparison insights

---

## Intelligent Visualization Engine

Automatically selects meaningful charts instead of generating generic plots.

### Supported visualizations

- Scatterplots with trend lines
- Boxplots for grouped comparisons
- Selective bar charts
- Histograms only when statistically useful

### Visualization intelligence

- Correlation-based chart selection
- ID-column filtering
- Group-size validation
- Layout-aware chart planning
- Chart metadata for future UI builders

---

## Professional PDF Reporting

Exports polished analytics reports with:

- Branded title page
- KPI summary cards
- Executive summary
- Analyst takeaways
- Intelligent chart layouts
- Chart insight captions
- Structured appendix
- Compact reference tables

---

## Interactive Streamlit Analyst Workspace

A lightweight interactive analytics UI built on top of the same backend engine.

### Current capabilities

- CSV upload
- Dataset preview
- KPI overview
- Interactive chart builder
- Chart type dropdowns
- Variable selection
- Grouped visualizations
- Auto insights tab
- PDF report export

### Supported chart builder controls

- Chart type
- X variable
- Y variable
- Grouping variable

---

## Optional AI Executive Summaries

Generate executive summaries using OpenAI.

### AI safety design

Only aggregated metadata is sent:
- Dataset overview
- Quality score
- Insights
- Warnings
- Outlier summaries
- Correlations
- Chart notes

**Raw CSV rows are never sent to OpenAI.**

If AI generation fails or no API key is present, the pipeline automatically falls back to a deterministic summary.

---

## Logging

Detailed logs are saved to:

```text
logs/app.log
```

---

# Architecture

The system uses a modular function-based pipeline.

```text
CSV File / Streamlit Upload
              │
              ▼
data_loader.py
Load + validate dataset
              │
              ▼
analyzer.py
Compute dataset summaries
              │
              ▼
insight_generator.py
Insights, quality score, outliers, correlations
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
chart_generator.py
Visual planning + chart metadata

ai_summary.py
Optional AI executive summaries

streamlit_app.py
Interactive analyst workspace
      │
      ▼
pdf_report.py
Professional PDF export
```

---

# Project Structure

```text
ai_csv_reporter_clean/
├── data/
│   └── sample_data.csv
│
├── docs/
│   └── FUTURE_UI_PLAN.md
│
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── analyzer.py
│   ├── insight_generator.py
│   ├── ai_summary.py
│   ├── chart_generator.py
│   ├── column_utils.py
│   ├── format_utils.py
│   ├── pdf_report.py
│   └── main.py
│
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── uploads/
│
├── logs/
│
├── streamlit_app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Setup

## 1. Clone the repository

```bash
git clone <repo_url>
cd ai_csv_reporter_clean
```

---

## 2. Create and activate a virtual environment

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. (Optional) Configure OpenAI

```bash
export OPENAI_API_KEY=your_api_key_here
```

---

# Usage

## CLI Analytics Pipeline

### Standard report generation

```bash
python main.py data/sample_data.csv
```

### With AI-assisted executive summary

```bash
python main.py data/sample_data.csv --ai-summary
```

---

## Interactive Streamlit Workspace

Launch the local analyst workspace:

```bash
streamlit run streamlit_app.py
```

Then:

1. Upload a CSV file
2. Explore KPIs and insights
3. Build charts interactively
4. Export a professional PDF report

---

# Sample CLI Output

```text
[INFO] Loading CSV: data/sample_data.csv
[INFO] Loaded 10 rows and 5 columns
[INFO] Analyzing dataset...
[INFO] Generating insights...
[INFO] Dataset quality score: 100/100 (Excellent)
[INFO] Generating charts...
[INFO] Generated 3 chart(s)
[INFO] Creating PDF report...
[SUCCESS] Report saved to:
outputs/reports/sample_data_report_20260522_172309.pdf
```

---

# Generated Outputs

| Location | Contents |
|---|---|
| `outputs/charts/` | Generated chart PNG files |
| `outputs/reports/` | Exported PDF reports |
| `outputs/uploads/` | Uploaded Streamlit CSV files |
| `logs/app.log` | Detailed execution logs |

---

# Chart Metadata System

Every generated chart includes metadata for future UI expansion:

```python
{
    "chart_type": "scatter",
    "x_column": "age",
    "y_column": "salary",
    "group_column": None,
    "title": "salary vs age",
    "caption": "...",
    "reason_selected": "...",
    "layout": "hero"
}
```

This prepares the backend for:
- Dropdown-based chart builders
- Future drag-and-drop workflows
- Dashboard layouts
- Chart export systems

without rewriting the analytics engine.

---

# Design Philosophy

This project intentionally avoids:

- Unnecessary abstraction
- Premature microservices
- Database complexity
- Overengineered architecture
- AI dependency for core analytics

The goal is:

> A clean, maintainable analytics platform with deterministic intelligence first and optional AI enhancement second.

---

# Roadmap

## Interactive Analytics Workspace

- [x] Streamlit analyst workspace
- [x] Interactive chart builder
- [x] Variable dropdown controls
- [ ] Auto-generated chart gallery
- [ ] Chart filtering controls
- [ ] In-browser PDF preview

---

## Future UI Expansion

- [ ] Multi-chart dashboard layouts
- [ ] Drag-and-drop graph builder inspired by JMP
- [ ] Saved chart configurations
- [ ] Interactive report editor
- [ ] HTML report export
- [ ] Theme customization

---

## Platform Enhancements

- [ ] Batch CSV processing
- [ ] Scheduled report generation
- [ ] Multi-file comparison reports
- [ ] Additional statistical modules
- [ ] Cloud deployment prototype

---

# License

MIT
