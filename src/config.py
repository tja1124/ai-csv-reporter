"""Project paths and configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"

_RUNTIME_DIRS = (
    OUTPUTS_DIR,
    UPLOADS_DIR,
    REPORTS_DIR,
    CHARTS_DIR,
    LOGS_DIR,
)


def ensure_output_dirs() -> None:
    """Create output folders required by the CLI and Streamlit workspace."""
    for directory in _RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

# Chart settings
HISTOGRAM_BINS = 20
CATEGORICAL_TOP_N = 8
FIGURE_DPI = 200
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 5.8
FIGURE_COMPACT_HEIGHT = 4.2
FIGURE_BOXPLOT_HEIGHT = 4.75
FIGURE_SAVE_PAD = 0.12

# Chart selection thresholds
MIN_UNIQUE_FOR_HISTOGRAM = 5
MIN_ROWS_FOR_HISTOGRAM = 50
MIN_ROWS_PER_GROUP = 3
SMALL_SAMPLE_THRESHOLD = 30
MAX_HISTOGRAMS = 1
MAX_SCATTER_PLOTS = 2
MAX_BOXPLOTS = 2
MAX_BAR_CHARTS = 2
MAX_TOTAL_CHARTS = 6
MIN_SCATTER_CORRELATION = 0.25
MIN_NUMERIC_VARIATION = 3
MAX_CATEGORIES_FOR_BOXPLOT = 10
MAX_UNIQUE_FOR_BAR_CHART = 12
ID_LIKE_NAME_KEYWORDS = ("id", "uuid", "guid", "key", "index", "code")
ID_LIKE_UNIQUENESS_RATIO = 0.95

# CSV loading
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

# Insight thresholds
CORRELATION_MIN_ABS = 0.5
SCATTER_CORRELATION_NOTE = 0.5
OUTLIER_IQR_MULTIPLIER = 1.5
DOMINANT_CATEGORY_THRESHOLD = 0.5

# OpenAI (optional Phase 3)
OPENAI_MODEL = "gpt-4o-mini"

# Shared visual theme
COLORS = {
    "primary": "#2E5090",
    "secondary": "#5B8DBE",
    "accent": "#94B8D9",
    "text": "#1E293B",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "background": "#FFFFFF",
    "surface": "#F8FAFC",
}

# PDF report theme (ReportLab)
REPORT_THEME = {
    "primary": COLORS["primary"],
    "muted": COLORS["muted"],
    "background": COLORS["background"],
    "surface": COLORS["surface"],
    "divider": COLORS["grid"],
    "title_size": 26,
    "section_size": 15,
    "subsection_size": 11,
    "body_size": 10,
    "caption_size": 8.5,
    "kpi_value_size": 15,
    "page_margin_in": 0.72,
}
