"""Project paths and configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"

# Chart settings
HISTOGRAM_BINS = 20
CATEGORICAL_TOP_N = 10
FIGURE_DPI = 180
FIGURE_WIDTH = 9
FIGURE_HEIGHT = 5.5

# Chart selection thresholds
MIN_UNIQUE_FOR_HISTOGRAM = 3
MAX_UNIQUE_FOR_BAR_CHART = 15
ID_LIKE_NAME_KEYWORDS = ("id", "uuid", "guid", "key", "index", "code")
ID_LIKE_UNIQUENESS_RATIO = 0.95

# CSV loading
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

# Insight thresholds
CORRELATION_MIN_ABS = 0.5
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
