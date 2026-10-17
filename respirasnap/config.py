"""
config.py — Adjustable parameters for RespriaSnap
"""

# Audio recording
SAMPLE_RATE = 16000  # Hz
DEFAULT_RECORD_DURATION = 15.0  # seconds

# Signal processing
ENVELOPE_SMOOTH_WINDOW = 21  # sample window for smoothing
PEAK_PROMINENCE = 0.1  # for peak detection
PEAK_DISTANCE = 100  # minimum samples between peaks

# Breathing analysis
MIN_SEGMENT_DURATION = 0.3  # seconds, minimum breath phase
OUTLIER_THRESHOLD = 2.5  # standard deviations for outlier detection

# Scoring
SCORE_IDEAL_EXHALE_RATIO = 1.5  # target exhale:inhale ratio
SCORE_REGULARITY_WEIGHT = 0.3  # importance of rhythm consistency
SCORE_EFFICIENCY_WEIGHT = 0.7  # importance of breathing efficiency

# LLM
LLM_MODEL = "llama-3.1-8b-instant"
LLM_MAX_TOKENS = 250
LLM_TEMPERATURE = 0.4

# Visualization
PLOT_FIGSIZE = (14, 10)
PLOT_DPI = 100
PLOT_COLORS = {
    "waveform": "#4FC3F7",
    "envelope": "#56D364",
    "peaks": "#F0883E",
    "valleys": "#F85149",
}

# GUI
GUI_THEME_BG = "#0D1117"
GUI_THEME_PANEL = "#161B22"
GUI_THEME_BORDER = "#30363D"
GUI_THEME_ACCENT = "#4FC3F7"
GUI_THEME_GREEN = "#56D364"
GUI_THEME_ORANGE = "#F0883E"
GUI_THEME_RED = "#F85149"

# History & Storage
HISTORY_FILE = "session_history.json"
REFERENCES_DIR = "references"
