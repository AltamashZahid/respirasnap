"""
score.py — Compute breathing score (0–100) from extracted features
"""


def compute_score(features: dict) -> dict:
    """
    Rule-based score starting at 100, with penalties.

    Returns score + breakdown dict.
    """
    score = 100
    deductions = {}

    bpm           = features["bpm"]
    ie_ratio      = features["inhale_exhale_ratio"]
    stability_cv  = features["stability_cv"]
    interruptions = features["interruptions"]

    # --- BPM penalty ---
    # Healthy resting breathing: 12–20 BPM
    if bpm == 0:
        score -= 30
        deductions["bpm"] = -30
    elif bpm < 8 or bpm > 25:
        score -= 20
        deductions["bpm"] = -20
    elif bpm < 12 or bpm > 20:
        score -= 10
        deductions["bpm"] = -10
    else:
        deductions["bpm"] = 0

    # --- Stability penalty ---
    # CV > 0.4 is quite variable
    if stability_cv > 0.6:
        score -= 20
        deductions["stability"] = -20
    elif stability_cv > 0.4:
        score -= 12
        deductions["stability"] = -12
    elif stability_cv > 0.25:
        score -= 6
        deductions["stability"] = -6
    else:
        deductions["stability"] = 0

    # --- Inhale/Exhale ratio penalty ---
    # Ideal: exhale ≥ inhale (ratio ≥ 1.0). Relaxation target: ~1.5–2x
    if ie_ratio == 0:
        score -= 10
        deductions["ie_ratio"] = -10
    elif ie_ratio < 0.8:
        score -= 10   # too short exhale
        deductions["ie_ratio"] = -10
    elif ie_ratio > 3.5:
        score -= 8    # unusually long exhale
        deductions["ie_ratio"] = -8
    else:
        deductions["ie_ratio"] = 0

    # --- Interruption penalty ---
    if interruptions >= 3:
        score -= 15
        deductions["interruptions"] = -15
    elif interruptions >= 1:
        score -= 7
        deductions["interruptions"] = -7
    else:
        deductions["interruptions"] = 0

    score = max(0, min(100, score))

    return {
        "score":      score,
        "deductions": deductions,
    }
