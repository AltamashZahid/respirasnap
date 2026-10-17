"""
features.py — Compute all breathing metrics from segments + peaks
"""

import numpy as np


def extract_features(segments: list[dict],
                     peaks: dict,
                     audio_duration: float) -> dict:
    """
    Compute structured breathing features from segment list.

    Returns a flat dict ready for scoring and LLM input.
    """
    inhales = [s for s in segments if s["phase"] == "INHALE"]
    exhales = [s for s in segments if s["phase"] == "EXHALE"]
    holds   = [s for s in segments if s["phase"] == "HOLD"]

    # Breathing rate: count full breath cycles
    # One cycle ≈ one inhale + one exhale
    num_cycles = min(len(inhales), len(exhales))
    duration_min = audio_duration / 60.0
    # Prefer the median peak-to-peak interval: counting only complete cycles in a
    # short window under-reads the rate (e.g. 2 vs 3 cycles in 15 s = 8 vs 12 BPM).
    peak_times = np.asarray(peaks.get("peak_times", []), dtype=float)
    if len(peak_times) >= 3 and num_cycles > 0:
        bpm = round(60.0 / float(np.median(np.diff(peak_times))), 2)
    else:
        bpm = round(num_cycles / duration_min, 2) if duration_min > 0 else 0.0

    # Average durations
    avg_inhale = float(np.mean([s["duration"] for s in inhales])) if inhales else 0.0
    avg_exhale = float(np.mean([s["duration"] for s in exhales])) if exhales else 0.0

    # Inhale-to-exhale ratio
    ie_ratio = round(avg_exhale / avg_inhale, 3) if avg_inhale > 1e-3 else 0.0

    # Stability: coefficient of variation across inhale durations
    if len(inhales) > 1:
        durations = [s["duration"] for s in inhales]
        stability_cv = float(np.std(durations) / (np.mean(durations) + 1e-6))
    else:
        stability_cv = 0.0  # Can't measure with one cycle

    # Interruptions: sharp very-short exhales or inhales (< 0.4s) between cycles
    short_segments = [s for s in segments if s["duration"] < 0.4 and s["phase"] != "HOLD"]
    interruptions = max(0, len(short_segments) - 1)  # allow one short segment

    return {
        "bpm":               bpm,
        "inhale_exhale_ratio": ie_ratio,
        "stability_cv":      round(stability_cv, 4),   # lower = more stable
        "interruptions":     interruptions,
        "avg_inhale_s":      round(avg_inhale, 3),
        "avg_exhale_s":      round(avg_exhale, 3),
        "holds_detected":    len(holds),
        "num_cycles":        num_cycles,
        "audio_duration_s":  round(audio_duration, 2),
    }
