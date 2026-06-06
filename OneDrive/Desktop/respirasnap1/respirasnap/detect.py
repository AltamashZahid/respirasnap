"""
detect.py — Detect peaks (inhale peaks) and valleys in the envelope
"""

import numpy as np
from scipy.signal import find_peaks


def detect_peaks_valleys(envelope: np.ndarray,
                          times: np.ndarray) -> dict:
    """
    Find peaks (inhale) and valleys (exhale troughs) in the envelope.

    Returns dict with peak/valley indices and timestamps.
    """
    # Minimum distance between peaks: ~1.5 seconds (at 10ms hop → 150 frames)
    min_dist = max(50, int(len(envelope) * 0.08))

    # Peaks = high RMS = inhale effort
    peak_idx, _ = find_peaks(
        envelope,
        distance=min_dist,
        prominence=np.std(envelope) * 0.3
    )

    # Valleys = low RMS = exhale troughs
    valley_idx, _ = find_peaks(
        -envelope,
        distance=min_dist,
        prominence=np.std(envelope) * 0.2
    )

    return {
        "peak_indices":   peak_idx,
        "peak_times":     times[peak_idx] if len(peak_idx) > 0 else np.array([]),
        "valley_indices": valley_idx,
        "valley_times":   times[valley_idx] if len(valley_idx) > 0 else np.array([]),
    }
