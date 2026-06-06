"""
segment.py — Label breath phases using peak/valley positions (robust to noise)

Peak/valley approach is used instead of gradient slope because real mic audio
is noisy — gradient-based labeling produces hundreds of tiny flipping segments
that all get filtered out, giving zero results.

Logic:
  valley → next peak  = INHALE  (amplitude rising)
  peak   → next valley = EXHALE  (amplitude falling)
  before first event / after last event = HOLD
"""

import numpy as np


def segment_breathing(envelope: np.ndarray,
                      times: np.ndarray,
                      peaks: dict) -> list[dict]:
    """
    Build breath segments from detected peaks and valleys.

    Args:
        envelope : RMS envelope array
        times    : time axis (same length as envelope)
        peaks    : dict from detect.detect_peaks_valleys()

    Returns:
        list of {phase, start, end, duration}
    """
    peak_idx   = peaks["peak_indices"]
    valley_idx = peaks["valley_indices"]

    if len(peak_idx) == 0 and len(valley_idx) == 0:
        # Fallback: no events detected — label everything HOLD
        return [{"phase": "HOLD",
                 "start": float(times[0]),
                 "end":   float(times[-1]),
                 "duration": float(times[-1] - times[0])}]

    # Merge peaks and valleys into a sorted event list
    events = []
    for i in peak_idx:
        if i < len(times):
            events.append({"t": float(times[i]), "kind": "PEAK"})
    for i in valley_idx:
        if i < len(times):
            events.append({"t": float(times[i]), "kind": "VALLEY"})
    events.sort(key=lambda e: e["t"])

    if not events:
        return []

    segments = []
    audio_end = float(times[-1])
    audio_start = float(times[0])

    # Gap before first event → HOLD
    if events[0]["t"] > audio_start + 0.2:
        segments.append({
            "phase": "HOLD",
            "start": audio_start,
            "end":   events[0]["t"],
            "duration": events[0]["t"] - audio_start,
        })

    # Segments between consecutive events
    for i in range(len(events) - 1):
        e0, e1 = events[i], events[i + 1]
        duration = e1["t"] - e0["t"]

        if e0["kind"] == "VALLEY" and e1["kind"] == "PEAK":
            phase = "INHALE"
        elif e0["kind"] == "PEAK" and e1["kind"] == "VALLEY":
            phase = "EXHALE"
        else:
            # Two peaks or two valleys in a row → hold between them
            phase = "HOLD"

        segments.append({
            "phase":    phase,
            "start":    e0["t"],
            "end":      e1["t"],
            "duration": duration,
        })

    # Gap after last event → HOLD
    if audio_end - events[-1]["t"] > 0.2:
        segments.append({
            "phase": "HOLD",
            "start": events[-1]["t"],
            "end":   audio_end,
            "duration": audio_end - events[-1]["t"],
        })

    # Filter noise: remove segments under 0.15s
    segments = [s for s in segments if s["duration"] >= 0.15]

    return segments
