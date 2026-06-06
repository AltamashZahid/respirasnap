"""
envelope.py — Extract amplitude envelope using RMS windowing
"""

import numpy as np
from scipy.signal import savgol_filter


def extract_envelope(audio: np.ndarray, sr: int,
                     frame_ms: int = 20, hop_ms: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS envelope of audio.

    Returns:
        times   — time axis in seconds (one per hop)
        envelope — RMS amplitude per frame
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms / 1000)

    frames = []
    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start: start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append(rms)

    envelope = np.array(frames)
    times = np.arange(len(envelope)) * (hop_ms / 1000.0)

    # Smooth with Savitzky-Golay to remove micro-jitter
    if len(envelope) > 21:
        envelope = savgol_filter(envelope, window_length=21, polyorder=3)

    envelope = np.clip(envelope, 0, None)  # no negatives after smoothing
    return times, envelope
