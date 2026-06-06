"""
preprocess.py — Normalize and bandpass filter the audio signal
"""

import numpy as np
from scipy.signal import butter, sosfilt


def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio  # silence guard
    return audio / peak


def bandpass_filter(audio: np.ndarray, sr: int,
                    low_hz: float = 100.0, high_hz: float = 1000.0) -> np.ndarray:
    """
    Apply a bandpass filter to isolate breathing frequencies.
    Breathing sounds are typically 100–1000 Hz.
    """
    nyq = sr / 2.0
    sos = butter(
        N=4,
        Wn=[low_hz / nyq, high_hz / nyq],
        btype="bandpass",
        output="sos"
    )
    return sosfilt(sos, audio)


def preprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """Full preprocessing pipeline: normalize → bandpass."""
    audio = normalize(audio)
    audio = bandpass_filter(audio, sr)
    return audio
