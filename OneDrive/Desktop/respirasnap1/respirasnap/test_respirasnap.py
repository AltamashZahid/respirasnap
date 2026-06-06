"""
test_respirasnap.py — Basic unit tests for RespriaSnap
Run: python -m pytest test_respirasnap.py -v
"""

import pytest
import numpy as np
from audio import load_file
from preprocess import preprocess
from envelope import extract_envelope
from detect import detect_peaks_valleys
from segment import segment_breathing
from features import extract_features


def create_test_audio(sr=16000, duration=5.0, freq=0.25):
    """Create synthetic test audio with known breathing pattern."""
    t = np.linspace(0, duration, int(sr * duration))
    envelope_shape = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t)
    carrier = np.sin(2 * np.pi * 300 * t)
    noise = np.random.normal(0, 0.02, len(t))
    return (envelope_shape * carrier + noise).astype(np.float32)


class TestAudioProcessing:
    """Test audio preprocessing pipeline."""

    def test_preprocess_shape(self):
        audio = create_test_audio()
        cleaned = preprocess(audio, 16000)
        assert cleaned.shape == audio.shape
        assert cleaned.dtype == np.float32

    def test_preprocess_reduces_noise(self):
        audio = create_test_audio()
        cleaned = preprocess(audio, 16000)
        assert np.std(cleaned) < np.std(audio)

    def test_envelope_extraction(self):
        audio = create_test_audio()
        cleaned = preprocess(audio, 16000)
        times, envelope = extract_envelope(cleaned, 16000)
        assert len(times) == len(envelope)
        assert np.all(envelope >= 0)
        assert np.max(envelope) <= 1.0

    def test_peak_detection(self):
        audio = create_test_audio(freq=0.25)  # 4 breaths in 5 seconds
        cleaned = preprocess(audio, 16000)
        times, envelope = extract_envelope(cleaned, 16000)
        peaks = detect_peaks_valleys(envelope, times)

        assert "peak_indices" in peaks
        assert "valley_indices" in peaks
        assert len(peaks["peak_indices"]) > 0
        assert len(peaks["valley_indices"]) > 0

    def test_segmentation(self):
        audio = create_test_audio()
        cleaned = preprocess(audio, 16000)
        times, envelope = extract_envelope(cleaned, 16000)
        peaks = detect_peaks_valleys(envelope, times)
        segments = segment_breathing(envelope, times, peaks)

        assert len(segments) > 0
        for seg in segments:
            assert "phase" in seg
            assert seg["phase"] in ["INHALE", "EXHALE", "HOLD"]
            assert "start_time" in seg
            assert "end_time" in seg


class TestFeatureExtraction:
    """Test feature extraction."""

    def test_features_structure(self):
        audio = create_test_audio()
        cleaned = preprocess(audio, 16000)
        times, envelope = extract_envelope(cleaned, 16000)
        peaks = detect_peaks_valleys(envelope, times)
        segments = segment_breathing(envelope, times, peaks)
        features = extract_features(segments, peaks, 5.0)

        assert isinstance(features, dict)
        assert len(features) > 0
        for key, val in features.items():
            assert val is not None

    def test_features_non_negative(self):
        audio = create_test_audio()
        cleaned = preprocess(audio, 16000)
        times, envelope = extract_envelope(cleaned, 16000)
        peaks = detect_peaks_valleys(envelope, times)
        segments = segment_breathing(envelope, times, peaks)
        features = extract_features(segments, peaks, 5.0)

        # Duration and count features should be non-negative
        for key, val in features.items():
            if isinstance(val, (int, float)):
                assert val >= 0, f"{key} = {val} should be >= 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
