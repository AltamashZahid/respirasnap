"""
test_respirasnap.py — Basic unit tests for RespriaSnap
Run: python -m pytest test_respirasnap.py -v
"""

import pytest
import numpy as np
import os
import json
import csv
from datetime import datetime
from audio import load_file
from preprocess import preprocess
from envelope import extract_envelope
from detect import detect_peaks_valleys
from segment import segment_breathing
from features import extract_features
from export import export_session_csv, export_history_csv, export_session_json, export_comparison_csv, get_default_export_filename
from trending import calculate_trend_stats, get_trend_summary, plot_trending_chart
from compare import format_comparison_for_display


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
        assert isinstance(cleaned, np.ndarray)

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
            assert "start" in seg
            assert "end" in seg
            assert "duration" in seg


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


class TestExport:
    """Test export functionality."""

    def test_export_session_csv(self):
        """Test CSV export of single session."""
        session_data = {
            "features": {
                "bpm": 12.5,
                "inhale_exhale_ratio": 1.5,
                "stability_cv": 0.25,
                "avg_inhale_s": 0.6,
                "avg_exhale_s": 0.9,
                "num_cycles": 3,
                "holds_detected": 1,
                "interruptions": 0,
                "audio_duration_s": 15.0,
            },
            "score": 85,
            "timestamp": "2026-06-06T12:00:00"
        }

        filename = "test_session.csv"
        result = export_session_csv(session_data, filename)

        assert result["success"]
        assert os.path.exists(filename)

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) > 0
            assert "BREATHING SCORE" in str(rows)

        os.remove(filename)

    def test_export_history_csv(self):
        """Test CSV export of session history."""
        history = [
            {
                "timestamp": "2026-06-01T12:00:00",
                "score": 80,
                "features": {"bpm": 12.0, "inhale_exhale_ratio": 1.5, "stability_cv": 0.25,
                            "avg_inhale_s": 0.6, "avg_exhale_s": 0.9, "num_cycles": 3,
                            "holds_detected": 1, "interruptions": 0, "audio_duration_s": 15.0}
            },
            {
                "timestamp": "2026-06-02T12:00:00",
                "score": 85,
                "features": {"bpm": 11.5, "inhale_exhale_ratio": 1.6, "stability_cv": 0.23,
                            "avg_inhale_s": 0.65, "avg_exhale_s": 1.0, "num_cycles": 3,
                            "holds_detected": 1, "interruptions": 0, "audio_duration_s": 15.0}
            }
        ]

        filename = "test_history.csv"
        result = export_history_csv(history, filename)

        assert result["success"]
        assert result["count"] == 2
        assert os.path.exists(filename)

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2

        os.remove(filename)

    def test_export_session_json(self):
        """Test JSON export of session."""
        session_data = {
            "features": {"bpm": 12.0, "stability_cv": 0.25},
            "score": 85,
            "timestamp": "2026-06-06T12:00:00",
            "clinical": "Test clinical note",
            "coaching": "Test coaching",
            "followup": "Test followup"
        }

        filename = "test_session.json"
        result = export_session_json(session_data, filename)

        assert result["success"]
        assert os.path.exists(filename)

        with open(filename, 'r') as f:
            data = json.load(f)
            assert data["score"] == 85
            assert data["clinical"] == "Test clinical note"

        os.remove(filename)

    def test_default_export_filename(self):
        """Test default export filename generation."""
        filename1 = get_default_export_filename("session_csv")
        filename2 = get_default_export_filename("history_csv")

        assert filename1.endswith(".csv")
        assert "session" in filename1
        assert filename2.endswith(".csv")
        assert "history" in filename2


class TestTrending:
    """Test trending functionality."""

    def test_calculate_trend_stats_empty(self):
        """Test trend stats with empty history."""
        stats = calculate_trend_stats([])

        assert stats["session_count"] == 0
        assert stats["overall_trend"] == "stable"
        assert stats["improvement_pct"] == 0

    def test_calculate_trend_stats(self):
        """Test trend stats calculation."""
        history = [
            {
                "score": 70,
                "features": {"bpm": 14.0, "stability_cv": 0.3, "inhale_exhale_ratio": 1.2},
                "timestamp": "2026-06-01T12:00:00"
            },
            {
                "score": 75,
                "features": {"bpm": 13.0, "stability_cv": 0.28, "inhale_exhale_ratio": 1.3},
                "timestamp": "2026-06-02T12:00:00"
            },
            {
                "score": 85,
                "features": {"bpm": 12.5, "stability_cv": 0.25, "inhale_exhale_ratio": 1.4},
                "timestamp": "2026-06-03T12:00:00"
            }
        ]

        stats = calculate_trend_stats(history)

        assert stats["session_count"] == 3
        assert stats["best_score"] == 85
        assert stats["worst_score"] == 70
        assert stats["avg_score"] == 76.7
        assert stats["improvement_pct"] > 0  # Improving

    def test_get_trend_summary(self):
        """Test trend summary generation."""
        history = [{"score": 80, "features": {"bpm": 12.0, "stability_cv": 0.25, "inhale_exhale_ratio": 1.5},
                   "timestamp": "2026-06-01T12:00:00"}]

        summary = get_trend_summary(history)

        assert isinstance(summary, str)
        assert "BREATHING TREND SUMMARY" in summary
        assert "Average Score" in summary

    def test_plot_trending_chart(self):
        """Test trending chart generation."""
        history = [
            {"score": 75, "features": {"bpm": 13.0, "stability_cv": 0.28, "inhale_exhale_ratio": 1.2},
             "timestamp": "2026-06-01T12:00:00"},
            {"score": 80, "features": {"bpm": 12.5, "stability_cv": 0.25, "inhale_exhale_ratio": 1.3},
             "timestamp": "2026-06-02T12:00:00"},
            {"score": 85, "features": {"bpm": 12.0, "stability_cv": 0.23, "inhale_exhale_ratio": 1.4},
             "timestamp": "2026-06-03T12:00:00"}
        ]

        fig = plot_trending_chart(history)

        assert fig is not None
        # Figure should have 3 subplots
        assert len(fig.axes) == 3


class TestComparison:
    """Test comparison functionality."""

    def test_format_comparison_for_display(self):
        """Test comparison formatting for display."""
        comparison = {
            "overall_alignment_score": 85.5,
            "pose_comparison": {"pose_accuracy": 90.0, "matching_poses": 9, "total_poses": 10},
            "breathing_comparison": {
                "breathing_alignment_score": 88.0,
                "metrics_comparison": {
                    "bpm": {"label": "BPM", "user": 12.0, "reference": 13.0, "diff_pct": 7.7, "unit": "breaths/min"},
                    "inhale_exhale_ratio": {"label": "IE Ratio", "user": 1.5, "reference": 1.4, "diff_pct": 7.1, "unit": "x"}
                }
            },
            "segment_comparison": {"segment_accuracy": 80.0},
            "recommendations": ["✨ Excellent alignment with reference!"]
        }

        text = format_comparison_for_display(comparison)

        assert isinstance(text, str)
        assert "COMPARISON ANALYSIS" in text
        assert "85.5" in text or "85" in text
        assert "BREATHING METRICS COMPARISON" in text
        assert "RECOMMENDATIONS" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
