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


# ── Posture matching ─────────────────────────────────────

import pose
import references
from compare import (compare_pose_sequences, compare_breath_pattern,
                     compare_breathing_patterns, overall_comparison, live_pose_match)


def stick_figure(arms="T", left_knee_bend=0.0, lean=0.0):
    """33 MediaPipe-style landmarks (x, y, visibility) in pixels for a simple pose.

    arms: "T" (horizontal), "down", "up". left_knee_bend in degrees.
    lean rotates the whole body (degrees, 90 = horizontal).
    """
    lm = [(0.0, 0.0, 1.0)] * 33
    pts = {
        11: (-40, -100), 12: (40, -100),          # shoulders
        23: (-25, 0), 24: (25, 0),                # hips
        25: (-25, 90), 26: (25, 90),              # knees
        27: (-25, 180), 28: (25, 180),            # ankles
    }
    arm = {"T": (1, 0), "down": (0, 1), "up": (0, -1)}[arms]
    pts[13] = (-40 - 70 * arm[0], -100 + 70 * arm[1])
    pts[15] = (-40 - 140 * arm[0], -100 + 140 * arm[1])
    pts[14] = (40 + 70 * arm[0], -100 + 70 * arm[1])
    pts[16] = (40 + 140 * arm[0], -100 + 140 * arm[1])
    if left_knee_bend:
        a = np.radians(left_knee_bend)
        pts[27] = (-25 - 90 * np.sin(a), 90 + 90 * np.cos(a))
    th = np.radians(lean)
    out = list(lm)
    for i, (x, y) in pts.items():
        xr, yr = x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th)
        out[i] = (300 + xr, 300 + yr, 1.0)
    return out


def frames_from(seq, fps=10):
    """[(angles, seconds), ...] → pose frame list."""
    frames, t = [], 0.0
    for angles, secs in seq:
        for _ in range(int(secs * fps)):
            frames.append({"timestamp": round(t, 2), "pose": pose.classify_pose(angles),
                           "angles": pose.clean_angles(angles)})
            t += 1.0 / fps
    return frames


class TestPose:
    def test_t_pose_angles(self):
        a = dict(zip(pose.ANGLE_NAMES, pose.joint_angles(stick_figure("T"))))
        assert a["left_elbow"] == pytest.approx(180, abs=1)
        assert a["left_shoulder"] == pytest.approx(90, abs=10)
        assert a["right_knee"] == pytest.approx(180, abs=1)
        assert a["torso_tilt"] == pytest.approx(0, abs=1)

    def test_torso_tilt_horizontal(self):
        a = dict(zip(pose.ANGLE_NAMES, pose.joint_angles(stick_figure("down", lean=90))))
        assert a["torso_tilt"] == pytest.approx(90, abs=1)

    def test_classify(self):
        assert pose.classify_pose(pose.joint_angles(stick_figure("T"))) == "T Pose"
        assert pose.classify_pose(pose.joint_angles(stick_figure("down"))) == "Mountain Pose"
        assert pose.classify_pose(pose.joint_angles(stick_figure("up"))) == "Upward Salute"

    def test_invisible_joints_are_nan(self):
        lm = stick_figure("T")
        lm[25] = (lm[25][0], lm[25][1], 0.1)  # left knee hidden
        a = dict(zip(pose.ANGLE_NAMES, pose.joint_angles(lm)))
        assert np.isnan(a["left_knee"]) and np.isnan(a["left_hip"])
        assert np.isfinite(a["right_knee"])

    def test_similarity_identical_and_different(self):
        t = pose.joint_angles(stick_figure("T"))
        down = pose.joint_angles(stick_figure("down"))
        assert pose.pose_similarity(t, t) == 100.0
        assert pose.pose_similarity(t, down) < 50

    def test_similarity_mirror(self):
        bent_left = pose.joint_angles(stick_figure("T", left_knee_bend=80))
        bent_right = pose.mirror_angles(bent_left)
        assert pose.pose_similarity(bent_left, bent_right, allow_mirror=True) == 100.0
        assert pose.pose_similarity(bent_left, bent_right, allow_mirror=False) < 100.0

    def test_similarity_not_comparable(self):
        assert pose.pose_similarity([None] * pose.N_ANGLES, pose.joint_angles(stick_figure())) is None


class TestPoseSequenceComparison:
    T = pose.joint_angles(stick_figure("T"))
    DOWN = pose.joint_angles(stick_figure("down"))
    UP = pose.joint_angles(stick_figure("up"))

    def test_same_routine_with_delay_scores_high(self):
        ref = frames_from([(self.DOWN, 3), (self.T, 3), (self.UP, 3)])
        late = frames_from([(self.DOWN, 4), (self.T, 3), (self.UP, 2)])
        res = compare_pose_sequences(late, ref)
        assert res["available"] and res["method"] == "joint-angle DTW"
        assert res["pose_accuracy"] > 90

    def test_wrong_routine_scores_low(self):
        ref = frames_from([(self.DOWN, 3), (self.T, 3), (self.UP, 3)])
        wrong = frames_from([(self.UP, 9)])
        assert compare_pose_sequences(wrong, ref)["pose_accuracy"] < 50

    def test_per_pose_breakdown(self):
        ref = frames_from([(self.DOWN, 3), (self.T, 3)])
        res = compare_pose_sequences(frames_from([(self.DOWN, 3), (self.UP, 3)]), ref)
        assert res["per_pose"]["Mountain Pose"] > res["per_pose"]["T Pose"]

    def test_no_pose_data_is_unavailable(self):
        assert not compare_pose_sequences(None, frames_from([(self.T, 1)]))["available"]
        assert not compare_pose_sequences([], [])["available"]

    def test_legacy_unknown_labels_are_not_counted(self):
        legacy = [{"timestamp": i / 10, "pose": "Unknown Pose", "angles": None} for i in range(30)]
        assert not compare_pose_sequences(legacy, legacy)["available"]

    def test_live_match(self):
        ref = frames_from([(self.DOWN, 3), (self.T, 3)])
        sim, target = live_pose_match(self.T, ref, 4.0)
        assert sim == 100.0 and target == "T Pose"
        sim, _ = live_pose_match(self.UP, ref, 1.0)
        assert sim < 50


class TestBreathComparison:
    @staticmethod
    def segs(period=5.0, dur=15.0, offset=0.0):
        out, t, phase = [], offset, "INHALE"
        while t < dur:
            end = min(t + period / 2, dur)
            out.append({"phase": phase, "start": t, "end": end, "duration": end - t})
            phase = "EXHALE" if phase == "INHALE" else "INHALE"
            t = end
        return out

    def test_identical_pattern(self):
        assert compare_breath_pattern(self.segs(), self.segs())["segment_accuracy"] == 100.0

    def test_delayed_pattern_uses_lag(self):
        res = compare_breath_pattern(self.segs(offset=1.0), self.segs())
        assert res["segment_accuracy"] > 85 and res["lag_s"] == pytest.approx(1.0, abs=0.2)

    def test_faster_breathing_scores_lower(self):
        assert compare_breath_pattern(self.segs(period=2.5), self.segs())["segment_accuracy"] < 75

    def test_metric_tolerances(self):
        same = compare_breathing_patterns({"bpm": 12, "avg_exhale_s": 2.0}, {"bpm": 12, "avg_exhale_s": 2.0})
        assert same["breathing_alignment_score"] == 100.0
        far = compare_breathing_patterns({"bpm": 24}, {"bpm": 12})
        assert far["breathing_alignment_score"] == 0.0

    def test_overall_ignores_missing_components(self):
        feats = {"bpm": 12, "inhale_exhale_ratio": 1.2, "stability_cv": 0.1,
                 "avg_inhale_s": 2.0, "avg_exhale_s": 2.4}
        segs = self.segs()
        res = overall_comparison({"features": feats, "segments": segs, "pose_data": None},
                                 {"features": feats, "segments": segs, "pose_data": []})
        assert not res["pose_comparison"]["available"]
        assert res["overall_alignment_score"] == 100.0


class TestReferences:
    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(references, "REFS_DIR", str(tmp_path))
        frames = frames_from([(pose.joint_angles(stick_figure("T")), 1)])
        segs = TestBreathComparison.segs()
        result = {"features": {"bpm": 12.0}, "segments": segs, "pose_data": frames,
                  "audio": np.zeros(16000 * 15, dtype=np.float32), "sr": 16000}

        saved = references.save_reference("Teacher: Sun Salutation!", result, "video.mp4")
        assert saved["success"]
        listed = references.list_references()
        assert len(listed) == 1 and listed[0]["duration"] == 15.0

        r = references.reconstruct_result(references.load_reference(saved["ref_id"]))
        assert r["features"]["bpm"] == 12.0
        assert len(r["segments"]) == len(segs)
        assert r["pose_data"][0]["angles"] == frames[0]["angles"]
        assert compare_pose_sequences(frames, r["pose_data"])["pose_accuracy"] == 100.0

    def test_legacy_reference_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr(references, "REFS_DIR", str(tmp_path))
        d = tmp_path / "old_ref"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps(
            {"name": "Old", "created": "2026-06-06T18:06:50", "features": {"bpm": 8.0}}))
        (d / "analysis.json").write_text(json.dumps(
            {"duration": 15.0, "timestamps": [0.0, 0.1], "poses": ["Unknown Pose", "T Pose"]}))
        r = references.reconstruct_result(references.load_reference("old_ref"))
        assert r["pose_data"][1] == {"timestamp": 0.1, "pose": "T Pose", "angles": None}
        assert r["segments"] == []


class TestBpm:
    def test_bpm_from_peak_intervals(self):
        sr = 16000
        audio = create_test_audio(sr=sr, duration=15.0, freq=0.2)
        clean = preprocess(audio, sr)
        times, env = extract_envelope(clean, sr)
        peaks = detect_peaks_valleys(env, times)
        f = extract_features(segment_breathing(env, times, peaks), peaks, 15.0)
        assert f["bpm"] == pytest.approx(12.0, abs=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
