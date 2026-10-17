"""
export.py — Export breathing sessions in multiple formats (CSV, JSON)
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


def export_session_csv(session_data: Dict, filename: str) -> Dict:
    """Export a single session as CSV with metrics.

    Args:
        session_data: {"features": {...}, "score": int, ...}
        filename: Output CSV file path

    Returns:
        {"success": bool, "message": str, "file": str}
    """
    try:
        features = session_data.get("features", {})
        score = session_data.get("score", {})

        rows = [
            ["Metric", "Value", "Unit"],
            ["", "", ""],
            ["BREATHING SCORE", score, "/100"],
            ["", "", ""],
            ["BREATHING METRICS", "", ""],
            ["BPM (Breaths per Minute)", f"{features.get('bpm', 0):.2f}", "breaths/min"],
            ["Inhale/Exhale Ratio", f"{features.get('inhale_exhale_ratio', 0):.2f}", "x"],
            ["Breathing Stability", f"{features.get('stability_cv', 0):.4f}", "CV (lower=better)"],
            ["Average Inhale Duration", f"{features.get('avg_inhale_s', 0):.2f}", "seconds"],
            ["Average Exhale Duration", f"{features.get('avg_exhale_s', 0):.2f}", "seconds"],
            ["", "", ""],
            ["BREATH PATTERN", "", ""],
            ["Breath Cycles Detected", features.get("num_cycles", 0), "cycles"],
            ["Breath Holds Detected", features.get("holds_detected", 0), "holds"],
            ["Breathing Interruptions", features.get("interruptions", 0), "count"],
            ["", "", ""],
            ["SESSION INFO", "", ""],
            ["Recording Duration", f"{features.get('audio_duration_s', 0):.2f}", "seconds"],
            ["Timestamp", session_data.get("timestamp", "unknown"), ""],
        ]

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return {
            "success": True,
            "message": f"Session exported: {filename}",
            "file": filename
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to export session: {e}",
            "file": None
        }


def export_history_csv(history_list: List[Dict], filename: str) -> Dict:
    """Export all sessions as bulk CSV for trending analysis.

    Args:
        history_list: List of session dicts from load_history()
        filename: Output CSV file path

    Returns:
        {"success": bool, "message": str, "file": str, "count": int}
    """
    try:
        if not history_list:
            return {
                "success": False,
                "message": "No sessions to export",
                "file": None,
                "count": 0
            }

        fieldnames = [
            "timestamp", "score", "bpm", "inhale_exhale_ratio", "stability_cv",
            "avg_inhale_s", "avg_exhale_s", "num_cycles", "holds_detected",
            "interruptions", "audio_duration_s"
        ]

        rows = []
        for session in history_list:
            features = session.get("features", {})
            row = {
                "timestamp": session.get("timestamp", ""),
                "score": session.get("score", 0),
                "bpm": features.get("bpm", 0),
                "inhale_exhale_ratio": features.get("inhale_exhale_ratio", 0),
                "stability_cv": features.get("stability_cv", 0),
                "avg_inhale_s": features.get("avg_inhale_s", 0),
                "avg_exhale_s": features.get("avg_exhale_s", 0),
                "num_cycles": features.get("num_cycles", 0),
                "holds_detected": features.get("holds_detected", 0),
                "interruptions": features.get("interruptions", 0),
                "audio_duration_s": features.get("audio_duration_s", 0),
            }
            rows.append(row)

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return {
            "success": True,
            "message": f"History exported: {len(history_list)} sessions",
            "file": filename,
            "count": len(history_list)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to export history: {e}",
            "file": None,
            "count": 0
        }


def export_session_json(session_data: Dict, filename: str) -> Dict:
    """Export a session as JSON (similar to references.py format).

    Args:
        session_data: Full session result dict
        filename: Output JSON file path

    Returns:
        {"success": bool, "message": str, "file": str}
    """
    try:
        features = session_data.get("features", {})
        export_data = {
            "timestamp": session_data.get("timestamp", datetime.now().isoformat()),
            "score": session_data.get("score", 0),
            "features": features,
            "segments_count": len(session_data.get("segments", [])),
            "clinical": session_data.get("clinical", ""),
            "coaching": session_data.get("coaching", ""),
            "followup": session_data.get("followup", ""),
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        return {
            "success": True,
            "message": f"Session exported as JSON: {filename}",
            "file": filename
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to export JSON: {e}",
            "file": None
        }


def export_comparison_csv(comparison_result: Dict, filename: str) -> Dict:
    """Export comparison report as CSV.

    Args:
        comparison_result: Result from compare.overall_comparison()
        filename: Output CSV file path

    Returns:
        {"success": bool, "message": str, "file": str}
    """
    try:
        rows = [
            ["COMPARISON ANALYSIS REPORT", "", ""],
            ["Generated", datetime.now().isoformat(), ""],
            ["", "", ""],
            ["OVERALL SCORES", "", ""],
            ["Overall Alignment Score", f"{comparison_result.get('overall_alignment_score', 0)}/100", ""],
            ["", "", ""],
            ["BREATHING COMPARISON", "", ""],
            ["Breathing Alignment Score", f"{comparison_result.get('breathing_comparison', {}).get('breathing_alignment_score', 0)}/100", ""],
            ["Total Variance", f"{comparison_result.get('breathing_comparison', {}).get('total_variance', 0):.1f}%", ""],
            ["", "", ""],
            ["METRICS COMPARISON", "User", "Reference", "Difference %"],
        ]

        metrics = comparison_result.get("breathing_comparison", {}).get("metrics_comparison", {})
        for key, data in metrics.items():
            rows.append([
                data.get("label", key),
                f"{data.get('user', 0):.2f}",
                f"{data.get('reference', 0):.2f}",
                f"{data.get('diff_pct', 0):.1f}%"
            ])

        rows.extend([
            ["", "", "", ""],
            ["RECOMMENDATIONS", "", ""],
        ])

        for rec in comparison_result.get("recommendations", []):
            rows.append([rec, "", ""])

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        return {
            "success": True,
            "message": f"Comparison exported: {filename}",
            "file": filename
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to export comparison: {e}",
            "file": None
        }


def get_default_export_filename(export_type: str) -> str:
    """Generate default filename for export based on type.

    Args:
        export_type: "session_csv", "session_json", "history_csv", "comparison_csv"

    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mapping = {
        "session_csv": f"session_{timestamp}.csv",
        "session_json": f"session_{timestamp}.json",
        "history_csv": f"breathing_history_{timestamp}.csv",
        "comparison_csv": f"comparison_{timestamp}.csv",
    }
    return mapping.get(export_type, f"export_{timestamp}.csv")
