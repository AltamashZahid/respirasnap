"""
compare.py — Comparative analysis of user breathing + poses vs reference video
"""

import numpy as np
from typing import Dict, List, Tuple


def compare_pose_sequences(user_poses: List[Dict], ref_poses: List[Dict]) -> Dict:
    """Compare user pose sequence against reference.
    
    Args:
        user_poses: [{"timestamp": t, "pose": label}, ...]
        ref_poses: [{"timestamp": t, "pose": label}, ...]
    
    Returns:
        {
            "pose_accuracy": float (0-100),
            "matching_poses": int,
            "total_poses": int,
            "pose_timeline": [...],  # detailed comparison by time
        }
    """
    if not user_poses or not ref_poses:
        return {"pose_accuracy": 0, "matching_poses": 0, "total_poses": 0, "pose_timeline": []}
    
    # Create time-indexed pose dicts
    user_dict = {p["timestamp"]: p["pose"] for p in user_poses}
    ref_dict = {p["timestamp"]: p["pose"] for p in ref_poses}
    
    # Find common timepoints
    user_times = set(user_dict.keys())
    ref_times = set(ref_dict.keys())
    
    # For comparison, snap user poses to nearest reference timestamps
    all_times = sorted(ref_times)
    matches = 0
    total = 0
    pose_timeline = []
    
    for ref_time in all_times:
        ref_pose = ref_dict[ref_time]
        
        # Find closest user pose
        closest_user_time = min(user_times, key=lambda t: abs(t - ref_time)) if user_times else None
        
        if closest_user_time:
            user_pose = user_dict[closest_user_time]
            is_match = (user_pose == ref_pose)
            if is_match:
                matches += 1
            total += 1
            
            pose_timeline.append({
                "time": ref_time,
                "reference": ref_pose,
                "user": user_pose,
                "match": is_match,
                "time_offset": abs(closest_user_time - ref_time)
            })
    
    accuracy = (matches / total * 100) if total > 0 else 0
    
    return {
        "pose_accuracy": round(accuracy, 1),
        "matching_poses": matches,
        "total_poses": total,
        "pose_timeline": pose_timeline,
    }


def compare_breathing_patterns(user_features: Dict, ref_features: Dict) -> Dict:
    """Compare user breathing metrics against reference.
    
    Args:
        user_features: {"bpm": ..., "inhale_exhale_ratio": ..., ...}
        ref_features: same structure
    
    Returns:
        {
            "bpm_diff": float,
            "ie_ratio_diff": float,
            "stability_diff": float,
            "breathing_alignment_score": float (0-100),
            "metrics_comparison": {...}
        }
    """
    metrics = {}
    total_diffs = []
    
    # Key metrics to compare
    compare_keys = [
        ("bpm", "BPM", "breaths/min"),
        ("inhale_exhale_ratio", "IE Ratio", "x"),
        ("stability_cv", "Stability CV", "%"),
        ("avg_inhale_s", "Avg Inhale", "s"),
        ("avg_exhale_s", "Avg Exhale", "s"),
    ]
    
    for key, label, unit in compare_keys:
        user_val = user_features.get(key, 0)
        ref_val = ref_features.get(key, 0)
        
        if ref_val and user_val:
            # Calculate percentage difference
            pct_diff = abs(user_val - ref_val) / ref_val * 100 if ref_val != 0 else 0
            total_diffs.append(pct_diff)
            
            metrics[key] = {
                "user": round(user_val, 2),
                "reference": round(ref_val, 2),
                "diff_pct": round(pct_diff, 1),
                "label": label,
                "unit": unit,
            }
    
    # Alignment score: lower diff is better
    avg_diff = np.mean(total_diffs) if total_diffs else 100
    alignment_score = max(0, 100 - avg_diff)
    
    return {
        "breathing_alignment_score": round(alignment_score, 1),
        "metrics_comparison": metrics,
        "total_variance": round(avg_diff, 1),
    }


def overall_comparison(user_result: Dict, ref_result: Dict) -> Dict:
    """Generate overall comparison report.
    
    Args:
        user_result: {"features": {...}, "segments": [...], "pose_data": [...]}
        ref_result: same structure
    
    Returns:
        Comprehensive comparison report
    """
    pose_cmp = compare_pose_sequences(
        user_result.get("pose_data", []),
        ref_result.get("pose_data", [])
    )
    
    breathing_cmp = compare_breathing_patterns(
        user_result.get("features", {}),
        ref_result.get("features", {})
    )
    
    # Segment-by-segment comparison
    user_segs = user_result.get("segments", [])
    ref_segs = ref_result.get("segments", [])
    
    segment_comparison = _compare_segments(user_segs, ref_segs)
    
    # Calculate overall score
    weights = {"pose": 0.3, "breathing": 0.4, "segments": 0.3}
    overall_score = (
        pose_cmp["pose_accuracy"] * weights["pose"] +
        breathing_cmp["breathing_alignment_score"] * weights["breathing"] +
        segment_comparison["segment_accuracy"] * weights["segments"]
    )
    
    return {
        "overall_alignment_score": round(overall_score, 1),
        "pose_comparison": pose_cmp,
        "breathing_comparison": breathing_cmp,
        "segment_comparison": segment_comparison,
        "recommendations": _generate_recommendations(
            pose_cmp, breathing_cmp, segment_comparison
        ),
    }


def _compare_segments(user_segs: List[Dict], ref_segs: List[Dict]) -> Dict:
    """Compare breathing segments (inhale/exhale/hold patterns)."""
    if not user_segs or not ref_segs:
        return {"segment_accuracy": 0, "details": []}
    
    # Count phase matches
    user_phases = [s["phase"] for s in user_segs]
    ref_phases = [s["phase"] for s in ref_segs]
    
    matches = sum(1 for u, r in zip(user_phases, ref_phases) if u == r)
    accuracy = (matches / len(ref_phases) * 100) if ref_phases else 0
    
    return {
        "segment_accuracy": round(accuracy, 1),
        "user_segment_count": len(user_segs),
        "ref_segment_count": len(ref_segs),
        "segment_matches": matches,
    }


def _generate_recommendations(pose_cmp: Dict, breathing_cmp: Dict,
                              segment_cmp: Dict) -> List[str]:
    """Generate actionable recommendations based on comparison."""
    recs = []

    # Pose recommendations
    if pose_cmp["pose_accuracy"] < 50:
        recs.append("🧘 Focus on holding the correct poses longer")
    elif pose_cmp["pose_accuracy"] < 80:
        recs.append("🧘 Practice smoother pose transitions")

    # Breathing recommendations
    metrics = breathing_cmp.get("metrics_comparison", {})
    if metrics.get("bpm", {}).get("diff_pct", 0) > 20:
        ref_bpm = metrics["bpm"]["reference"]
        recs.append(f"🫁 Adjust breathing pace to ~{ref_bpm} BPM (reference)")

    if metrics.get("inhale_exhale_ratio", {}).get("diff_pct", 0) > 30:
        recs.append("🫁 Work on inhale/exhale balance")

    if segment_cmp["segment_accuracy"] < 60:
        recs.append("📊 Try to match the reference's inhale/exhale/hold sequence")

    # Overall
    if len(recs) == 0:
        recs.append("✨ Excellent alignment with reference!")

    return recs


def format_comparison_for_display(comparison_result: Dict) -> str:
    """Format comparison result as readable text for UI display.

    Args:
        comparison_result: Result from overall_comparison()

    Returns:
        Formatted string for display in UI
    """
    overall_score = comparison_result.get("overall_alignment_score", 0)
    pose_cmp = comparison_result.get("pose_comparison", {})
    breathing_cmp = comparison_result.get("breathing_comparison", {})
    recs = comparison_result.get("recommendations", [])

    # Score indicator
    if overall_score >= 80:
        score_emoji = "✨"
    elif overall_score >= 60:
        score_emoji = "👍"
    else:
        score_emoji = "💪"

    text = f"""
╔═════════════════════════════════════════════════╗
║          ⚖️  COMPARISON ANALYSIS                 ║
╚═════════════════════════════════════════════════╝

{score_emoji} Overall Alignment Score:  {overall_score}/100

─────────────────────────────────────────────────

📊 BREATHING METRICS COMPARISON:

"""

    metrics = breathing_cmp.get("metrics_comparison", {})
    for key, data in metrics.items():
        label = data.get("label", key)
        user_val = data.get("user", 0)
        ref_val = data.get("reference", 0)
        diff = data.get("diff_pct", 0)
        unit = data.get("unit", "")

        # Color indicator based on difference
        if diff < 10:
            indicator = "✓"
        elif diff < 25:
            indicator = "⚠"
        else:
            indicator = "✗"

        text += f"  {indicator} {label:20s}  You: {user_val:6.2f}  Ref: {ref_val:6.2f}  Diff: {diff:5.1f}%\n"

    text += f"\n  Breathing Alignment Score: {breathing_cmp.get('breathing_alignment_score', 0)}/100\n"

    # Pose comparison if available
    if pose_cmp.get("pose_accuracy", 0) > 0:
        text += f"\n🧘 POSE ACCURACY: {pose_cmp.get('pose_accuracy', 0):.1f}%\n"
        text += f"   Matching Poses: {pose_cmp.get('matching_poses', 0)}/{pose_cmp.get('total_poses', 0)}\n"

    # Segment comparison
    text += f"\n📈 BREATH PATTERN ALIGNMENT: {comparison_result.get('segment_comparison', {}).get('segment_accuracy', 0):.1f}%\n"

    # Recommendations
    text += "\n💡 RECOMMENDATIONS:\n"
    for rec in recs:
        text += f"   • {rec}\n"

    return text
