"""
compare.py — Compare a user's session (breathing + yoga posture) against a reference

Three independent components, each scored 0–100:
  • Posture   — joint-angle sequences aligned with DTW (tolerates the user
                being a little ahead/behind the reference, and mirroring)
  • Breathing — metric-by-metric closeness (BPM, IE ratio, inhale/exhale length)
  • Pattern   — how well the inhale/exhale timeline lines up in time

The overall score averages only the components that are available, so an
audio-only session is not dragged down by a missing posture score.
"""

import numpy as np
from typing import Dict, List, Optional

from pose import (ANGLE_NAMES, N_ANGLES, NOT_A_POSE, SIMILARITY_SCALE_DEG,
                  _MIRROR, _angle_array, pose_similarity, similarity_from_rms)


MAX_DTW_FRAMES = 300       # sequences are subsampled to at most this many frames
MIN_JOINTS = 4             # joints needed for a frame to be comparable
MATCH_THRESHOLD = 70.0     # a frame "matches" at or above this similarity

WEIGHTS = {"pose": 0.40, "breathing": 0.35, "pattern": 0.25}

# key, label, unit, tolerance (difference at which the metric scores 0)
BREATH_METRICS = [
    ("bpm",                 "BPM",          "breaths/min", 6.0),
    ("inhale_exhale_ratio", "IE Ratio",     "x",           1.0),
    ("stability_cv",        "Stability CV", "",            0.4),
    ("avg_inhale_s",        "Avg Inhale",   "s",           1.5),
    ("avg_exhale_s",        "Avg Exhale",   "s",           1.5),
]


# ══════════════════════════════════════════════════════════
# POSTURE
# ══════════════════════════════════════════════════════════

def _usable_frames(frames: Optional[List[Dict]]) -> List[Dict]:
    """Frames that carry enough visible joint angles to compare."""
    out = []
    for f in frames or []:
        a = f.get("angles")
        if a is not None and np.isfinite(_angle_array(a)).sum() >= MIN_JOINTS:
            out.append(f)
    return out


def _subsample(frames: List[Dict], limit: int) -> List[Dict]:
    if len(frames) <= limit:
        return frames
    idx = np.linspace(0, len(frames) - 1, limit).round().astype(int)
    return [frames[i] for i in idx]


def _error_tensor(U: np.ndarray, R: np.ndarray, allow_mirror: bool):
    """Per-cell, per-joint error (N×M×J) using the closer of normal/mirrored."""
    err = np.abs(U[:, None, :] - R[None, :, :])
    if allow_mirror:
        err_m = np.abs(U[:, None, :] - R[:, _MIRROR][None, :, :])
        use_m = _cell_rms(err_m) < _cell_rms(err)
        err = np.where(use_m[:, :, None], err_m, err)
    return err


def _cell_rms(err: np.ndarray) -> np.ndarray:
    """RMS over finite joints per cell; +inf where fewer than MIN_JOINTS."""
    finite = np.isfinite(err)
    cnt = finite.sum(axis=2)
    s = np.where(finite, err ** 2, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(s / cnt)
    rms[cnt < MIN_JOINTS] = np.inf
    return rms


def dtw_path(cost: np.ndarray, band: Optional[int] = None):
    """Classic DTW with an optional Sakoe-Chiba band. Returns list of (i, j)."""
    n, m = cost.shape
    if band is None:
        band = max(n, m)
    band = max(band, abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])

    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        k = int(np.argmin(steps))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def compare_pose_sequences(user_poses: Optional[List[Dict]],
                           ref_poses: Optional[List[Dict]],
                           allow_mirror: bool = True) -> Dict:
    """Compare user posture sequence against a reference.

    Frames: {"timestamp": s, "pose": label, "angles": [9 floats or None]}.
    Old references without angles fall back to label agreement.
    """
    empty = {"available": False, "pose_accuracy": None, "matching_poses": 0,
             "total_poses": 0, "joint_errors": {}, "per_pose": {},
             "frame_similarity": [], "method": None,
             "user_frames": len(user_poses or []), "ref_frames": len(ref_poses or [])}

    user = _usable_frames(user_poses)
    ref = _usable_frames(ref_poses)
    if not user or not ref:
        if user_poses and ref_poses:
            return {**empty, **_compare_labels(user_poses, ref_poses)}
        return empty

    user_s = _subsample(user, MAX_DTW_FRAMES)
    ref_s = _subsample(ref, MAX_DTW_FRAMES)
    U = np.stack([_angle_array(f["angles"]) for f in user_s])
    R = np.stack([_angle_array(f["angles"]) for f in ref_s])

    err = _error_tensor(U, R, allow_mirror)
    cell_err = _cell_rms(err)
    cost = np.where(np.isfinite(cell_err), cell_err, SIMILARITY_SCALE_DEG * 1.5)

    band = int(0.35 * max(len(user_s), len(ref_s))) + 1
    path = dtw_path(cost, band)

    sims, joint_sum, joint_cnt = [], np.zeros(N_ANGLES), np.zeros(N_ANGLES)
    per_pose: Dict[str, List[float]] = {}
    best_by_user: Dict[int, float] = {}
    for i, j in path:
        if not np.isfinite(cell_err[i, j]):
            continue
        sim = similarity_from_rms(cell_err[i, j])
        sims.append(sim)
        e = err[i, j]
        ok = np.isfinite(e)
        joint_sum[ok] += e[ok]
        joint_cnt[ok] += 1
        label = ref_s[j].get("pose", "")
        if label and label not in NOT_A_POSE and label != "Unknown Pose":
            per_pose.setdefault(label, []).append(sim)
        best_by_user[i] = max(sim, best_by_user.get(i, 0.0))

    if not sims:
        return empty

    joint_errors = {ANGLE_NAMES[k]: round(float(joint_sum[k] / joint_cnt[k]), 1)
                    for k in range(N_ANGLES) if joint_cnt[k] > 0}

    return {
        "available": True,
        "method": "joint-angle DTW",
        "pose_accuracy": round(float(np.mean(sims)), 1),
        "matching_poses": int(sum(s >= MATCH_THRESHOLD for s in sims)),
        "total_poses": len(sims),
        "joint_errors": joint_errors,
        "per_pose": {k: round(float(np.mean(v)), 1) for k, v in per_pose.items()},
        "frame_similarity": [(float(user_s[i]["timestamp"]), round(s, 1))
                             for i, s in sorted(best_by_user.items())],
        "user_frames": len(user_poses or []),
        "ref_frames": len(ref_poses or []),
    }


def _compare_labels(user_poses: List[Dict], ref_poses: List[Dict]) -> Dict:
    """Fallback for references saved before joint angles were recorded."""
    informative = [p for p in ref_poses
                   if p.get("pose") not in NOT_A_POSE and p.get("pose") != "Unknown Pose"]
    if not informative:
        return {}
    user_t = np.array([p["timestamp"] for p in user_poses], dtype=float)
    matches = 0
    for rp in informative:
        k = int(np.argmin(np.abs(user_t - rp["timestamp"])))
        matches += user_poses[k].get("pose") == rp["pose"]
    return {"available": True, "method": "pose labels (legacy reference)",
            "pose_accuracy": round(100.0 * matches / len(informative), 1),
            "matching_poses": matches, "total_poses": len(informative)}


def live_pose_match(user_angles, ref_frames: Optional[List[Dict]], t: float,
                    window_s: float = 1.5):
    """Similarity of the current user pose vs the reference pose around time t.

    Looks ±window_s around t and takes the best match so small lags don't hurt.
    Returns (similarity or None, target label or None).
    """
    if not ref_frames or user_angles is None:
        return None, None
    best, target = None, None
    for f in ref_frames:
        if abs(f["timestamp"] - t) <= window_s and f.get("angles") is not None:
            s = pose_similarity(user_angles, f["angles"])
            if s is not None and (best is None or s > best):
                best, target = s, f.get("pose")
    if target in NOT_A_POSE or target == "Unknown Pose":
        target = None
    return best, target


# ══════════════════════════════════════════════════════════
# BREATHING
# ══════════════════════════════════════════════════════════

def compare_breathing_patterns(user_features: Dict, ref_features: Dict) -> Dict:
    """Compare breathing metrics. Each metric scores 100 when equal and 0 when
    the difference reaches its tolerance (e.g. 6 BPM)."""
    metrics, matches, pcts = {}, [], []
    for key, label, unit, tol in BREATH_METRICS:
        u, r = user_features.get(key), ref_features.get(key)
        if u is None or r is None:
            continue
        u, r = float(u), float(r)
        diff = abs(u - r)
        match = max(0.0, 100.0 * (1.0 - diff / tol))
        pct = diff / abs(r) * 100 if r else (0.0 if u == r else 100.0)
        matches.append(match)
        pcts.append(pct)
        metrics[key] = {"user": round(u, 2), "reference": round(r, 2),
                        "diff_pct": round(pct, 1), "match": round(match, 1),
                        "label": label, "unit": unit}

    return {
        "available": bool(matches),
        "breathing_alignment_score": round(float(np.mean(matches)), 1) if matches else 0.0,
        "metrics_comparison": metrics,
        "total_variance": round(float(np.mean(pcts)), 1) if pcts else 0.0,
    }


_PHASE_VAL = {"INHALE": 1, "EXHALE": -1, "HOLD": 0}


def _phase_signal(segs: List[Dict], t: np.ndarray) -> np.ndarray:
    sig = np.zeros(len(t), dtype=int)
    for s in segs:
        mask = (t >= s["start"]) & (t < s["end"])
        sig[mask] = _PHASE_VAL.get(s["phase"], 0)
    return sig


def compare_breath_pattern(user_segs: List[Dict], ref_segs: List[Dict],
                           hz: float = 10.0, max_lag_s: float = 2.0) -> Dict:
    """How well the inhale/exhale timeline matches the reference over time.

    Both timelines are sampled at `hz`; the best lag within ±max_lag_s is used.
    Only moments where either side is actively inhaling/exhaling count.
    """
    if not user_segs or not ref_segs:
        return {"available": False, "segment_accuracy": None,
                "user_segment_count": len(user_segs or []),
                "ref_segment_count": len(ref_segs or [])}

    end = min(max(s["end"] for s in user_segs), max(s["end"] for s in ref_segs))
    t = np.arange(0.0, max(end, 1.0 / hz), 1.0 / hz)
    u, r = _phase_signal(user_segs, t), _phase_signal(ref_segs, t)

    best, best_lag = 0.0, 0
    max_lag = int(max_lag_s * hz)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            uu, rr = u[lag:], r[:len(r) - lag]
        else:
            uu, rr = u[:lag], r[-lag:]
        active = (uu != 0) | (rr != 0)
        if active.sum() < hz:  # need at least 1 s of breathing
            continue
        agree = float((uu[active] == rr[active]).mean())
        if agree > best:
            best, best_lag = agree, lag

    return {
        "available": True,
        "segment_accuracy": round(best * 100, 1),
        "lag_s": round(best_lag / hz, 1),
        "user_segment_count": len(user_segs),
        "ref_segment_count": len(ref_segs),
    }


# ══════════════════════════════════════════════════════════
# OVERALL
# ══════════════════════════════════════════════════════════

def overall_comparison(user_result: Dict, ref_result: Dict) -> Dict:
    """Full comparison report of a user session vs a reference."""
    pose_cmp = compare_pose_sequences(user_result.get("pose_data"),
                                      ref_result.get("pose_data"))
    breathing_cmp = compare_breathing_patterns(user_result.get("features", {}),
                                               ref_result.get("features", {}))
    pattern_cmp = compare_breath_pattern(user_result.get("segments", []),
                                         ref_result.get("segments", []))

    parts = []
    if pose_cmp["available"]:
        parts.append((pose_cmp["pose_accuracy"], WEIGHTS["pose"]))
    if breathing_cmp["available"]:
        parts.append((breathing_cmp["breathing_alignment_score"], WEIGHTS["breathing"]))
    if pattern_cmp["available"]:
        parts.append((pattern_cmp["segment_accuracy"], WEIGHTS["pattern"]))
    overall = sum(s * w for s, w in parts) / sum(w for _, w in parts) if parts else 0.0

    return {
        "overall_alignment_score": round(overall, 1),
        "pose_comparison": pose_cmp,
        "breathing_comparison": breathing_cmp,
        "segment_comparison": pattern_cmp,
        "recommendations": _generate_recommendations(
            pose_cmp, breathing_cmp, pattern_cmp,
            ref_has_pose=bool(_usable_frames(ref_result.get("pose_data"))),
            user_has_pose=bool(user_result.get("pose_data"))),
    }


_JOINT_TEXT = {
    "left_elbow": "left elbow", "right_elbow": "right elbow",
    "left_shoulder": "left arm (shoulder)", "right_shoulder": "right arm (shoulder)",
    "left_hip": "left hip", "right_hip": "right hip",
    "left_knee": "left knee", "right_knee": "right knee",
    "torso_tilt": "torso lean",
}


def _generate_recommendations(pose_cmp: Dict, breathing_cmp: Dict, pattern_cmp: Dict,
                              ref_has_pose: bool = False, user_has_pose: bool = False) -> List[str]:
    recs = []

    # Posture
    if pose_cmp.get("available"):
        acc = pose_cmp["pose_accuracy"]
        if acc < 50:
            recs.append("🧘 Your postures differ a lot from the reference — slow down and hold each pose")
        elif acc < 75:
            recs.append("🧘 Postures are close — refine alignment on the joints below")
        worst = sorted(pose_cmp.get("joint_errors", {}).items(), key=lambda kv: -kv[1])
        for name, deg in worst[:2]:
            if deg >= 20:
                recs.append(f"🧘 {_JOINT_TEXT.get(name, name).capitalize()} is off by ~{deg:.0f}° on average")
        weak = [(p, s) for p, s in pose_cmp.get("per_pose", {}).items() if s < 60]
        for p, s in sorted(weak, key=lambda kv: kv[1])[:2]:
            recs.append(f"🧘 Work on {p} ({s:.0f}% match)")
    elif ref_has_pose and not user_has_pose:
        recs.append("📷 Record with the camera on to get posture matching against this reference")
    elif ref_has_pose and user_has_pose:
        recs.append("📷 Your body wasn't clearly visible — step back so shoulders to ankles are in frame")

    # Breathing metrics
    m = breathing_cmp.get("metrics_comparison", {})
    if "bpm" in m and abs(m["bpm"]["user"] - m["bpm"]["reference"]) > 3:
        faster = m["bpm"]["user"] > m["bpm"]["reference"]
        recs.append(f"🫁 Breathe {'slower' if faster else 'faster'}: you {m['bpm']['user']:.0f} BPM "
                    f"vs reference {m['bpm']['reference']:.0f} BPM")
    if "avg_exhale_s" in m and m["avg_exhale_s"]["reference"] - m["avg_exhale_s"]["user"] > 0.5:
        recs.append(f"🫁 Lengthen your exhale to ~{m['avg_exhale_s']['reference']:.1f}s")
    if "avg_inhale_s" in m and m["avg_inhale_s"]["reference"] - m["avg_inhale_s"]["user"] > 0.5:
        recs.append(f"🫁 Take slower, fuller inhales (~{m['avg_inhale_s']['reference']:.1f}s)")

    # Timing
    if pattern_cmp.get("available") and pattern_cmp["segment_accuracy"] < 60:
        recs.append("📊 Time your inhales/exhales with the reference's movements")

    if not recs:
        recs.append("✨ Excellent alignment with reference!")
    return recs


# ══════════════════════════════════════════════════════════
# SESSION vs SESSION (history)
# ══════════════════════════════════════════════════════════

def compare_two_sessions(session1: Dict, session2: Dict,
                         name1: str = "Session 1", name2: str = "Session 2") -> Dict:
    """Compare any two sessions from history."""
    score1 = session1.get("score", 0)
    score2 = session2.get("score", 0)
    score_diff = score2 - score1
    score_trend = "improved ✓" if score_diff > 0 else "declined ✗" if score_diff < 0 else "stable ➡️"

    return {
        "session1_name": name1,
        "session2_name": name2,
        "score1": score1,
        "score2": score2,
        "score_diff": score_diff,
        "score_trend": score_trend,
        "breathing_comparison": compare_breathing_patterns(session1.get("features", {}),
                                                           session2.get("features", {})),
        "timestamp1": session1.get("timestamp", ""),
        "timestamp2": session2.get("timestamp", ""),
    }


def format_session_comparison(comparison: Dict) -> str:
    """Format session-to-session comparison for display."""
    name1 = comparison.get("session1_name", "Session 1")
    name2 = comparison.get("session2_name", "Session 2")
    score1 = comparison.get("score1", 0)
    score2 = comparison.get("score2", 0)
    score_diff = comparison.get("score_diff", 0)
    trend = comparison.get("score_trend", "unknown")

    text = (
        "📊 SESSION COMPARISON\n\n"
        f"Comparing: {name1} → {name2}\n\n"
        f"Score: {score1}/100 → {score2}/100 ({score_diff:+d}) {trend}\n\n"
        "BREATHING METRICS CHANGES:\n"
    )

    metrics = comparison.get("breathing_comparison", {}).get("metrics_comparison", {})
    if metrics:
        for data in metrics.values():
            v1, v2 = data.get("user", 0), data.get("reference", 0)
            diff = v2 - v1
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            text += f"  {arrow} {data.get('label', ''):14s} {v1:7.2f} → {v2:7.2f}  (Δ {diff:+.2f})\n"
    else:
        text += "  [No metric data available]\n"

    text += f"\n  Similarity: {comparison.get('breathing_comparison', {}).get('breathing_alignment_score', 0)}/100\n"
    text += f"\n  Session 1: {comparison.get('timestamp1', 'Unknown')}\n"
    text += f"  Session 2: {comparison.get('timestamp2', 'Unknown')}\n"
    return text


# ══════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════

def _fmt_score(v) -> str:
    return "n/a" if v is None else f"{v:.1f}/100"


def format_comparison_for_display(comparison_result: Dict) -> str:
    """Format overall_comparison() output as text for the UI."""
    overall = comparison_result.get("overall_alignment_score", 0)
    pose_cmp = comparison_result.get("pose_comparison", {})
    breathing_cmp = comparison_result.get("breathing_comparison", {})
    pattern_cmp = comparison_result.get("segment_comparison", {})
    recs = comparison_result.get("recommendations", [])

    emoji = "✨" if overall >= 80 else "👍" if overall >= 60 else "💪"
    pose_acc = pose_cmp.get("pose_accuracy")
    pose_ok = pose_cmp.get("available", pose_acc is not None) and pose_acc is not None
    pattern_acc = pattern_cmp.get("segment_accuracy")

    text = (
        "⚖️  COMPARISON ANALYSIS\n\n"
        f"{emoji} Overall Alignment Score:  {overall}/100\n\n"
        f"  🧘 Posture match:      {_fmt_score(pose_acc if pose_ok else None)}\n"
        f"  🫁 Breathing metrics:  {_fmt_score(breathing_cmp.get('breathing_alignment_score'))}\n"
        f"  📈 Breath timing:      {_fmt_score(pattern_acc)}\n"
    )

    if pose_ok:
        text += f"\n🧘 POSTURE ({pose_cmp.get('method') or 'pose'}):\n"
        text += (f"   Frames matching (≥{MATCH_THRESHOLD:.0f}%): "
                 f"{pose_cmp.get('matching_poses', 0)}/{pose_cmp.get('total_poses', 0)}\n")
        joints = pose_cmp.get("joint_errors", {})
        if joints:
            text += "   Average angle error per joint:\n"
            for name, deg in sorted(joints.items(), key=lambda kv: -kv[1]):
                mark = "✓" if deg < 12 else "⚠" if deg < 25 else "✗"
                text += f"     {mark} {_JOINT_TEXT.get(name, name):22s} {deg:5.1f}°\n"
        for p, s in pose_cmp.get("per_pose", {}).items():
            text += f"   {p}: {s:.0f}% match\n"

    text += "\n📊 BREATHING METRICS COMPARISON:\n"
    for data in breathing_cmp.get("metrics_comparison", {}).values():
        diff = data.get("diff_pct", 0)
        mark = "✓" if diff < 10 else "⚠" if diff < 25 else "✗"
        text += (f"  {mark} {data.get('label', ''):14s} You: {data.get('user', 0):6.2f}  "
                 f"Ref: {data.get('reference', 0):6.2f}  Diff: {diff:5.1f}%\n")

    lag = pattern_cmp.get("lag_s")
    if pattern_acc is not None and lag:
        text += (f"\n  Your breathing ran {abs(lag):.1f}s "
                 f"{'behind' if lag > 0 else 'ahead of'} the reference\n")

    text += "\n💡 RECOMMENDATIONS:\n"
    for rec in recs:
        text += f"   • {rec}\n"
    return text
