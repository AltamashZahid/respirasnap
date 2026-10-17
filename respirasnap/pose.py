"""
pose.py — Yoga posture detection and matching (MediaPipe Pose Landmarker)

Postures are compared by *joint angles*, not by labels. A frame is reduced to
9 angles (elbows, shoulders, hips, knees, torso tilt), which makes matching
independent of body size, distance from the camera and position in frame.
Labels like "Warrior II" are only for display.
"""

import math
import os
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_OK, MP_ERR = True, ""
except Exception as e:  # mediapipe missing or broken install
    mp = None
    MP_OK, MP_ERR = False, str(e)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

# (name, a, b, c) → interior angle at b, in degrees [0, 180]
JOINTS = [
    ("left_elbow",     11, 13, 15),
    ("right_elbow",    12, 14, 16),
    ("left_shoulder",  13, 11, 23),
    ("right_shoulder", 14, 12, 24),
    ("left_hip",       11, 23, 25),
    ("right_hip",      12, 24, 26),
    ("left_knee",      23, 25, 27),
    ("right_knee",     24, 26, 28),
]
ANGLE_NAMES = [j[0] for j in JOINTS] + ["torso_tilt"]
N_ANGLES = len(ANGLE_NAMES)

# Swapping these index pairs mirrors a pose (left ↔ right)
_MIRROR = [1, 0, 3, 2, 5, 4, 7, 6, 8]

MIN_VISIBILITY = 0.5
# RMS angle error (degrees) at which similarity reaches 0
SIMILARITY_SCALE_DEG = 60.0

NOT_A_POSE = {"—", "No pose detected", "Detection error",
              "Detecting…", "Pose model unavailable"}


# ── Model / detector ──────────────────────────────────────

def ensure_model() -> bool:
    """Download the pose model if missing. Returns True if it is available."""
    if not MP_OK:
        return False
    if os.path.exists(MODEL_PATH):
        return True
    try:
        print("Downloading pose model (~5 MB)…", end=" ", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("done.")
        return True
    except Exception as e:
        print(f"\nWarning: pose model download failed: {e}")
        return False


def build_detector():
    """Create a PoseLandmarker, or None if unavailable.

    A detector must only be used from one thread at a time, so create a
    separate one for each worker thread (e.g. reference video processing).
    """
    if not MP_OK or not os.path.exists(MODEL_PATH):
        return None
    try:
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.PoseLandmarker.create_from_options(opts)
    except Exception as e:
        print(f"Pose detector init failed: {e}")
        return None


def detect_landmarks(detector, bgr):
    """Run pose detection on a BGR frame.

    Returns a list of 33 (x_px, y_px, visibility) tuples, or None.
    """
    if detector is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = detector.detect(img)
    if not res.pose_landmarks:
        return None
    h, w = bgr.shape[:2]
    return [(lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0) or 0.0)
            for lm in res.pose_landmarks[0]]


# ── Angles ────────────────────────────────────────────────

def _interior_angle(a, b, c):
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def joint_angles(landmarks) -> list:
    """Landmarks → list of N_ANGLES floats (NaN where joints are not visible).

    torso_tilt: 0 = upright, 90 = horizontal (plank), 180 = upside down.
    """
    if not landmarks or len(landmarks) < 29:
        return [float("nan")] * N_ANGLES

    def vis(i):
        return len(landmarks[i]) < 3 or landmarks[i][2] >= MIN_VISIBILITY

    out = []
    for _, a, b, c in JOINTS:
        if vis(a) and vis(b) and vis(c):
            out.append(_interior_angle(landmarks[a], landmarks[b], landmarks[c]))
        else:
            out.append(float("nan"))

    if all(vis(i) for i in (11, 12, 23, 24)):
        sx = (landmarks[11][0] + landmarks[12][0]) / 2
        sy = (landmarks[11][1] + landmarks[12][1]) / 2
        hx = (landmarks[23][0] + landmarks[24][0]) / 2
        hy = (landmarks[23][1] + landmarks[24][1]) / 2
        # Angle between hip→shoulder and straight up (image y grows downward)
        out.append(_interior_angle((hx, hy - 100.0), (hx, hy), (sx, sy)))
    else:
        out.append(float("nan"))
    return out


def mirror_angles(angles) -> list:
    return [angles[i] for i in _MIRROR]


def _angle_array(angles):
    """None / list → float array of N_ANGLES (all NaN if missing)."""
    if angles is None:
        return np.full(N_ANGLES, np.nan)
    return np.array([np.nan if v is None else v for v in angles], dtype=float)


def angle_errors(user, ref, allow_mirror=True):
    """Per-joint absolute error (degrees) between two angle vectors.

    If allow_mirror, the reference is also compared mirrored (students often
    face the instructor) and the closer orientation is used.
    """
    u, r = _angle_array(user), _angle_array(ref)
    err = np.abs(u - r)
    if allow_mirror:
        err_m = np.abs(u - r[_MIRROR])
        if _rms_finite(err_m) < _rms_finite(err):
            err = err_m
    return err


def _rms_finite(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("inf")


def similarity_from_rms(rms_deg):
    return max(0.0, 100.0 * (1.0 - rms_deg / SIMILARITY_SCALE_DEG))


def pose_similarity(user, ref, allow_mirror=True, min_joints=4):
    """Similarity 0–100 between two angle vectors, or None if not comparable.

    Uses RMS angle error so one badly placed limb (e.g. arms down instead of
    out) counts heavily instead of being averaged away by the matching joints.
    """
    err = angle_errors(user, ref, allow_mirror)
    valid = np.isfinite(err)
    if valid.sum() < min_joints:
        return None
    rms = float(np.sqrt(np.mean(err[valid] ** 2)))
    return round(similarity_from_rms(rms), 1)


# ── Classification (display only) ─────────────────────────

def classify_pose(angles) -> str:
    """Name a few common yoga poses from joint angles. Display only."""
    a = dict(zip(ANGLE_NAMES, _angle_array(angles)))
    if sum(np.isfinite(v) for v in a.values()) < 5:
        return "Unknown Pose"

    def between(name, lo, hi):
        v = a[name]
        return bool(np.isfinite(v) and lo <= v <= hi)

    def both(prefix, lo, hi):
        # Side views hide one limb: require every *visible* side to be in range
        sides = [a[f"left_{prefix}"], a[f"right_{prefix}"]]
        seen = [v for v in sides if np.isfinite(v)]
        return bool(seen) and all(lo <= v <= hi for v in seen)

    def one_each(prefix, lo1, hi1, lo2, hi2):
        l, r = f"left_{prefix}", f"right_{prefix}"
        return ((between(l, lo1, hi1) and between(r, lo2, hi2)) or
                (between(r, lo1, hi1) and between(l, lo2, hi2)))

    straight_legs = both("knee", 155, 180)
    straight_arms = both("elbow", 150, 180)
    tilt = a["torso_tilt"]
    upright = np.isfinite(tilt) and tilt < 35
    horizontal = np.isfinite(tilt) and 55 <= tilt <= 125

    if straight_legs and both("hip", 0, 75) and not both("shoulder", 140, 180):
        return "Forward Fold"
    if straight_legs and both("hip", 40, 115) and both("shoulder", 140, 180):
        return "Downward Dog"
    if horizontal and straight_legs and straight_arms and both("hip", 155, 180):
        return "Plank"
    if upright:
        if one_each("knee", 155, 180, 0, 110) and not one_each("knee", 155, 180, 70, 140):
            return "Tree Pose"
        if one_each("knee", 70, 140, 150, 180) and both("shoulder", 65, 115) and straight_arms:
            return "Warrior II"
        if both("knee", 60, 145) and both("hip", 60, 150):
            return "Chair Pose"
        if straight_legs and both("shoulder", 65, 115) and straight_arms:
            return "T Pose"
        if straight_legs and both("shoulder", 145, 180):
            return "Upward Salute"
        if straight_legs and both("shoulder", 0, 35) and both("hip", 160, 180):
            return "Mountain Pose"
    return "Unknown Pose"


# ── Drawing ───────────────────────────────────────────────

def draw_pose(bgr, landmarks, label, match=None, target=None):
    """Draw skeleton + label (+ optional live match %) on a copy of the frame."""
    out = bgr.copy()
    if landmarks:
        pts = [(int(x), int(y)) for x, y, *_ in landmarks]
        for a, b in SKELETON:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], (0, 200, 0), 2)
        for i, p in enumerate(pts):
            if i >= 11:
                cv2.circle(out, p, 4, (0, 230, 230), -1)

    color = (0, 220, 0) if label not in NOT_A_POSE and label != "Unknown Pose" else (0, 140, 255)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if target:
        cv2.putText(out, f"Target: {target}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 80), 1)
    if match is not None:
        mc = (0, 220, 0) if match >= 75 else (0, 200, 255) if match >= 50 else (60, 60, 240)
        cv2.putText(out, f"Match {match:.0f}%", (10, out.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mc, 2)
    return out


def analyze_frame(detector, bgr):
    """Detect → angles → label. Returns dict with landmarks, angles, pose."""
    if detector is None:
        return {"landmarks": None, "angles": None, "pose": "Pose model unavailable"}
    try:
        lms = detect_landmarks(detector, bgr)
    except Exception:
        return {"landmarks": None, "angles": None, "pose": "Detection error"}
    if lms is None:
        return {"landmarks": None, "angles": None, "pose": "No pose detected"}
    angles = joint_angles(lms)
    return {"landmarks": lms, "angles": angles, "pose": classify_pose(angles)}


def fit_frame(frame, width, height):
    """Scale to `height`, centre-crop to `width` (keeps aspect ratio)."""
    h, w = frame.shape[:2]
    scale = height / h
    nw = max(1, int(w * scale))
    frame = cv2.resize(frame, (nw, height))
    if nw > width:
        x0 = (nw - width) // 2
        frame = frame[:, x0:x0 + width]
    return frame


def extract_poses_from_video(video_path, detector, target_fps=10, max_seconds=None,
                             width=640, height=480):
    """Sample a video at ~target_fps and return pose frames.

    Returns [{"timestamp": s, "pose": label, "angles": [...]}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0
    step = max(1, int(round(fps / target_fps)))

    frames, idx = [], 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = idx / fps
            if max_seconds and t > max_seconds:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    res = analyze_frame(detector, fit_frame(frame, width, height))
                    frames.append({"timestamp": round(t, 3), "pose": res["pose"],
                                   "angles": clean_angles(res["angles"])})
            idx += 1
    finally:
        cap.release()
    return frames


def clean_angles(angles):
    """NaN → None and round, so frames are JSON-serialisable."""
    if angles is None:
        return None
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 1) for v in angles]
