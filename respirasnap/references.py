"""
references.py — Save / load reference sessions (e.g. a yoga instructor's video)

A reference stores everything needed to compare against later, without the
raw audio or video:
  metadata.json — name, date, source, breathing features
  analysis.json — duration, breath segments, pose frames with joint angles
"""

import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


REFS_DIR = os.path.join(os.path.dirname(__file__), "references")


def ensure_refs_dir():
    os.makedirs(REFS_DIR, exist_ok=True)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (s or "reference")[:30]


def _duration(result: Dict) -> float:
    audio = result.get("audio")
    if audio is not None and len(audio):
        return float(len(audio) / result.get("sr", 16000))
    return float(result.get("features", {}).get("audio_duration_s", 0.0))


def _pose_frames(result: Dict) -> List[Dict]:
    """Pose frames without image data, JSON-safe."""
    out = []
    for p in result.get("pose_data") or []:
        angles = p.get("angles")
        if angles is not None:
            angles = [None if (v is None or not np.isfinite(v)) else round(float(v), 1)
                      for v in angles]
        out.append({"timestamp": round(float(p["timestamp"]), 3),
                    "pose": p.get("pose", ""), "angles": angles})
    return out


def save_reference(name: str, result_data: Dict, original_path: str = None) -> Dict:
    """Save an analysis result as a named reference.

    Returns {"success", "ref_id", "message", "path"}.
    """
    ensure_refs_dir()
    ref_id = f"{_slug(name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ref_path = os.path.join(REFS_DIR, ref_id)

    try:
        os.makedirs(ref_path, exist_ok=True)
        frames = _pose_frames(result_data)
        segments = [{"phase": s["phase"], "start": round(float(s["start"]), 3),
                     "end": round(float(s["end"]), 3),
                     "duration": round(float(s["duration"]), 3)}
                    for s in result_data.get("segments", [])]

        metadata = {
            "name": name,
            "ref_id": ref_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "original_path": original_path or "unknown",
            "features": result_data.get("features", {}),
            "pose_data_count": len(frames),
            "format": 2,
        }
        analysis = {
            "features": result_data.get("features", {}),
            "duration": round(_duration(result_data), 2),
            "segment_count": len(segments),
            "segments": segments,
            "pose_frames": frames,
            # kept for older readers
            "timestamps": [f["timestamp"] for f in frames],
            "poses": [f["pose"] for f in frames],
        }

        with open(os.path.join(ref_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with open(os.path.join(ref_path, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump(analysis, f)

        return {"success": True, "ref_id": ref_id,
                "message": f"Reference saved: {name}", "path": ref_path}
    except Exception as e:
        return {"success": False, "ref_id": None,
                "message": f"Failed to save reference: {e}", "path": None}


def load_reference(ref_id: str) -> Optional[Dict]:
    """Load a saved reference: {"metadata", "analysis", "ref_id"} or None."""
    ref_path = os.path.join(REFS_DIR, ref_id)
    if not os.path.isdir(ref_path):
        return None
    try:
        with open(os.path.join(ref_path, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)
        with open(os.path.join(ref_path, "analysis.json"), encoding="utf-8") as f:
            analysis = json.load(f)
        return {"metadata": metadata, "analysis": analysis, "ref_id": ref_id}
    except Exception as e:
        print(f"Error loading reference {ref_id}: {e}")
        return None


def list_references() -> List[Dict]:
    """[{"ref_id", "name", "created", "duration", "poses", "label"}, ...] newest first."""
    ensure_refs_dir()
    refs = []
    for item in os.listdir(REFS_DIR):
        data = load_reference(item)
        if not data:
            continue
        meta, analysis = data["metadata"], data["analysis"]
        frames = analysis.get("pose_frames") or analysis.get("poses") or []
        created = meta.get("created", "")
        refs.append({
            "ref_id": item,
            "name": meta.get("name", item),
            "created": created[:10],
            "created_full": created,
            "duration": round(float(analysis.get("duration", 0)), 1),
            "poses": len(frames),
            "label": f"{meta.get('name', item)} ({created[:16].replace('T', ' ')})",
        })
    return sorted(refs, key=lambda r: r["created_full"], reverse=True)


def delete_reference(ref_id: str) -> Dict:
    ref_path = os.path.join(REFS_DIR, ref_id)
    if not os.path.isdir(ref_path):
        return {"success": False, "message": "Reference not found"}
    try:
        shutil.rmtree(ref_path)
        return {"success": True, "message": f"Reference deleted: {ref_id}"}
    except Exception as e:
        return {"success": False, "message": f"Failed to delete: {e}"}


def reference_from_json(data: Dict, name: str) -> Optional[Dict]:
    """Build a comparison-ready result from a loaded analysis/metadata JSON."""
    if "pose_frames" in data or "timestamps" in data or "segments" in data:
        return reconstruct_result({"analysis": data, "metadata": {"features": data.get("features", {}),
                                                                 "name": name}})
    if "features" in data:
        return reconstruct_result({"analysis": {"duration": data["features"].get("audio_duration_s", 0)},
                                   "metadata": data})
    return None


def reconstruct_result(ref_data: Dict, audio: np.ndarray = None, sr: int = 16000,
                       pose_data: List = None) -> Dict:
    """Turn a saved reference into the same shape as a live analysis result."""
    analysis = ref_data["analysis"]
    metadata = ref_data["metadata"]

    if pose_data is None:
        if analysis.get("pose_frames") is not None:
            pose_data = analysis["pose_frames"]
        elif "timestamps" in analysis and "poses" in analysis:  # format 1
            pose_data = [{"timestamp": t, "pose": p, "angles": None}
                         for t, p in zip(analysis["timestamps"], analysis["poses"])]

    duration = float(analysis.get("duration") or metadata.get("features", {}).get("audio_duration_s", 0))
    if audio is None:
        audio = np.zeros(max(1, int(duration * sr)), dtype=np.float32)

    return {
        "features": metadata.get("features") or analysis.get("features", {}),
        "segments": analysis.get("segments", []),
        "pose_data": pose_data or [],
        "duration": duration,
        "audio": audio,
        "sr": sr,
        "envelope": np.array([]),
        "times": np.array([]),
        "peaks": {"peak_indices": np.array([], dtype=int), "valley_indices": np.array([], dtype=int)},
        "path": metadata.get("name", ref_data.get("ref_id", "reference")),
    }
