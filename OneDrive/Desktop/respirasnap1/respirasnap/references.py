"""
references.py — Manage saved reference videos for comparison
"""

import numpy as np
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


REFS_DIR = os.path.join(os.path.dirname(__file__), "references")


def ensure_refs_dir():
    """Create references directory if it doesn't exist."""
    os.makedirs(REFS_DIR, exist_ok=True)


def save_reference(name: str, result_data: Dict, original_path: str = None) -> Dict:
    """Save a reference result for future comparisons.
    
    Args:
        name: Reference name (e.g., "Yoga Master - Sun Salutation")
        result_data: Analysis result {"features": {...}, "segments": [...], ...}
        original_path: Original video path (stored as metadata)
    
    Returns:
        {"success": bool, "ref_id": str, "message": str}
    """
    ensure_refs_dir()
    
    # Generate reference ID
    ref_id = name.lower().replace(" ", "_")[:30]
    ref_id = f"{ref_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    ref_path = os.path.join(REFS_DIR, ref_id)
    os.makedirs(ref_path, exist_ok=True)
    
    try:
        # Save metadata
        metadata = {
            "name": name,
            "ref_id": ref_id,
            "created": datetime.now().isoformat(),
            "original_path": original_path or "unknown",
            "features": result_data.get("features", {}),
            "pose_data_count": len(result_data.get("pose_data", [])) if result_data.get("pose_data") else 0,
        }
        
        # Save analysis data (excluding large arrays)
        analysis_data = {
            "features": result_data.get("features", {}),
            "segment_count": len(result_data.get("segments", [])),
            "duration": len(result_data.get("audio", [])) / result_data.get("sr", 16000) if result_data.get("audio") is not None else 0,
            "timestamps": [p["timestamp"] for p in result_data.get("pose_data", [])],
            "poses": [p["pose"] for p in result_data.get("pose_data", [])],
        }
        
        with open(os.path.join(ref_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        with open(os.path.join(ref_path, "analysis.json"), "w") as f:
            json.dump(analysis_data, f, indent=2)
        
        return {
            "success": True,
            "ref_id": ref_id,
            "message": f"Reference saved: {name}",
            "path": ref_path
        }
    except Exception as e:
        return {
            "success": False,
            "ref_id": None,
            "message": f"Failed to save reference: {e}",
            "path": None
        }


def load_reference(ref_id: str) -> Optional[Dict]:
    """Load a saved reference.
    
    Args:
        ref_id: Reference ID
    
    Returns:
        Reference data dict or None if not found
    """
    ref_path = os.path.join(REFS_DIR, ref_id)
    
    if not os.path.exists(ref_path):
        return None
    
    try:
        with open(os.path.join(ref_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        with open(os.path.join(ref_path, "analysis.json"), "r") as f:
            analysis = json.load(f)
        
        return {
            "metadata": metadata,
            "analysis": analysis,
            "ref_id": ref_id,
        }
    except Exception as e:
        print(f"Error loading reference {ref_id}: {e}")
        return None


def list_references() -> List[Dict]:
    """List all saved references.
    
    Returns:
        [{"ref_id": ..., "name": ..., "created": ..., "duration": ...}, ...]
    """
    ensure_refs_dir()
    
    refs = []
    for item in os.listdir(REFS_DIR):
        ref_path = os.path.join(REFS_DIR, item)
        if not os.path.isdir(ref_path):
            continue
        
        try:
            with open(os.path.join(ref_path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            
            with open(os.path.join(ref_path, "analysis.json"), "r") as f:
                analysis = json.load(f)
            
            refs.append({
                "ref_id": item,
                "name": metadata["name"],
                "created": metadata["created"][:10],  # Date only
                "duration": round(analysis["duration"], 1),
                "poses": len(analysis.get("poses", [])),
            })
        except Exception:
            continue
    
    return sorted(refs, key=lambda x: x["created"], reverse=True)


def delete_reference(ref_id: str) -> Dict:
    """Delete a saved reference.
    
    Args:
        ref_id: Reference ID to delete
    
    Returns:
        {"success": bool, "message": str}
    """
    ref_path = os.path.join(REFS_DIR, ref_id)
    
    if not os.path.exists(ref_path):
        return {"success": False, "message": "Reference not found"}
    
    try:
        shutil.rmtree(ref_path)
        return {"success": True, "message": f"Reference deleted: {ref_id}"}
    except Exception as e:
        return {"success": False, "message": f"Failed to delete: {e}"}


def reconstruct_result(ref_data: Dict, audio: np.ndarray, sr: int, 
                       pose_data: List = None) -> Dict:
    """Reconstruct a result object from saved reference for comparison.
    
    This allows comparison to work with the minimal stored data.
    """
    import numpy as np
    
    analysis = ref_data["analysis"]
    metadata = ref_data["metadata"]
    
    # Reconstruct pose_data from saved poses
    if pose_data is None and "timestamps" in analysis and "poses" in analysis:
        pose_data = [
            {"timestamp": t, "pose": p}
            for t, p in zip(analysis["timestamps"], analysis["poses"])
        ]
    
    return {
        "features": metadata["features"],
        "pose_data": pose_data or [],
        "audio": audio,
        "sr": sr,
        # These are placeholders; they won't be used for comparison
        "segments": [],
        "envelope": np.array([]),
        "times": np.array([]),
        "peaks": {"peak_indices": np.array([]), "valley_indices": np.array([])},
    }
