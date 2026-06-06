"""
history.py — Store and retrieve session history as local JSON
"""

import json
import os
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "session_history.json")


def load_history() -> list[dict]:
    """Load all past sessions from disk."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_session(features: dict, score_result: dict,
                 clinical: str, coaching: str, followup: str) -> dict:
    """
    Append current session to history and persist to disk.
    Returns the saved session dict.
    """
    history = load_history()

    session = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "score":      score_result["score"],
        "deductions": score_result["deductions"],
        "features":   features,
        "clinical":   clinical,
        "coaching":   coaching,
        "followup":   followup,
    }

    history.append(session)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"💾 Session saved ({len(history)} total sessions)")
    return session


def print_history_summary():
    """Print a short summary of past sessions."""
    history = load_history()
    if not history:
        print("No sessions recorded yet.")
        return

    print(f"\n{'─'*50}")
    print(f"  Session History ({len(history)} sessions)")
    print(f"{'─'*50}")
    for s in history[-5:]:  # show last 5
        ts    = s.get("timestamp", "?")
        sc    = s.get("score", "?")
        bpm   = s.get("features", {}).get("bpm", "?")
        print(f"  {ts}  |  Score: {sc}/100  |  BPM: {bpm}")
    print(f"{'─'*50}\n")
