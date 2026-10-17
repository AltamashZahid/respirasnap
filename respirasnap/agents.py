"""
agents.py — LLM agents via Groq (free tier)
Three agents: Clinical Summary, Coaching, Follow-up
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

MODEL = "llama-3.1-8b-instant"


def _call(system_prompt: str, user_content: str) -> str:
    """Single Groq LLM call."""
    if not GROQ_API_KEY:
        return "[LLM unavailable: Set GROQ_API_KEY in .env file]"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=250,
            temperature=0.4,
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq error: {e}]"


# ── Agent 1: Clinical Summary ──────────────────────────────

CLINICAL_SYSTEM = """
You are a respiratory health assistant.
Given breathing metrics from a short audio recording (duration in audio_duration_s), write a 2-3 sentence
plain-English clinical summary. Focus on rhythm, exhale ratio, and irregularities.
Do not diagnose. Mention confidence level if data is limited. Be concise and factual.
"""

def clinical_summary_agent(features: dict) -> str:
    return _call(CLINICAL_SYSTEM,
                 f"Breathing metrics:\n{json.dumps(features, indent=2)}")


# ── Agent 2: Coaching ──────────────────────────────────────

COACHING_SYSTEM = """
You are a breathing coach.
Give ONE short specific actionable tip (1-2 sentences max).
Target the biggest weakness in the metrics.
Example: "Try extending your exhale to 1.5x your inhale length for deeper relaxation."
No preamble. Just the tip.
"""

def coaching_agent(features: dict) -> str:
    return _call(COACHING_SYSTEM,
                 f"Breathing metrics:\n{json.dumps(features, indent=2)}")


# ── Agent 3: Follow-up / Trend ─────────────────────────────

FOLLOWUP_SYSTEM = """
You are a wellness tracker reviewing breathing session history.
Write exactly 2 sentences:
1. Whether the user is improving, stable, or declining (cite score trend).
2. A concrete suggestion for their next session.
Be encouraging but honest.
"""

def followup_agent(current: dict, history: list[dict]) -> str:
    past = [
        {"timestamp": s.get("timestamp", "?"), "score": s.get("score", "?")}
        for s in history[-5:]
    ]
    user = (
        f"Current session:\n{json.dumps(current, indent=2)}\n\n"
        f"Past sessions (oldest to newest):\n{json.dumps(past, indent=2)}"
    )
    return _call(FOLLOWUP_SYSTEM, user)
