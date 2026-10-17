"""
main.py — RespriaSnap CLI entry point (GUI is in app.py)
Usage:
    python main.py                    # record from mic (15s)
    python main.py --file audio.mp3   # analyze a file
    python main.py --history          # show session history
    python main.py --demo             # synthetic signal
    python app.py                     # launch GUI
"""

import argparse
import sys
import numpy as np

for _stream in (sys.stdout, sys.stderr):  # emoji logs must not crash on cp1252 consoles
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from audio      import record_mic, load_file
from preprocess import preprocess
from envelope   import extract_envelope
from detect     import detect_peaks_valleys
from segment    import segment_breathing
from features   import extract_features
from score      import compute_score
from agents     import clinical_summary_agent, coaching_agent, followup_agent
from history    import load_history, save_session, print_history_summary
from visualize  import plot_analysis


def run_analysis(audio: np.ndarray, sr: int,
                 save_plot: str = "analysis.png",
                 run_llm: bool = True) -> dict:
    """Full pipeline: audio → metrics → score → LLM → save."""

    print("\n── Step 1: Preprocessing ─────────────────────────")
    clean = preprocess(audio, sr)

    print("── Step 2: Envelope Extraction ───────────────────")
    times, envelope = extract_envelope(clean, sr)

    print("── Step 3: Peak Detection ────────────────────────")
    peaks = detect_peaks_valleys(envelope, times)
    print(f"   Peaks: {len(peaks['peak_indices'])}  |  Valleys: {len(peaks['valley_indices'])}")

    print("── Step 4: Segmentation ──────────────────────────")
    segments = segment_breathing(envelope, times, peaks)   # pass peaks
    phase_counts = {p: sum(1 for s in segments if s["phase"] == p)
                    for p in ("INHALE", "EXHALE", "HOLD")}
    print(f"   Segments: {phase_counts}")

    print("── Step 5: Feature Extraction ────────────────────")
    duration = len(audio) / sr
    features = extract_features(segments, peaks, duration)
    _print_features(features)

    print("── Step 6: Scoring ───────────────────────────────")
    score_result = compute_score(features)
    score = score_result["score"]
    print(f"   Score: {score}/100  |  Deductions: {score_result['deductions']}")

    print("── Step 7: LLM Agents ────────────────────────────")
    if run_llm:
        history = load_history()
        print("   🤖 Clinical Summary Agent...")
        clinical = clinical_summary_agent(features)
        print("   🤖 Coaching Agent...")
        coaching = coaching_agent(features)
        print("   🤖 Follow-up Agent...")
        followup = followup_agent({"score": score, "features": features}, history)
    else:
        clinical = "[LLM disabled]"
        coaching  = "[LLM disabled]"
        followup  = "[LLM disabled]"

    print("\n── Results ───────────────────────────────────────")
    print(f"\n  Score:    {score}/100")
    print(f"\n  Clinical:\n  {clinical}")
    print(f"\n  Coaching:\n  {coaching}")
    print(f"\n  Follow-up:\n  {followup}")

    print("\n── Step 8: Visualization ─────────────────────────")
    plot_analysis(audio, sr, times, envelope, peaks, segments,
                  features, score, save_path=save_plot)

    print("── Step 9: Saving Session ────────────────────────")
    session = save_session(features, score_result, clinical, coaching, followup)

    return {
        "features": features, "score": score_result,
        "clinical": clinical, "coaching": coaching,
        "followup": followup, "segments": segments,
        "envelope": envelope, "times": times, "peaks": peaks,
    }


def _print_features(f: dict):
    for k, v in f.items():
        print(f"   {k:25s}: {v}")


def make_demo_audio(sr: int = 16000, duration: float = 15.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration))
    breath_freq = 0.25
    envelope_shape = 0.5 + 0.5 * np.sin(2 * np.pi * breath_freq * t)
    carrier = np.sin(2 * np.pi * 300 * t)
    noise   = np.random.normal(0, 0.05, len(t))
    return (envelope_shape * carrier + noise).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="RespriaSnap — Breathing Analyzer")
    parser.add_argument("--file",    type=str)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--no-llm",  action="store_true")
    parser.add_argument("--out",     type=str, default="analysis.png")
    args = parser.parse_args()

    if args.history:
        print_history_summary(); return

    print("\n╔══════════════════════════════════════╗")
    print("║      RespriaSnap — Breathing AI      ║")
    print("╚══════════════════════════════════════╝")

    SR = 16000
    if args.demo:
        print("\n🧪 Demo mode")
        audio = make_demo_audio(SR)
    elif args.file:
        audio, SR = load_file(args.file)
    else:
        audio, SR = record_mic(duration=15.0)

    run_analysis(audio, SR, save_plot=args.out, run_llm=not args.no_llm)


if __name__ == "__main__":
    main()
