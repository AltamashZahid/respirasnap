"""
visualize.py — Plot waveform, envelope, peaks, and breath segments
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


PHASE_COLORS = {
    "INHALE": "#4FC3F7",  # blue
    "EXHALE": "#81C784",  # green
    "HOLD":   "#FFB74D",  # orange
}


def plot_analysis(audio: np.ndarray, sr: int,
                  times: np.ndarray, envelope: np.ndarray,
                  peaks: dict, segments: list[dict],
                  features: dict, score: int,
                  save_path: str = None):
    """
    Create a 3-panel analysis plot:
      Panel 1 — Raw waveform
      Panel 2 — Envelope with peaks and segment shading
      Panel 3 — Segment timeline (Gantt-style)
    """
    audio_time = np.linspace(0, len(audio) / sr, len(audio))
    duration   = len(audio) / sr

    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [2, 3, 1]})
    fig.patch.set_facecolor("#0D1117")
    for ax in axes:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#AAAAAA")
        ax.spines[:].set_color("#333333")

    # ── Panel 1: Waveform ────────────────────────────────
    ax1 = axes[0]
    ax1.plot(audio_time, audio, color="#4FC3F7", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("Amplitude", color="#AAAAAA", fontsize=9)
    ax1.set_title(
        f"RespriaSnap  |  Score: {score}/100  |  BPM: {features['bpm']}  |  "
        f"IE Ratio: {features['inhale_exhale_ratio']}x",
        color="white", fontsize=11, pad=10
    )
    ax1.set_xlim(0, duration)
    ax1.axhline(0, color="#333333", linewidth=0.5)

    # ── Panel 2: Envelope + Peaks + Shading ─────────────
    ax2 = axes[1]

    # Shade segments
    for seg in segments:
        color = PHASE_COLORS.get(seg["phase"], "#555555")
        ax2.axvspan(seg["start"], seg["end"], alpha=0.25, color=color)

    # Envelope line
    ax2.plot(times, envelope, color="white", linewidth=1.2, label="Envelope")

    # Inhale peaks
    peak_t = peaks.get("peak_times", np.array([]))
    peak_i = peaks.get("peak_indices", np.array([]))
    if len(peak_t) > 0 and len(peak_i) > 0:
        valid = peak_i[peak_i < len(envelope)]
        ax2.scatter(
            times[valid], envelope[valid],
            color="#4FC3F7", s=50, zorder=5, label="Peak (inhale)"
        )

    # Valley markers
    valley_t = peaks.get("valley_times", np.array([]))
    valley_i = peaks.get("valley_indices", np.array([]))
    if len(valley_t) > 0 and len(valley_i) > 0:
        valid = valley_i[valley_i < len(envelope)]
        ax2.scatter(
            times[valid], envelope[valid],
            color="#81C784", s=30, marker="v", zorder=5, label="Valley (exhale)"
        )

    ax2.set_ylabel("RMS Envelope", color="#AAAAAA", fontsize=9)
    ax2.set_xlim(0, duration)
    ax2.legend(facecolor="#1C2128", edgecolor="#333333",
               labelcolor="white", fontsize=8, loc="upper right")

    # ── Panel 3: Segment Timeline ────────────────────────
    ax3 = axes[2]
    for seg in segments:
        color = PHASE_COLORS.get(seg["phase"], "#555555")
        ax3.barh(0, seg["end"] - seg["start"],
                 left=seg["start"], height=0.6,
                 color=color, edgecolor="#0D1117", linewidth=0.3)

    ax3.set_xlim(0, duration)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (seconds)", color="#AAAAAA", fontsize=9)

    # Legend
    patches = [
        mpatches.Patch(color=PHASE_COLORS["INHALE"], label="Inhale"),
        mpatches.Patch(color=PHASE_COLORS["EXHALE"], label="Exhale"),
        mpatches.Patch(color=PHASE_COLORS["HOLD"],   label="Hold"),
    ]
    ax3.legend(handles=patches, facecolor="#1C2128", edgecolor="#333333",
               labelcolor="white", fontsize=8, loc="upper right",
               ncol=3)

    plt.tight_layout(pad=1.5)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"📊 Plot saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)
