"""
trending.py — Calculate and visualize breathing trends over multiple sessions
"""

import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime


def calculate_trend_stats(history_list: List[Dict]) -> Dict:
    """Calculate trend statistics from session history.

    Args:
        history_list: List of session dicts with scores and features

    Returns:
        {
            "overall_trend": "improving" | "stable" | "declining",
            "best_score": int,
            "worst_score": int,
            "avg_score": float,
            "improvement_pct": float,  # % change from first to last
            "session_count": int,
            "bpm_trend": [values],
            "stability_trend": [values],
            "ie_ratio_trend": [values],
            "timestamps": [dates]
        }
    """
    if not history_list:
        return {
            "overall_trend": "stable",
            "best_score": 0,
            "worst_score": 0,
            "avg_score": 0,
            "improvement_pct": 0,
            "session_count": 0,
            "bpm_trend": [],
            "stability_trend": [],
            "ie_ratio_trend": [],
            "timestamps": []
        }

    scores = [s.get("score", 0) for s in history_list]
    bpms = [s.get("features", {}).get("bpm", 0) for s in history_list]
    stabilities = [s.get("features", {}).get("stability_cv", 0) for s in history_list]
    ie_ratios = [s.get("features", {}).get("inhale_exhale_ratio", 0) for s in history_list]
    timestamps = [s.get("timestamp", "")[:10] for s in history_list]

    avg_score = np.mean(scores) if scores else 0
    best_score = max(scores) if scores else 0
    worst_score = min(scores) if scores else 0

    # Calculate improvement from first to last session
    improvement_pct = 0
    if len(scores) > 1 and scores[0] != 0:
        improvement_pct = ((scores[-1] - scores[0]) / scores[0]) * 100

    # Determine trend
    if len(scores) > 1:
        recent_avg = np.mean(scores[-3:]) if len(scores) >= 3 else np.mean(scores[-2:])
        old_avg = np.mean(scores[:3]) if len(scores) >= 3 else scores[0]
        if recent_avg > old_avg * 1.05:
            trend = "improving"
        elif recent_avg < old_avg * 0.95:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"

    return {
        "overall_trend": trend,
        "best_score": int(best_score),
        "worst_score": int(worst_score),
        "avg_score": round(avg_score, 1),
        "improvement_pct": round(improvement_pct, 1),
        "session_count": len(history_list),
        "bpm_trend": bpms,
        "stability_trend": stabilities,
        "ie_ratio_trend": ie_ratios,
        "timestamps": timestamps,
    }


def get_trend_summary(history_list: List[Dict]) -> str:
    """Generate a text summary of trends.

    Args:
        history_list: List of sessions

    Returns:
        Formatted trend summary string
    """
    stats = calculate_trend_stats(history_list)

    if stats["session_count"] == 0:
        return "No sessions yet. Record a breathing session to start tracking!"

    trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}
    emoji = trend_emoji.get(stats["overall_trend"], "❓")

    text = f"""
╔════════════════════════════════════════════╗
║          BREATHING TREND SUMMARY           ║
╚════════════════════════════════════════════╝

Overall Trend: {emoji} {stats["overall_trend"].upper()}
Sessions Tracked: {stats["session_count"]}

Score Statistics:
  • Average Score:    {stats['avg_score']}/100
  • Best Score:       {stats['best_score']}/100
  • Worst Score:      {stats['worst_score']}/100
  • Improvement:      {stats['improvement_pct']:+.1f}%

📊 Key Metrics Trend:
  • BPM (Breathing Rate):      {stats['bpm_trend'][-1]:.1f} breaths/min
  • Stability (CV):            {stats['stability_trend'][-1]:.3f} (lower is better)
  • IE Ratio:                  {stats['ie_ratio_trend'][-1]:.2f}x

💡 Interpretation:
"""

    if stats["overall_trend"] == "improving":
        text += "  ✨ Great progress! Your breathing is becoming more stable.\n"
    elif stats["overall_trend"] == "declining":
        text += "  ⚠️  Your score is declining. Focus on steady breathing practice.\n"
    else:
        text += "  ✓ Your breathing is stable. Keep practicing regularly.\n"

    text += "\n🎯 Next Steps:\n"
    if stats["stability_trend"][-1] > 0.35:
        text += "  • Work on breathing consistency\n"
    if stats["bpm_trend"][-1] > 20:
        text += "  • Try to slow down your breathing\n"
    if stats["ie_ratio_trend"][-1] < 1.0:
        text += "  • Focus on longer exhales\n"

    return text


def plot_trending_chart(history_list: List[Dict], save_path: Optional[str] = None):
    """Generate matplotlib figure with trending charts.

    Args:
        history_list: List of sessions
        save_path: Optional path to save figure (PNG/JPG)

    Returns:
        matplotlib.figure.Figure object
    """
    stats = calculate_trend_stats(history_list)

    if stats["session_count"] == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No sessions yet.\nRecord a breathing session to see trends.",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        return fig

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    fig.patch.set_facecolor('#0D1117')

    session_nums = list(range(1, len(stats["timestamps"]) + 1))
    x_ticks = session_nums[::max(1, len(session_nums)//5)]

    # Plot 1: Score Trend
    ax1.set_facecolor('#161B22')
    ax1.plot(session_nums, stats["bpm_trend"], 'o-', color='#56D364', linewidth=2, markersize=6, label='Score')
    ax1.set_ylabel('Breathing Score', color='#E6EDF3', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.2, color='#30363D')
    ax1.set_title('Breathing Trend Analysis', color='#E6EDF3', fontsize=12, fontweight='bold', pad=15)
    ax1.tick_params(colors='#E6EDF3')
    ax1.set_xticks(x_ticks)

    # Plot 2: BPM Trend
    ax2.set_facecolor('#161B22')
    ax2.plot(session_nums, stats["bpm_trend"], 'o-', color='#4FC3F7', linewidth=2, markersize=6, label='BPM')
    if len(stats["bpm_trend"]) > 1:
        bpm_avg = np.mean(stats["bpm_trend"])
        ax2.axhline(y=bpm_avg, color='#F0883E', linestyle='--', alpha=0.7, label=f'Average: {bpm_avg:.1f}')
    ax2.set_ylabel('BPM (breaths/min)', color='#E6EDF3', fontsize=10, fontweight='bold')
    ax2.set_ylim(max(0, min(stats["bpm_trend"]) - 5), max(stats["bpm_trend"]) + 5)
    ax2.grid(True, alpha=0.2, color='#30363D')
    ax2.legend(loc='upper left', facecolor='#30363D', edgecolor='#30363D', labelcolor='#E6EDF3')
    ax2.tick_params(colors='#E6EDF3')
    ax2.set_xticks(x_ticks)

    # Plot 3: Stability Trend
    ax3.set_facecolor('#161B22')
    ax3.plot(session_nums, stats["stability_trend"], 'o-', color='#F85149', linewidth=2, markersize=6, label='Stability CV')
    if len(stats["stability_trend"]) > 1:
        stability_avg = np.mean(stats["stability_trend"])
        ax3.axhline(y=stability_avg, color='#F0883E', linestyle='--', alpha=0.7, label=f'Average: {stability_avg:.3f}')
    ax3.set_ylabel('Stability CV (lower=better)', color='#E6EDF3', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Session Number', color='#E6EDF3', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.2, color='#30363D')
    ax3.legend(loc='upper left', facecolor='#30363D', edgecolor='#30363D', labelcolor='#E6EDF3')
    ax3.tick_params(colors='#E6EDF3')
    ax3.set_xticks(x_ticks)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, facecolor='#0D1117', edgecolor='none', dpi=100)

    return fig
