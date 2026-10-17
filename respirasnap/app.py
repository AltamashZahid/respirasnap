"""
app.py — RespriaSnap: Yoga Breathing + Posture Analyzer
Run:  python app.py

Layout:
  Left  — Posture: live webcam, skeleton overlay, pose label, live match % vs reference
  Right — Breathing: record/upload, score, metrics, plots, AI + comparison tabs

Typical flow:
  1. "Load Reference" → pick an instructor video (mp4/mov/...). Its breathing
     (audio) and postures (joint angles) are extracted and saved.
  2. "Record" → follow along. Mic + webcam are captured together.
  3. The Comparison tab scores posture, breathing and breath timing vs the reference.
"""

import os
import queue
import sys
import threading
import time
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
for _stream in (sys.stdout, sys.stderr):  # emoji logs must not crash on cp1252 consoles
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from audio      import record_mic, load_file, NoAudioTrack
from preprocess import preprocess
from envelope   import extract_envelope
from detect     import detect_peaks_valleys
from segment    import segment_breathing
from features   import extract_features
from score      import compute_score
from agents     import clinical_summary_agent, coaching_agent, followup_agent
from history    import load_history, save_session, get_session_list
from compare    import (overall_comparison, format_comparison_for_display,
                        compare_two_sessions, format_session_comparison, live_pose_match)
from references import (save_reference, load_reference, list_references, delete_reference,
                        reconstruct_result, reference_from_json)
from export     import export_session_csv, export_session_json, get_default_export_filename
from trending   import get_trend_summary, plot_trending_chart
import pose as posemod

# ── Theme ─────────────────────────────────────────────────
BG, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
ACCENT, GREEN, ORANGE, RED = "#4FC3F7", "#56D364", "#F0883E", "#F85149"
TEXT, MUTED = "#E6EDF3", "#8B949E"
PHASE_CLR = {"INHALE": "#4FC3F7", "EXHALE": "#56D364", "HOLD": "#F0883E"}

REC_SECS = 15
MAX_REC_SECS = 180
COUNTDOWN_SECS = 3
CAM_W, CAM_H = 430, 320
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ══════════════════════════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════════════════════════

def _lbl(parent, text, fg=MUTED, font_size=8, bold=False, **kw):
    style = "bold" if bold else "normal"
    return tk.Label(parent, text=text, bg=parent["bg"],
                    fg=fg, font=("Helvetica", font_size, style), **kw)


def _btn(parent, text, cmd, fg=ACCENT, state=tk.NORMAL):
    return tk.Button(parent, text=text, command=cmd,
                     bg=BORDER, fg=fg, activebackground="#21262d",
                     activeforeground=fg, relief=tk.FLAT,
                     font=("Helvetica", 10), cursor="hand2",
                     pady=5, padx=8, state=state)


def _section_frame(parent, title):
    """Labelled section. Returns (outer_frame, inner_panel)."""
    outer = tk.Frame(parent, bg=BG)
    _lbl(outer, title.upper(), MUTED, 7, bold=True).pack(anchor=tk.W, pady=(0, 2))
    inner = tk.Frame(outer, bg=PANEL, padx=8, pady=7)
    inner.pack(fill=tk.BOTH, expand=True)
    return outer, inner


def _txt_box(parent, height=4):
    t = scrolledtext.ScrolledText(parent, bg=PANEL, fg=TEXT,
                                  font=("Consolas", 9), wrap=tk.WORD,
                                  relief=tk.FLAT, height=height,
                                  state=tk.DISABLED)
    t.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
    return t


def _write(widget, text):
    widget.config(state=tk.NORMAL)
    widget.delete("1.0", tk.END)
    widget.insert(tk.END, text)
    widget.config(state=tk.DISABLED)


# ══════════════════════════════════════════════════════════
# ANALYSIS HELPERS (run in worker threads, no Tk calls)
# ══════════════════════════════════════════════════════════

def analyze_breathing(audio, sr):
    """Audio → (features, segments, envelope, times, peaks)."""
    clean = preprocess(audio, sr)
    times, env = extract_envelope(clean, sr)
    peaks = detect_peaks_valleys(env, times)
    segs = segment_breathing(env, times, peaks)
    features = extract_features(segs, peaks, len(audio) / sr)
    return features, segs, env, times, peaks


def analyze_media_file(path, want_poses=True, max_seconds=None):
    """Audio and/or video file → result dict usable as session or reference.

    Uses its own pose detector, since MediaPipe detectors are not thread-safe.
    """
    ext = os.path.splitext(path)[1].lower()
    result = {"path": os.path.basename(path), "pose_data": None,
              "features": {}, "segments": [], "envelope": np.array([]),
              "times": np.array([]),
              "peaks": {"peak_indices": np.array([], dtype=int),
                        "valley_indices": np.array([], dtype=int)},
              "audio": None, "sr": 16000, "notes": []}

    try:
        audio, sr = load_file(path)
        if max_seconds:
            audio = audio[:int(max_seconds * sr)]
        features, segs, env, times, peaks = analyze_breathing(audio, sr)
        result.update(audio=audio, sr=sr, features=features, segments=segs,
                      envelope=env, times=times, peaks=peaks)
    except NoAudioTrack:
        result["notes"].append("no audio track — breathing comparison unavailable")

    if want_poses and ext in VIDEO_EXTS:
        detector = posemod.build_detector()
        if detector is None:
            result["notes"].append("pose model unavailable — posture comparison skipped")
        else:
            try:
                result["pose_data"] = posemod.extract_poses_from_video(
                    path, detector, max_seconds=max_seconds)
            finally:
                detector.close()

    if result["audio"] is None and not result["pose_data"]:
        raise RuntimeError(f"Nothing to analyse in {result['path']} "
                           "(no audio track and no detectable person).")
    if result["audio"] is None:
        dur = result["pose_data"][-1]["timestamp"] if result["pose_data"] else 0.0
        result["audio"] = np.zeros(max(1, int(dur * 16000)), dtype=np.float32)
        result["features"] = {}
    return result


# ══════════════════════════════════════════════════════════
# POSTURE PANEL — sole owner of the webcam and the live detector
# ══════════════════════════════════════════════════════════

class PosturePanel(tk.Frame):
    def __init__(self, parent, detector):
        super().__init__(parent, bg=BG)
        self._det = detector
        self._cap = None
        self._running = False
        self._capturing = False
        self._cap_t0 = 0.0
        self._frames = []
        self._ref_frames = None
        self._last_label = None
        self._build()

    @property
    def has_detector(self):
        return self._det is not None

    def _build(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill=tk.X, pady=(0, 6))
        _lbl(hdr, "🧘  Yoga Posture", ACCENT, 11, bold=True).pack(side=tk.LEFT)

        border = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        border.pack()
        self._cam = tk.Label(border, bg="#111111", width=CAM_W, height=CAM_H)
        self._cam.pack()
        self._blank = ImageTk.PhotoImage(Image.new("RGB", (CAM_W, CAM_H), "#111111"))
        self._cam.configure(image=self._blank)

        self._pose_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._pose_var, bg=BG, fg=GREEN,
                 font=("Helvetica", 13, "bold")).pack(pady=(5, 0))
        self._match_var = tk.StringVar(value="")
        self._match_lbl = tk.Label(self, textvariable=self._match_var, bg=BG, fg=MUTED,
                                   font=("Helvetica", 10))
        self._match_lbl.pack(pady=(0, 2))

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=3)
        self._start_btn = _btn(btn_row, "▶  Start Camera", self.start, GREEN)
        self._start_btn.pack(side=tk.LEFT, padx=3)
        self._stop_btn = _btn(btn_row, "■  Stop", self.stop, RED, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=3)

        if self._det is None:
            msg = posemod.MP_ERR or "pose model missing"
            _lbl(self, f"Pose model unavailable ({msg[:60]}) — camera only", ORANGE, 8).pack()

        log_outer, log_inner = _section_frame(self, "Pose Log")
        log_outer.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        top_row = tk.Frame(log_inner, bg=PANEL)
        top_row.pack(fill=tk.X, pady=(0, 3))
        _lbl(top_row, "pose changes", MUTED, 7).pack(side=tk.LEFT)
        _btn(top_row, "Clear", self._clear_log, MUTED).pack(side=tk.RIGHT)
        self._log = scrolledtext.ScrolledText(
            log_inner, bg="#0d1117", fg=TEXT, font=("Consolas", 8),
            relief=tk.FLAT, height=6, state=tk.DISABLED)
        self._log.pack(fill=tk.BOTH, expand=True)

    # ── Camera ───────────────────────────────────────────

    def start(self) -> bool:
        if self._running:
            return True
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap = cap
        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._pose_var.set("Detecting…")
        self._poll()
        return True

    def stop(self):
        if self._capturing:
            return  # don't pull the camera out from under a recording
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._cam.configure(image=self._blank)
        self._pose_var.set("—")
        self._match_var.set("")
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)

    # ── Capture API used by the breathing panel ──────────

    def begin_capture(self, ref_frames=None) -> bool:
        """Start collecting pose frames (starts the camera if needed)."""
        if not self.start():
            return False
        self._frames = []
        self._ref_frames = ref_frames if ref_frames else None
        self._cap_t0 = time.time()
        self._capturing = True
        self._stop_btn.config(state=tk.DISABLED)
        return True

    def end_capture(self):
        """Stop collecting and return frames [{"timestamp","pose","angles"}]."""
        self._capturing = False
        self._ref_frames = None
        self._match_var.set("")
        if self._running:
            self._stop_btn.config(state=tk.NORMAL)
        frames, self._frames = self._frames, []
        return frames

    # ── Frame loop (main thread) ─────────────────────────

    def _poll(self):
        if not self._running:
            return
        ok, frame = self._cap.read()
        if ok:
            frame = posemod.fit_frame(cv2.flip(frame, 1), CAM_W, CAM_H)
            res = posemod.analyze_frame(self._det, frame)
            label = res["pose"]
            match = target = None

            if self._capturing:
                t = time.time() - self._cap_t0
                self._frames.append({"timestamp": round(t, 3), "pose": label,
                                     "angles": posemod.clean_angles(res["angles"])})
                if self._ref_frames:
                    match, target = live_pose_match(res["angles"], self._ref_frames, t)
                    if match is None:
                        self._match_var.set("Match: —  (get your whole body in frame)")
                        self._match_lbl.config(fg=MUTED)
                    else:
                        color = GREEN if match >= 75 else ORANGE if match >= 50 else RED
                        tgt = f"  ·  target: {target}" if target else ""
                        self._match_var.set(f"Match vs reference: {match:.0f}%{tgt}")
                        self._match_lbl.config(fg=color)
                else:
                    self._match_var.set(f"● Recording {t:4.1f}s")
                    self._match_lbl.config(fg=RED)

            self._pose_var.set(label)
            if label != self._last_label and label not in posemod.NOT_A_POSE:
                self._log_entry(label)
            self._last_label = label

            annotated = posemod.draw_pose(frame, res["landmarks"], label, match, target)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
            self._cam.configure(image=photo)
            self._cam.image = photo

        self.after(15, self._poll)

    def _log_entry(self, label):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.config(state=tk.NORMAL)
        self._log.insert("1.0", f"[{ts}]  {label}\n")
        if int(self._log.index("end-1c").split(".")[0]) > 50:
            self._log.delete("51.0", tk.END)
        self._log.config(state=tk.DISABLED)

    def _clear_log(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

    def cleanup(self):
        self._capturing = False
        self._running = False
        if self._cap:
            self._cap.release()


# ══════════════════════════════════════════════════════════
# BREATHING PANEL
# ══════════════════════════════════════════════════════════

class BreathingPanel(tk.Frame):
    def __init__(self, parent, posture: PosturePanel):
        super().__init__(parent, bg=BG)
        self._posture = posture
        self._audio = None
        self._sr = 16000
        self._pose_data = None
        self._busy_flag = False
        self._reference_result = None
        self._current_result = None
        self._ui_queue = queue.Queue()
        self._build()
        self._pump_ui()

    # ── Thread → UI bridge (Tk is not thread-safe) ───────

    def _ui(self, fn, *args):
        self._ui_queue.put((fn, args))

    def _pump_ui(self):
        try:
            while True:
                fn, args = self._ui_queue.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(50, self._pump_ui)

    # ── Layout ───────────────────────────────────────────

    def _build(self):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill=tk.X, pady=(0, 6))
        _lbl(hdr, "🫁  Breathing Analysis", ACCENT, 11, bold=True).pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Ready")
        self._status_lbl = tk.Label(hdr, textvariable=self._status_var,
                                    bg=BG, fg=GREEN, font=("Helvetica", 9))
        self._status_lbl.pack(side=tk.RIGHT)

        ctrl_outer, ctrl = _section_frame(self, "Session")
        ctrl_outer.pack(fill=tk.X, pady=(0, 6))

        btn_row = tk.Frame(ctrl, bg=PANEL)
        btn_row.pack(fill=tk.X)
        self._rec_btn = _btn(btn_row, "🎙  Record", self._start_record, ACCENT)
        self._rec_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._dur_var = tk.IntVar(value=REC_SECS)
        tk.Spinbox(btn_row, from_=5, to=MAX_REC_SECS, increment=5, width=4,
                   textvariable=self._dur_var, bg=BORDER, fg=TEXT,
                   buttonbackground=BORDER, relief=tk.FLAT,
                   font=("Helvetica", 10)).pack(side=tk.LEFT)
        _lbl(btn_row, "s", MUTED, 9).pack(side=tk.LEFT, padx=(2, 8))
        self._upl_btn = _btn(btn_row, "📂  Upload", self._upload_file, MUTED)
        self._upl_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._exp_btn = _btn(btn_row, "💾  Export", self._export_session, GREEN)
        self._exp_btn.pack(side=tk.LEFT)

        ref_row = tk.Frame(ctrl, bg=PANEL)
        ref_row.pack(fill=tk.X, pady=(6, 0))
        _lbl(ref_row, "Reference:", MUTED, 8).pack(side=tk.LEFT, padx=(0, 4))
        self._ref_var = tk.StringVar(value="None")
        self._ref_dropdown = ttk.Combobox(ref_row, textvariable=self._ref_var,
                                          state="readonly", width=34)
        self._ref_dropdown.pack(side=tk.LEFT, padx=(0, 6))
        self._ref_dropdown.bind("<<ComboboxSelected>>", lambda e: self._load_saved_reference())
        self._ref_btn = _btn(ref_row, "🧘 Load Video", self._upload_reference, ORANGE)
        self._ref_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._save_ref_btn = _btn(ref_row, "⭐ Save Session as Ref", self._save_current_reference, GREEN)
        self._save_ref_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._del_ref_btn = _btn(ref_row, "🗑", self._delete_saved_reference, RED)
        self._del_ref_btn.pack(side=tk.LEFT)

        comp_row = tk.Frame(ctrl, bg=PANEL)
        comp_row.pack(fill=tk.X, pady=(6, 0))
        _lbl(comp_row, "History:", MUTED, 8).pack(side=tk.LEFT, padx=(0, 4))
        self._sess1_var = tk.StringVar(value="None")
        self._sess1_dropdown = ttk.Combobox(comp_row, textvariable=self._sess1_var,
                                            state="readonly", width=28)
        self._sess1_dropdown.pack(side=tk.LEFT, padx=(0, 3))
        _lbl(comp_row, "vs", MUTED, 8).pack(side=tk.LEFT, padx=(0, 3))
        self._sess2_var = tk.StringVar(value="None")
        self._sess2_dropdown = ttk.Combobox(comp_row, textvariable=self._sess2_var,
                                            state="readonly", width=28)
        self._sess2_dropdown.pack(side=tk.LEFT, padx=(0, 6))
        self._cmp_btn = _btn(comp_row, "Compare", self._compare_sessions, ORANGE)
        self._cmp_btn.pack(side=tk.LEFT)

        self._progress = ttk.Progressbar(ctrl, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=(6, 2))
        self._file_lbl = _lbl(ctrl, "Record with mic + camera, or upload wav · mp3 · mp4 · mov · m4a",
                              MUTED, 7)
        self._file_lbl.pack(anchor=tk.W)

        # Score + metrics
        sm_row = tk.Frame(self, bg=BG)
        sm_row.pack(fill=tk.X, pady=(0, 6))
        sc_outer, sc = _section_frame(sm_row, "Score")
        sc_outer.pack(side=tk.LEFT, padx=(0, 6))
        self._score_var = tk.StringVar(value="—")
        tk.Label(sc, textvariable=self._score_var, bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 32, "bold"), width=4).pack()
        _lbl(sc, "/ 100", MUTED, 8).pack()

        met_outer, met = _section_frame(sm_row, "Metrics")
        met_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self._mvars = {}
        fields = [
            ("BPM", "bpm", ""), ("IE Ratio", "inhale_exhale_ratio", "x"),
            ("Stability CV", "stability_cv", ""), ("Interrupts", "interruptions", ""),
            ("Avg Inhale", "avg_inhale_s", "s"), ("Avg Exhale", "avg_exhale_s", "s"),
            ("Holds", "holds_detected", ""), ("Cycles", "num_cycles", ""),
        ]
        for i, (label, key, unit) in enumerate(fields):
            row, col = divmod(i, 4)
            _lbl(met, label, MUTED, 8).grid(row=row * 2, column=col, sticky=tk.W, padx=(0, 14))
            var = tk.StringVar(value="—")
            self._mvars[key] = (var, unit)
            tk.Label(met, textvariable=var, bg=PANEL, fg=TEXT,
                     font=("Helvetica", 9, "bold")).grid(row=row * 2 + 1, column=col,
                                                         sticky=tk.W, padx=(0, 14))

        al_outer, al = _section_frame(sm_row, "Match vs Ref")
        al_outer.pack(side=tk.LEFT)
        self._align_var = tk.StringVar(value="—")
        tk.Label(al, textvariable=self._align_var, bg=PANEL, fg=ORANGE,
                 font=("Helvetica", 32, "bold"), width=4).pack()
        _lbl(al, "/ 100", MUTED, 8).pack()

        # Tabs fill the remaining height; plots live in their own tab
        ai_outer, ai = _section_frame(self, "Insights")
        ai_outer.pack(fill=tk.BOTH, expand=True)
        _style_notebook()
        self._nb = ttk.Notebook(ai)
        self._nb.pack(fill=tk.BOTH, expand=True)
        self._compare_txt = self._add_tab("⚖️ Comparison")

        plot_tab = tk.Frame(self._nb, bg=BG)
        self._nb.add(plot_tab, text="📉 Signals")
        self._fig = Figure(figsize=(6, 3.0), facecolor=BG)
        self._axes = [self._fig.add_subplot(4, 1, i + 1, facecolor=PANEL) for i in range(4)]
        self._style_axes()
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_tab)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._clinical_txt = self._add_tab("🩺 Clinical")
        self._coaching_txt = self._add_tab("💡 Coaching")
        self._followup_txt = self._add_tab("📈 Follow-up")
        self._trends_txt = self._add_tab("📊 Trends", ("Open trend chart", self._open_trend_chart))
        _write(self._compare_txt,
               "1. Load a reference (instructor video) with “🧘 Load Video”, or pick a saved one.\n"
               "2. Press Record and follow along — mic + camera are captured together.\n"
               "3. Your posture, breathing and breath timing are scored against the reference here.")

        self._refresh_reference_dropdown()
        self._refresh_session_dropdowns()
        self._update_trends_display()

    def _add_tab(self, title, button=None):
        f = tk.Frame(self._nb, bg=PANEL)
        self._nb.add(f, text=title)
        if button:
            _btn(f, button[0], button[1], ACCENT).pack(anchor=tk.E, padx=3, pady=(3, 0))
        return _txt_box(f, height=8)

    # ── Recording ────────────────────────────────────────

    def _record_secs(self):
        try:
            return max(5, min(MAX_REC_SECS, int(self._dur_var.get())))
        except (tk.TclError, ValueError):
            return REC_SECS

    def _start_record(self):
        if self._busy_flag:
            return
        self._lock(True)
        self._countdown(COUNTDOWN_SECS)

    def _countdown(self, n):
        if n > 0:
            self._setstatus(f"Get ready… {n}", ORANGE)
            self.after(1000, self._countdown, n - 1)
            return
        secs = self._record_secs()
        ref_frames = (self._reference_result or {}).get("pose_data")
        cam = self._posture.has_detector and self._posture.begin_capture(ref_frames)
        self._progress.start(10)
        self._setstatus(f"Recording {secs}s…", ACCENT)
        self._file_lbl.config(text=f"Live mic{' + camera' if cam else ' (no posture: camera/model unavailable)'}")
        threading.Thread(target=self._rec_thread, args=(secs,), daemon=True).start()

    def _rec_thread(self, secs):
        try:
            audio, sr = record_mic(secs)
            self._ui(self._on_recorded, audio, sr)
        except Exception as e:
            self._ui(self._posture.end_capture)
            self._ui(self._on_error, f"Recording failed: {e}")

    def _on_recorded(self, audio, sr):
        frames = self._posture.end_capture()
        self._audio, self._sr = audio, sr
        self._pose_data = frames or None
        self._start_analysis()

    # ── Upload (audio or video) ──────────────────────────

    def _upload_file(self):
        if self._busy_flag:
            return
        path = filedialog.askopenfilename(
            title="Select your session (audio or video)",
            filetypes=[("Audio/Video", "*.wav *.mp3 *.mp4 *.mov *.m4a *.ogg *.flac *.aac *.webm *.mkv *.avi"),
                       ("All files", "*.*")])
        if not path:
            return
        self._lock(True)
        self._progress.start(10)
        self._setstatus("Loading…", ACCENT)
        self._file_lbl.config(text=os.path.basename(path))
        threading.Thread(target=self._load_thread, args=(path,), daemon=True).start()

    def _load_thread(self, path):
        try:
            r = analyze_media_file(path)
            self._ui(self._on_uploaded, r)
        except Exception as e:
            self._ui(self._on_error, str(e))

    def _on_uploaded(self, r):
        self._audio, self._sr = r["audio"], r["sr"]
        self._pose_data = r["pose_data"]
        if r["notes"]:
            self._file_lbl.config(text=f"{r['path']} — {'; '.join(r['notes'])}")
        self._start_analysis()

    # ── Analysis ─────────────────────────────────────────

    def _start_analysis(self):
        self._setstatus("Analysing…", ORANGE)
        threading.Thread(target=self._analysis_thread,
                         args=(self._audio, self._sr, self._pose_data, self._reference_result),
                         daemon=True).start()

    def _analysis_thread(self, audio, sr, pose_data, reference):
        try:
            features, segs, env, times, peaks = analyze_breathing(audio, sr)
            score_res = compute_score(features)
            score = score_res["score"]

            history = load_history()
            clinical = clinical_summary_agent(features)
            coaching = coaching_agent(features)
            followup = followup_agent({"score": score, "features": features}, history)
            save_session(features, score_res, clinical, coaching, followup)

            r = {"features": features, "score": score,
                 "clinical": clinical, "coaching": coaching, "followup": followup,
                 "segments": segs, "envelope": env, "times": times, "peaks": peaks,
                 "audio": audio, "sr": sr, "pose_data": pose_data,
                 "timestamp": datetime.now().isoformat(timespec="seconds"),
                 "comparison": overall_comparison(
                     {"features": features, "segments": segs, "pose_data": pose_data},
                     reference) if reference else None}
            self._ui(self._on_done, r)
        except Exception:
            import traceback
            self._ui(self._on_error, traceback.format_exc())

    def _on_done(self, r):
        self._current_result = r
        self._progress.stop()
        self._lock(False)
        self._setstatus("Done ✓", GREEN)

        self._score_var.set(str(r["score"]))
        for key, (var, unit) in self._mvars.items():
            v = r["features"].get(key)
            var.set("—" if v is None else f"{v}{unit}")

        _write(self._clinical_txt, r["clinical"])
        _write(self._coaching_txt, r["coaching"])
        _write(self._followup_txt, r["followup"])

        cmp = r["comparison"]
        if cmp:
            self._align_var.set(f"{cmp['overall_alignment_score']:.0f}")
            _write(self._compare_txt, format_comparison_for_display(cmp))
            self._nb.select(0)
        else:
            self._align_var.set("—")
            n = len(r["pose_data"] or [])
            _write(self._compare_txt,
                   "No reference loaded — nothing to compare against.\n\n"
                   f"Captured {n} pose frames this session.\n"
                   "Load an instructor video (🧘 Load Video) or pick a saved reference, then record again.\n"
                   "You can also save this session as a reference with ⭐.")
            self._nb.select(1)

        self._refresh_session_dropdowns()
        self._update_trends_display()
        self._draw(r)

    def _on_error(self, msg):
        self._progress.stop()
        self._lock(False)
        self._setstatus("Error", RED)
        messagebox.showerror("Error", str(msg)[:800])

    # ── References ───────────────────────────────────────

    def _upload_reference(self):
        if self._busy_flag:
            return
        path = filedialog.askopenfilename(
            title="Select reference (instructor video, audio, or saved JSON)",
            filetypes=[("All supported", "*.mp4 *.mov *.avi *.mkv *.webm *.wav *.mp3 *.m4a *.ogg *.flac *.aac *.json"),
                       ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                       ("JSON reference", "*.json"),
                       ("All files", "*.*")])
        if not path:
            return

        if path.lower().endswith(".json"):
            try:
                import json
                with open(path, encoding="utf-8") as f:
                    ref = reference_from_json(json.load(f), os.path.basename(path))
                if not ref:
                    messagebox.showerror("Error", "JSON file doesn't contain reference data.")
                    return
                self._set_reference(ref)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON: {e}")
            return

        self._lock(True)
        self._progress.start(10)
        self._setstatus("Processing reference (extracting breathing + poses)…", ORANGE)
        threading.Thread(target=self._ref_thread, args=(path,), daemon=True).start()

    def _ref_thread(self, path):
        try:
            r = analyze_media_file(path, max_seconds=MAX_REC_SECS)
            name = os.path.splitext(os.path.basename(path))[0]
            saved = save_reference(name, r, path)
            self._ui(self._on_reference_processed, r, saved)
        except Exception as e:
            self._ui(self._on_error, f"Reference failed: {e}")

    def _on_reference_processed(self, r, saved):
        self._progress.stop()
        self._lock(False)
        self._refresh_reference_dropdown(select_id=saved.get("ref_id"))
        self._set_reference(r)
        if not saved.get("success"):
            messagebox.showwarning("Not saved", f"Reference is loaded but could not be saved:\n{saved.get('message')}")
        if r["notes"]:
            messagebox.showinfo("Reference loaded", "Note: " + "; ".join(r["notes"]))

    def _set_reference(self, ref):
        self._reference_result = ref
        frames = ref.get("pose_data") or []
        usable = sum(1 for f in frames if f.get("angles") and
                     sum(v is not None for v in f["angles"]) >= 4)
        dur = ref.get("duration") or (len(ref["audio"]) / ref.get("sr", 16000)
                                      if ref.get("audio") is not None else 0)
        if dur:
            self._dur_var.set(int(max(5, min(MAX_REC_SECS, round(dur)))))
        self._setstatus(f"Reference: {ref.get('path', 'reference')}", GREEN)
        f = ref.get("features") or {}
        _write(self._compare_txt,
               f"✓ Reference loaded: {ref.get('path')}\n\n"
               f"  Duration:        {dur:.1f}s  (record length set to match)\n"
               f"  Breathing:       {'BPM ' + str(f.get('bpm')) if f else 'n/a (no audio)'}\n"
               f"  Posture frames:  {usable} usable of {len(frames)}\n\n"
               + ("Press Record and follow along — the camera panel shows a live match %."
                  if usable else
                  "No person detected in the reference, so only breathing will be compared."))
        self._nb.select(0)

    def _load_saved_reference(self):
        selected = self._ref_var.get()
        if selected == "None":
            self._reference_result = None
            self._setstatus("Reference cleared", MUTED)
            return
        for ref in list_references():
            if ref["label"] == selected:
                data = load_reference(ref["ref_id"])
                if data:
                    self._set_reference(reconstruct_result(data))
                return

    def _save_current_reference(self):
        """Save the last analysed session (e.g. the teacher's own recording)."""
        r = self._current_result
        if not r:
            messagebox.showwarning("No Session", "Record or upload a session first.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Save Session as Reference")
        dialog.geometry("400x140")
        dialog.configure(bg=BG)
        dialog.transient(self.winfo_toplevel())
        tk.Label(dialog, text="Reference name:", bg=BG, fg=TEXT).pack(pady=10)
        entry = tk.Entry(dialog, width=40, bg=PANEL, fg=TEXT, insertbackground=TEXT)
        entry.pack(pady=5)
        entry.insert(0, f"My session {datetime.now():%Y-%m-%d %H:%M}")
        entry.focus()

        def save():
            name = entry.get().strip()
            if not name:
                return
            res = save_reference(name, r, "recorded session")
            if res["success"]:
                self._refresh_reference_dropdown(select_id=res["ref_id"])
                dialog.destroy()
                self._setstatus(f"Saved reference: {name}", GREEN)
            else:
                messagebox.showerror("Error", res["message"])

        entry.bind("<Return>", lambda e: save())
        tk.Button(dialog, text="Save", command=save, bg=GREEN, fg=BG,
                  font=("Helvetica", 10, "bold")).pack(pady=8)

    def _delete_saved_reference(self):
        selected = self._ref_var.get()
        if selected == "None":
            messagebox.showwarning("No Selection", "Select a saved reference to delete")
            return
        if not messagebox.askyesno("Confirm Delete", f"Delete reference?\n{selected}"):
            return
        for ref in list_references():
            if ref["label"] == selected:
                res = delete_reference(ref["ref_id"])
                if not res["success"]:
                    messagebox.showerror("Error", res["message"])
                self._reference_result = None
                self._refresh_reference_dropdown()
                return

    def _refresh_reference_dropdown(self, select_id=None):
        refs = list_references()
        self._ref_dropdown["values"] = ["None"] + [r["label"] for r in refs]
        chosen = next((r["label"] for r in refs if r["ref_id"] == select_id), "None")
        self._ref_var.set(chosen)

    # ── History / export / trends ────────────────────────

    def _refresh_session_dropdowns(self):
        sessions = get_session_list()
        self._sess1_dropdown["values"] = sessions
        self._sess2_dropdown["values"] = sessions

    def _compare_sessions(self):
        s1, s2 = self._sess1_var.get(), self._sess2_var.get()
        if s1 == "None" or s2 == "None":
            messagebox.showwarning("Selection Required", "Select two sessions to compare")
            return
        history = load_history()
        try:
            i1, i2 = int(s1.split()[1]) - 1, int(s2.split()[1]) - 1
            cmp = compare_two_sessions(history[i1], history[i2],
                                       name1=s1.split("(")[0].strip(),
                                       name2=s2.split("(")[0].strip())
        except (IndexError, ValueError) as e:
            messagebox.showerror("Error", f"Could not compare: {e}")
            return
        _write(self._compare_txt, format_session_comparison(cmp))
        self._nb.select(0)
        self._setstatus(f"Compared: Session {i1 + 1} vs {i2 + 1}", GREEN)

    def _export_session(self):
        if not self._current_result:
            messagebox.showwarning("No Session", "Record or upload a session first")
            return
        as_csv = messagebox.askyesno("Export Format", "Yes = CSV\nNo = JSON")
        ext, kind, fn = ((".csv", "session_csv", export_session_csv) if as_csv
                         else (".json", "session_json", export_session_json))
        filename = filedialog.asksaveasfilename(
            defaultextension=ext, initialfile=get_default_export_filename(kind),
            filetypes=[(ext.upper()[1:] + " files", "*" + ext), ("All files", "*.*")])
        if not filename:
            return
        result = fn(self._current_result, filename)
        if result.get("success"):
            self._setstatus(f"Exported: {os.path.basename(filename)}", GREEN)
        else:
            messagebox.showerror("Export Failed", result["message"])

    def _update_trends_display(self):
        try:
            _write(self._trends_txt, get_trend_summary(load_history()))
        except Exception as e:
            _write(self._trends_txt, f"Error loading trends: {e}")

    def _open_trend_chart(self):
        history = load_history()
        win = tk.Toplevel(self)
        win.title("Breathing Trends")
        win.configure(bg=BG)
        canvas = FigureCanvasTkAgg(plot_trending_chart(history), master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Plot ─────────────────────────────────────────────

    def _style_axes(self):
        titles = ("Waveform", "Envelope + Phases", "Breath Timeline", "Posture (match vs reference)")
        for ax, title in zip(self._axes, titles):
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=MUTED, fontsize=7, pad=2)
            ax.tick_params(colors=MUTED, labelsize=6)
            for sp in ax.spines.values():
                sp.set_color(BORDER)

    def _draw(self, r):
        ax1, ax2, ax3, ax4 = self._axes
        audio, sr = r["audio"], r["sr"]
        times, env = r["times"], r["envelope"]
        peaks, segs = r["peaks"], r["segments"]
        dur = max(len(audio) / sr, 1e-3)

        for ax in self._axes:
            ax.cla()
        self._style_axes()

        step = max(1, len(audio) // 20000)  # keep plotting fast for long recordings
        ax1.plot(np.arange(0, len(audio), step) / sr, audio[::step], color=ACCENT, lw=0.4, alpha=0.85)
        ax1.set_xlim(0, dur)

        for seg in segs:
            ax2.axvspan(seg["start"], seg["end"], alpha=0.22, color=PHASE_CLR.get(seg["phase"], BORDER))
        if len(env):
            ax2.plot(times, env, color="white", lw=1.0)
            pi = peaks["peak_indices"][peaks["peak_indices"] < len(env)]
            vi = peaks["valley_indices"][peaks["valley_indices"] < len(env)]
            if len(pi):
                ax2.scatter(times[pi], env[pi], color=ACCENT, s=20, zorder=5)
            if len(vi):
                ax2.scatter(times[vi], env[vi], color=GREEN, s=14, marker="v", zorder=5)
        ax2.set_xlim(0, dur)

        for seg in segs:
            ax3.barh(0, seg["end"] - seg["start"], left=seg["start"], height=0.6,
                     color=PHASE_CLR.get(seg["phase"], BORDER), edgecolor=BG, lw=0.3)
        ax3.set_xlim(0, dur)
        ax3.set_yticks([])
        ax3.legend(handles=[mpatches.Patch(color=PHASE_CLR[p], label=p) for p in PHASE_CLR],
                   loc="upper right", facecolor=PANEL, edgecolor=BORDER,
                   labelcolor=TEXT, fontsize=6, ncol=3)

        pose_data = r.get("pose_data")
        ax4.set_xlim(0, dur)
        ax4.set_yticks([])
        if pose_data:
            for seg in segs:
                ax4.axvspan(seg["start"], seg["end"], alpha=0.08, color=PHASE_CLR.get(seg["phase"], BORDER))
            # Runs of the same pose label as text
            run_start, run_label = pose_data[0]["timestamp"], pose_data[0]["pose"]
            runs = []
            for p in pose_data[1:] + [None]:
                if p is None or p["pose"] != run_label:
                    end = p["timestamp"] if p else pose_data[-1]["timestamp"]
                    runs.append((run_start, end, run_label))
                    if p:
                        run_start, run_label = p["timestamp"], p["pose"]
            for s, e, lab in runs:
                if e - s >= 1.0 and lab not in posemod.NOT_A_POSE and lab != "Unknown Pose":
                    ax4.text((s + e) / 2, 0.85, lab.replace(" Pose", ""), ha="center",
                             fontsize=6, color=TEXT, alpha=0.8)
            sim = (r.get("comparison") or {}).get("pose_comparison", {}).get("frame_similarity")
            if sim:
                ts, vals = zip(*sim)
                ax4.plot(ts, np.array(vals) / 100.0, color=ORANGE, lw=1.2)
                ax4.set_ylim(0, 1.05)
                ax4.set_yticks([0, 0.5, 1])
                ax4.set_yticklabels(["0", "50", "100%"])
            else:
                seen = [0.5 if p.get("angles") else 0.0 for p in pose_data]
                ax4.fill_between([p["timestamp"] for p in pose_data], seen, step="post",
                                 color=GREEN, alpha=0.3)
                ax4.set_ylim(0, 1)
        else:
            ax4.text(0.5, 0.5, "No posture data (record with camera or upload a video)",
                     ha="center", va="center", color=MUTED, fontsize=7, transform=ax4.transAxes)
        ax4.set_xlabel("Time (s)", color=MUTED, fontsize=6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # tight_layout complains while the tab is hidden/small
            self._fig.tight_layout(pad=0.6)
        self._canvas.draw()

    # ── Helpers ──────────────────────────────────────────

    def _lock(self, on: bool):
        self._busy_flag = on
        s = tk.DISABLED if on else tk.NORMAL
        for b in (self._rec_btn, self._upl_btn, self._ref_btn,
                  self._save_ref_btn, self._del_ref_btn, self._exp_btn):
            b.config(state=s)
        self._ref_dropdown.config(state=tk.DISABLED if on else "readonly")

    def _setstatus(self, msg, color=GREEN):
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)


# ══════════════════════════════════════════════════════════
# HISTORY WINDOW
# ══════════════════════════════════════════════════════════

def show_history():
    history = load_history()
    win = tk.Toplevel()
    win.title("Session History")
    win.geometry("880x420")
    win.configure(bg=BG)

    _lbl(win, "Session History", ACCENT, 13, bold=True).pack(anchor=tk.W, padx=14, pady=10)

    _style_notebook()
    cols = ("Date", "Score", "BPM", "IE Ratio", "Stability", "Cycles", "Coaching")
    tree = ttk.Treeview(win, columns=cols, show="headings", height=15)
    for col, w in zip(cols, [135, 55, 60, 75, 80, 60, 340]):
        tree.heading(col, text=col)
        tree.column(col, width=w, anchor=tk.W if w > 100 else tk.CENTER)

    for s in reversed(history):
        f = s.get("features", {})
        tree.insert("", tk.END, values=(
            s.get("timestamp", "")[:19].replace("T", " "),
            s.get("score", "—"),
            f.get("bpm", "—"),
            f"{f.get('inhale_exhale_ratio', '—')}x",
            f.get("stability_cv", "—"),
            f.get("num_cycles", "—"),
            (s.get("coaching", "") or "")[:80],
        ))

    vsb = ttk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0), pady=(0, 10))
    vsb.pack(side=tk.LEFT, fill=tk.Y, pady=(0, 10))
    if not history:
        _lbl(win, "No sessions yet.", MUTED, 10).pack(pady=20)


def _style_notebook():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TNotebook", background=PANEL, borderwidth=0)
    s.configure("TNotebook.Tab", background=BORDER, foreground=TEXT,
                padding=[8, 3], font=("Helvetica", 8))
    s.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", BG)])
    s.configure("Treeview", background=PANEL, foreground=TEXT,
                fieldbackground=PANEL, rowheight=22, font=("Helvetica", 8))
    s.configure("Treeview.Heading", background=BORDER, foreground=ACCENT,
                font=("Helvetica", 8, "bold"))
    s.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", BG)])


# ══════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self, detector):
        super().__init__()
        self.title("RespriaSnap — Yoga Breathing + Posture Analyzer")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1280, sw - 20)}x{min(900, sh - 70)}+5+5")
        self.minsize(min(1050, sw - 20), min(680, sh - 70))
        if os.name == "nt":
            self.state("zoomed")
        self.configure(bg=BG)
        self._build(detector)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _build(self, detector):
        bar = tk.Frame(self, bg=PANEL, height=46)
        bar.pack(fill=tk.X, padx=6, pady=6)
        bar.pack_propagate(False)
        _lbl(bar, "RespriaSnap", ACCENT, 14, bold=True).pack(side=tk.LEFT, padx=14)
        _lbl(bar, "Yoga Breathing + Posture Analyzer", MUTED, 9).pack(side=tk.LEFT)
        _btn(bar, "📋  Session History", show_history, ORANGE).pack(side=tk.RIGHT, padx=12)

        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X, padx=6)

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG, sashwidth=5, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._posture = PosturePanel(paned, detector)
        self._breath = BreathingPanel(paned, self._posture)
        paned.add(self._posture, minsize=370, width=460)
        paned.add(self._breath, minsize=560)

    def _quit(self):
        self._posture.cleanup()
        self.destroy()


if __name__ == "__main__":
    posemod.ensure_model()
    detector = posemod.build_detector()
    print("✅ Pose detector ready." if detector else
          "⚠️  Pose detector unavailable — webcam will work without skeleton/matching.")
    App(detector).mainloop()
