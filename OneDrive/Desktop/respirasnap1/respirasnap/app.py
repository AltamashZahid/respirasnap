"""
app.py — RespriaSnap: Breathing Analysis + Posture Detection
Run:  python app.py

Layout:
  Left  — Posture: live webcam, skeleton overlay, pose label, pose log
  Right — Breathing: record/upload, score, metrics, waveform plots, AI tabs
"""

import sys, os, threading, math, urllib.request, time
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from datetime import datetime
import numpy as np
import sounddevice as sd

import cv2
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from audio      import record_mic, load_file
from preprocess import preprocess
from envelope   import extract_envelope
from detect     import detect_peaks_valleys
from segment    import segment_breathing
from features   import extract_features
from score      import compute_score
from agents     import clinical_summary_agent, coaching_agent, followup_agent
from history    import load_history, save_session
from compare    import overall_comparison
from references import save_reference, load_reference, list_references, delete_reference, reconstruct_result

# ── MediaPipe (optional) ──────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_OK = True; _MP_ERR = ""
except Exception as e:
    mp = None; _MP_OK = False; _MP_ERR = str(e)

# ── Theme ─────────────────────────────────────────────────
BG, PANEL, BORDER = "#0D1117", "#161B22", "#30363D"
ACCENT, GREEN, ORANGE, RED = "#4FC3F7", "#56D364", "#F0883E", "#F85149"
TEXT, MUTED = "#E6EDF3", "#8B949E"
PHASE_CLR = {"INHALE": "#4FC3F7", "EXHALE": "#56D364", "HOLD": "#F0883E"}

REC_SECS  = 15
CAM_W, CAM_H = 430, 310

# ── Pose model ────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
SKELETON = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]


# ══════════════════════════════════════════════════════════
# PRE-LAUNCH: download model on main thread (avoids
# "main thread is not in main loop" from background thread)
# ══════════════════════════════════════════════════════════
def ensure_model():
    """Download pose model if not present. Called before mainloop()."""
    if not _MP_OK:
        return
    if os.path.exists(MODEL_PATH):
        return
    try:
        print("Downloading pose model (~3 MB)…", end=" ", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("done.")
    except Exception as e:
        print(f"\nWarning: pose model download failed: {e}")


def build_detector():
    """Build MediaPipe PoseLandmarker. Call from main thread."""
    if not _MP_OK or not os.path.exists(MODEL_PATH):
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
        print(f"Detector init failed: {e}")
        return None


# ══════════════════════════════════════════════════════════
# POSE LOGIC
# ══════════════════════════════════════════════════════════

def _angle(a, b, c):
    ax,ay,_ = a; bx,by,_ = b; cx,cy,_ = c
    ang = math.degrees(math.atan2(cy-by,cx-bx) - math.atan2(ay-by,ax-bx))
    return ang + 360 if ang < 0 else ang


def classify_pose(lm):
    if len(lm) < 29:
        return "Unknown Pose"
    le = _angle(lm[11], lm[13], lm[15])
    re = _angle(lm[12], lm[14], lm[16])
    ls = _angle(lm[13], lm[11], lm[23])
    rs = _angle(lm[24], lm[12], lm[14])
    lk = _angle(lm[23], lm[25], lm[27])
    rk = _angle(lm[24], lm[26], lm[28])
    if (165<lk<195 and 165<rk<195 and
        130<le<180 and 175<re<220 and
        100<ls<200 and  50<rs<130):
        return "T Pose"
    if (165<le<195 and 165<re<195 and
        80<ls<110 and 80<rs<110):
        if ((165<lk<195 or 165<rk<195) and
            (90<lk<120  or 90<rk<120)):
            return "Warrior II Pose"
    if 165<lk<195 or 165<rk<195:
        if 315<lk<335 or 25<rk<45:
            return "Tree Pose"
    return "Unknown Pose"


def run_pose(detector, bgr):
    """Run pose on one BGR frame. Returns (annotated_bgr, label)."""
    out = bgr.copy()
    if detector is None:
        cv2.putText(out, "Pose model unavailable", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,200), 2)
        return out, "—"

    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        res = detector.detect(mp_img)
    except Exception:
        return out, "Detection error"

    if not res.pose_landmarks:
        return out, "No pose detected"

    pose = res.pose_landmarks[0]
    h, w = out.shape[:2]
    pts  = [(int(lm.x*w), int(lm.y*h), lm.z*w) for lm in pose]

    for a, b in SKELETON:
        if a < len(pts) and b < len(pts):
            cv2.line(out, pts[a][:2], pts[b][:2], (0,200,0), 2)
    for x,y,_ in pts:
        cv2.circle(out, (x,y), 4, (0,230,230), -1)

    label = classify_pose(pts)
    color = (0,220,0) if label != "Unknown Pose" else (0,80,220)
    cv2.putText(out, label, (10,36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out, label


# ══════════════════════════════════════════════════════════
# SYNCHRONIZED AUDIO + POSE RECORDING
# ══════════════════════════════════════════════════════════

def record_with_pose(detector, duration=15.0):
    """Record audio and capture pose frames simultaneously for given duration.
    Returns: (audio, sr, pose_data)
    where pose_data = [(timestamp, pose_label, frame), ...]
    """
    TARGET_SR = 16000
    pose_data = []
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Start recording audio in background thread
    audio_buffer = []
    stop_audio = threading.Event()
    
    def capture_audio():
        try:
            audio = sd.rec(
                int(duration * TARGET_SR),
                samplerate=TARGET_SR,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            audio_buffer.append(audio.flatten())
        except Exception as e:
            print(f"Audio capture error: {e}")
    
    audio_thread = threading.Thread(target=capture_audio, daemon=True)
    audio_thread.start()
    
    # Capture pose frames during recording
    start_time = time.time()
    frame_count = 0
    target_fps = 10  # capture ~10 pose frames per second
    frame_interval = 1.0 / target_fps
    next_frame_time = start_time
    
    print(f"📹 Recording audio + pose for {duration}s...")
    
    while time.time() - start_time < duration:
        current_time = time.time()
        
        if current_time >= next_frame_time:
            ok, frame = cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                # Fit to CAM_W x CAM_H
                h, w = frame.shape[:2]
                scale = CAM_H / h
                nw = int(w * scale)
                frame = cv2.resize(frame, (nw, CAM_H))
                if nw > CAM_W:
                    x0 = (nw - CAM_W) // 2
                    frame = frame[:, x0:x0+CAM_W]
                
                # Run pose detection
                annotated, label = run_pose(detector, frame)
                timestamp = current_time - start_time  # time relative to start
                pose_data.append({
                    "timestamp": timestamp,
                    "pose": label,
                    "frame": annotated
                })
                frame_count += 1
            
            next_frame_time += frame_interval
        
        time.sleep(0.01)  # Small sleep to avoid busy waiting
    
    cap.release()
    audio_thread.join(timeout=2.0)
    
    if not audio_buffer:
        raise RuntimeError("Audio recording failed")
    
    audio = audio_buffer[0]
    print(f"✅ Recorded: {len(audio)/TARGET_SR:.1f}s audio, {frame_count} pose frames")
    
    return audio, TARGET_SR, pose_data


def _extract_poses_from_video(video_path, detector, max_frames=None):
    """Extract pose frames from a video file.
    
    Returns: [{"timestamp": t, "pose": label, "frame": bgr}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pose_data = []
    frame_count = 0
    target_fps = 10  # capture ~10 frames per second
    frame_interval = int(fps / target_fps)
    frame_idx = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Sample frames at target rate
        if frame_idx % frame_interval == 0:
            # Fit frame
            h, w = frame.shape[:2]
            scale = CAM_H / h
            nw = int(w * scale)
            frame = cv2.resize(frame, (nw, CAM_H))
            if nw > CAM_W:
                x0 = (nw - CAM_W) // 2
                frame = frame[:, x0:x0+CAM_W]
            
            # Run pose detection
            annotated, label = run_pose(detector, frame)
            timestamp = frame_idx / fps  # time in seconds
            pose_data.append({
                "timestamp": timestamp,
                "pose": label,
                "frame": annotated
            })
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    return pose_data


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
                     pady=6, padx=8, state=state)

def _section_frame(parent, title):
    """Labelled section. Returns (outer_frame, inner_panel)."""
    outer = tk.Frame(parent, bg=BG)
    _lbl(outer, title.upper(), MUTED, 7, bold=True).pack(anchor=tk.W, pady=(0,2))
    inner = tk.Frame(outer, bg=PANEL, padx=8, pady=7)
    inner.pack(fill=tk.BOTH, expand=True)
    return outer, inner

def _txt_box(parent, height=4):
    t = scrolledtext.ScrolledText(parent, bg=PANEL, fg=TEXT,
                                   font=("Helvetica",9), wrap=tk.WORD,
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
# POSTURE PANEL
# ══════════════════════════════════════════════════════════

class PosturePanel(tk.Frame):
    def __init__(self, parent, detector):
        super().__init__(parent, bg=BG)
        self._det     = detector
        self._cap     = None
        self._running = False
        self._blank   = None
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill=tk.X, pady=(0,6))
        _lbl(hdr, "📷  Posture Detection", ACCENT, 11, bold=True).pack(side=tk.LEFT)

        # Camera display
        border = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        border.pack()
        self._cam = tk.Label(border, bg="#111111",
                              width=CAM_W, height=CAM_H)
        self._cam.pack()
        self._blank = self._make_blank("Camera off — press Start")
        self._cam.configure(image=self._blank)
        self._cam.image = self._blank

        # Pose label
        self._pose_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._pose_var, bg=BG, fg=GREEN,
                 font=("Helvetica",13,"bold")).pack(pady=(5,2))

        # Buttons
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=3)
        self._start_btn = _btn(btn_row, "▶  Start Camera",
                                self._start, GREEN)
        self._start_btn.pack(side=tk.LEFT, padx=3)
        self._stop_btn  = _btn(btn_row, "■  Stop",
                                self._stop, RED, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=3)

        # Pose log
        log_outer, log_inner = _section_frame(self, "Pose Log")
        log_outer.pack(fill=tk.X, pady=(8,0))
        top_row = tk.Frame(log_inner, bg=PANEL)
        top_row.pack(fill=tk.X, pady=(0,3))
        _btn(top_row, "Clear", self._clear_log, MUTED).pack(side=tk.RIGHT)
        self._log = scrolledtext.ScrolledText(
            log_inner, bg="#0d1117", fg=TEXT,
            font=("Courier",8), relief=tk.FLAT,
            height=5, state=tk.DISABLED)
        self._log.pack(fill=tk.X)

    def _make_blank(self, msg=""):
        img  = Image.new("RGB", (CAM_W, CAM_H), "#111111")
        photo = ImageTk.PhotoImage(img)
        return photo

    # ── Camera start / stop ──────────────────────────────

    def _start(self):
        if self._running: return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            self._cap = None; return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._running = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._pose_var.set("Detecting…")
        self._poll()     # begins tkinter-scheduled loop (main thread)

    def _stop(self):
        self._running = False
        if self._cap:
            self._cap.release(); self._cap = None
        self._cam.configure(image=self._blank)
        self._cam.image = self._blank
        self._pose_var.set("—")
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)

    def _poll(self):
        """Read one webcam frame, run pose, update label. Reschedule via after()."""
        if not self._running: return

        ok, frame = self._cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            # Fit to CAM_W x CAM_H
            h, w = frame.shape[:2]
            scale = CAM_H / h
            nw    = int(w * scale)
            frame = cv2.resize(frame, (nw, CAM_H))
            if nw > CAM_W:
                x0 = (nw - CAM_W) // 2
                frame = frame[:, x0:x0+CAM_W]

            annotated, label = run_pose(self._det, frame)
            self._pose_var.set(label)

            if label not in ("—","No pose detected","Detection error",
                             "Detecting…","Pose model unavailable"):
                self._log_entry(label)

            rgb   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._cam.configure(image=photo)
            self._cam.image = photo   # keep reference

        self.after(33, self._poll)    # ~30 fps

    def _log_entry(self, label):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.config(state=tk.NORMAL)
        self._log.insert("1.0", f"[{ts}]  {label}\n")
        # Keep last 50 lines
        lines = int(self._log.index("end-1c").split(".")[0])
        if lines > 50:
            self._log.delete("51.0", tk.END)
        self._log.config(state=tk.DISABLED)

    def _clear_log(self):
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

    def cleanup(self):
        self._running = False
        if self._cap: self._cap.release()


# ══════════════════════════════════════════════════════════
# BREATHING PANEL
# ══════════════════════════════════════════════════════════

class BreathingPanel(tk.Frame):
    def __init__(self, parent, detector=None):
        super().__init__(parent, bg=BG)
        self._audio     = None
        self._sr        = 16000
        self._pose_data = None  # Store pose frames from synchronized recording
        self._detector  = detector  # For synchronized pose+audio recording
        self._recording = False
        self._analyzing = False
        self._reference_result = None  # Store reference video analysis
        self._build()

    def _build(self):
        # ── Header ───────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill=tk.X, pady=(0,6))
        _lbl(hdr, "🫁  Breathing Analysis", ACCENT, 11, bold=True).pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Ready")
        self._status_lbl = tk.Label(hdr, textvariable=self._status_var,
                                     bg=BG, fg=GREEN, font=("Helvetica",9))
        self._status_lbl.pack(side=tk.RIGHT)

        # ── Control row ──────────────────────────────────
        ctrl_outer, ctrl = _section_frame(self, "Input")
        ctrl_outer.pack(fill=tk.X, pady=(0,6))

        btn_row = tk.Frame(ctrl, bg=PANEL)
        btn_row.pack(fill=tk.X)
        self._rec_btn = _btn(btn_row, f"🎙  Record {REC_SECS}s", self._start_record, ACCENT)
        self._rec_btn.pack(side=tk.LEFT, padx=(0,6))
        self._upl_btn = _btn(btn_row, "📂  Upload File", self._upload_file, MUTED)
        self._upl_btn.pack(side=tk.LEFT, padx=(0,6))
        self._ref_btn = _btn(btn_row, "🧘  Load Reference", self._upload_reference, ORANGE)
        self._ref_btn.pack(side=tk.LEFT)

        # ── Reference management row ────────────────────
        ref_row = tk.Frame(ctrl, bg=PANEL)
        ref_row.pack(fill=tk.X, pady=(6,0))
        
        _lbl(ref_row, "Saved References:", MUTED, 7).pack(side=tk.LEFT, padx=(0,6))
        
        self._ref_var = tk.StringVar(value="None")
        self._ref_dropdown = ttk.Combobox(
            ref_row, textvariable=self._ref_var, state="readonly", width=30,
            values=self._get_reference_list()
        )
        self._ref_dropdown.pack(side=tk.LEFT, padx=(0,6))
        self._ref_dropdown.bind("<<ComboboxSelected>>", lambda e: self._load_saved_reference())
        
        self._save_ref_btn = _btn(ref_row, "💾 Save as Reference", self._save_current_reference, GREEN)
        self._save_ref_btn.pack(side=tk.LEFT, padx=(0,6))
        
        self._del_ref_btn = _btn(ref_row, "🗑  Delete", self._delete_saved_reference, RED)
        self._del_ref_btn.pack(side=tk.LEFT)

        self._progress = ttk.Progressbar(ctrl, mode="indeterminate")
        self._progress.pack(fill=tk.X, pady=(6,2))
        self._file_lbl = _lbl(ctrl, "wav · mp3 · mp4 · m4a · ogg · flac",
                               MUTED, 7)
        self._file_lbl.pack(anchor=tk.W)

        # ── Score + Metrics row ──────────────────────────
        sm_row = tk.Frame(self, bg=BG)
        sm_row.pack(fill=tk.X, pady=(0,6))

        # Score
        sc_outer, sc = _section_frame(sm_row, "Score")
        sc_outer.pack(side=tk.LEFT, padx=(0,6))
        self._score_var = tk.StringVar(value="—")
        tk.Label(sc, textvariable=self._score_var, bg=PANEL,
                 fg=ACCENT, font=("Helvetica",38,"bold"),
                 width=4).pack()
        _lbl(sc, "/ 100", MUTED, 8).pack()

        # Metrics
        met_outer, met = _section_frame(sm_row, "Metrics")
        met_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._mvars = {}
        fields = [
            ("BPM",          "bpm",                  ""),
            ("IE Ratio",     "inhale_exhale_ratio",   "x"),
            ("Stability CV", "stability_cv",          ""),
            ("Interrupts",   "interruptions",         ""),
            ("Avg Inhale",   "avg_inhale_s",          "s"),
            ("Avg Exhale",   "avg_exhale_s",          "s"),
            ("Holds",        "holds_detected",        ""),
            ("Cycles",       "num_cycles",            ""),
        ]
        for i, (label, key, unit) in enumerate(fields):
            row, col = divmod(i, 2)
            _lbl(met, label, MUTED, 8).grid(
                row=row*2, column=col*2, sticky=tk.W, padx=(0,4))
            var = tk.StringVar(value="—")
            self._mvars[key] = (var, unit)
            tk.Label(met, textvariable=var, bg=PANEL, fg=TEXT,
                     font=("Helvetica",8,"bold")).grid(
                row=row*2+1, column=col*2, sticky=tk.W, padx=(0,16))

        # ── Matplotlib plot ──────────────────────────────
        self._fig = Figure(figsize=(6,3.8), facecolor=BG)
        self._fig.subplots_adjust(hspace=0.5, top=0.93, bottom=0.1)
        self._ax1 = self._fig.add_subplot(411, facecolor=PANEL)
        self._ax2 = self._fig.add_subplot(412, facecolor=PANEL)
        self._ax3 = self._fig.add_subplot(413, facecolor=PANEL)
        self._ax4 = self._fig.add_subplot(414, facecolor=PANEL)
        self._style_axes()
        cv = FigureCanvasTkAgg(self._fig, master=self)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = cv

        # ── AI tabs ──────────────────────────────────────
        ai_outer, ai = _section_frame(self, "AI Analysis  (Groq · Llama 3)")
        ai_outer.pack(fill=tk.X, pady=(4,0))

        _style_notebook()
        nb = ttk.Notebook(ai)
        nb.pack(fill=tk.BOTH, expand=True)
        self._clinical_txt = self._add_tab(nb, "🩺 Clinical")
        self._coaching_txt = self._add_tab(nb, "💡 Coaching")
        self._followup_txt = self._add_tab(nb, "📈 Follow-up")
        self._compare_txt = self._add_tab(nb, "⚖️  Comparison")

    def _add_tab(self, nb, title):
        f = tk.Frame(nb, bg=PANEL)
        nb.add(f, text=title)
        return _txt_box(f, height=4)

    # ── Input handlers ───────────────────────────────────

    def _start_record(self):
        if self._busy(): return
        self._lock(True)
        self._progress.start(10)
        self._setstatus("Recording…", ACCENT)
        self._file_lbl.config(text="Live microphone")
        threading.Thread(target=self._rec_thread, daemon=True).start()

    def _rec_thread(self):
        try:
            # Use synchronized recording if detector available, otherwise just record audio
            if self._detector:
                audio, sr, pose_data = record_with_pose(self._detector, REC_SECS)
                self._pose_data = pose_data
            else:
                audio, sr = record_mic(REC_SECS)
                self._pose_data = None
            
            self._audio, self._sr = audio, sr
            self.after(0, self._on_audio_ready)
        except Exception as e:
            self.after(0, self._on_error, str(e))

    def _upload_file(self):
        if self._busy(): return
        path = filedialog.askopenfilename(
            title="Select Audio / Video File",
            filetypes=[
                ("Audio/Video", "*.wav *.mp3 *.mp4 *.m4a *.ogg *.flac *.aac"),
                ("All files",   "*.*"),
            ]
        )
        if not path: return
        self._lock(True)
        self._progress.start(10)
        self._setstatus("Loading…", ACCENT)
        self._file_lbl.config(text=os.path.basename(path))
        threading.Thread(target=self._load_thread,
                          args=(path,), daemon=True).start()

    def _load_thread(self, path):
        try:
            audio, sr = load_file(path)
            self._audio, self._sr = audio, sr
            self.after(0, self._on_audio_ready)
        except Exception as e:
            self.after(0, self._on_error, str(e))

    def _upload_reference(self):
        """Upload and process a reference video or load from JSON."""
        if self._busy(): return
        path = filedialog.askopenfilename(
            title="Select Reference Video or JSON",
            filetypes=[
                ("All supported", "*.mp4 *.mov *.avi *.mkv *.wav *.mp3 *.m4a *.ogg *.flac *.aac *.json"),
                ("JSON reference", "*.json"),
                ("Video files", "*.mp4 *.mov *.avi *.mkv"),
                ("Audio/Video", "*.wav *.mp3 *.mp4 *.m4a *.ogg *.flac *.aac"),
                ("All files",   "*.*"),
            ]
        )
        if not path: return

        # If JSON selected, load directly without recording
        if path.lower().endswith(".json"):
            try:
                import json as _json
                with open(path, "r") as f:
                    data = _json.load(f)

                # analysis.json has timestamps + poses + features
                if "timestamps" in data and "poses" in data:
                    features = data.get("features", {})
                    pose_data = [
                        {"timestamp": t, "pose": p}
                        for t, p in zip(data["timestamps"], data["poses"])
                    ]
                    ref_name = os.path.basename(path)
                # metadata.json has features only
                elif "features" in data:
                    features = data["features"]
                    pose_data = None
                    ref_name = data.get("name", os.path.basename(path))
                else:
                    messagebox.showerror("Error", "JSON file doesn't contain valid reference data.")
                    return

                self._reference_result = {
                    "features":  features,
                    "segments":  [],
                    "envelope":  np.array([]),
                    "times":     np.array([]),
                    "peaks":     {"peak_indices": np.array([]), "valley_indices": np.array([])},
                    "audio":     np.zeros(16000),
                    "sr":        16000,
                    "pose_data": pose_data,
                    "path":      ref_name,
                }
                self._setstatus(f"Reference loaded: {ref_name}", GREEN)
                _write(self._compare_txt, f"✓ Reference loaded from JSON\n\n{ref_name}\n\nNow record your session to compare.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON: {e}")
            return

        # Otherwise treat as video/audio file as before
        self._lock(True)
        self._progress.start(10)
        self._setstatus("Loading reference…", ORANGE)
        threading.Thread(target=self._ref_thread,
                          args=(path,), daemon=True).start()

    def _ref_thread(self, path):
        """Process reference video: extract audio and poses."""
        try:
            audio, sr = load_file(path)
            
            # Extract audio features
            clean      = preprocess(audio, sr)
            times, env = extract_envelope(clean, sr)
            peaks      = detect_peaks_valleys(env, times)
            segs       = segment_breathing(env, times, peaks)
            features   = extract_features(segs, peaks, len(audio)/sr)
            
            # Extract poses if detector available
            pose_data = None
            if self._detector:
                try:
                    # Try to extract poses from video file
                    pose_data = _extract_poses_from_video(path, self._detector)
                except Exception as e:
                    print(f"Could not extract poses from reference: {e}")
            
            self._reference_result = {
                "features": features,
                "segments": segs,
                "envelope": env,
                "times": times,
                "peaks": peaks,
                "audio": audio,
                "sr": sr,
                "pose_data": pose_data,
                "path": os.path.basename(path),
            }
            
            self.after(0, self._on_reference_loaded)
        except Exception as e:
            self.after(0, self._on_error, str(e))

    def _on_reference_loaded(self):
        """Called when reference is loaded."""
        self._progress.stop()
        self._lock(False)
        self._setstatus(f"Reference loaded: {self._reference_result['path']}", GREEN)
        _write(self._compare_txt, f"✓ Reference loaded\n\n{self._reference_result['path']}\n\nNow record or upload your session to compare.")

    def _get_reference_list(self):
        """Get list of saved references for dropdown."""
        refs = list_references()
        if not refs:
            return ["None"]
        return [f"{r['name']} ({r['created']})" for r in refs]

    def _save_current_reference(self):
        """Save the current reference for future use."""
        if not self._reference_result and self._audio is not None:
            clean      = preprocess(self._audio, self._sr)
            times, env = extract_envelope(clean, self._sr)
            peaks      = detect_peaks_valleys(env, times)
            segs       = segment_breathing(env, times, peaks)
            features   = extract_features(segs, peaks, len(self._audio)/self._sr)
            self._reference_result = {
                "features":  features,
                "segments":  segs,
                "envelope":  env,
                "times":     times,
                "peaks":     peaks,
                "audio":     self._audio,
                "sr":        self._sr,
                "pose_data": self._pose_data,
                "path":      "my_recording",
            }
        if not self._reference_result:
            messagebox.showwarning("No Reference", "Record or upload a session first.")
            return
        
        # Ask for reference name
        dialog = tk.Toplevel(self)
        dialog.title("Save Reference")
        dialog.geometry("400x150")
        dialog.configure(bg=BG)
        dialog.transient(self.winfo_toplevel())
        
        tk.Label(dialog, text="Reference Name:", bg=BG, fg=TEXT).pack(pady=10)
        
        entry = tk.Entry(dialog, width=40, bg=PANEL, fg=TEXT, insertbackground=TEXT)
        entry.pack(pady=5)
        entry.insert(0, "Yoga Master - Sun Salutation")
        entry.focus()
        
        def save():
            name = entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a name")
                return
            
            result = save_reference(name, self._reference_result, 
                                  self._reference_result.get("path", "unknown"))
            
            if result["success"]:
                messagebox.showinfo("Saved", f"Reference saved: {name}")
                self._refresh_reference_dropdown()
                dialog.destroy()
            else:
                messagebox.showerror("Error", result["message"])
        
        tk.Button(dialog, text="Save", command=save, bg=GREEN, fg=BG,
                 font=("Helvetica", 10, "bold")).pack(pady=10)

    def _load_saved_reference(self):
        """Load a saved reference from the dropdown."""
        selected = self._ref_var.get()
        if selected == "None" or not selected:
            return
        
        # Extract ref_id from dropdown text
        refs = list_references()
        for ref in refs:
            ref_display = f"{ref['name']} ({ref['created']})"
            if selected == ref_display:
                ref_data = load_reference(ref["ref_id"])
                if ref_data:
                    # Reconstruct result for comparison
                    self._reference_result = reconstruct_result(
                        ref_data,
                        np.zeros(int(ref_data["analysis"]["duration"] * 16000)),  # dummy audio
                        16000
                    )
                    self._reference_result["path"] = ref["name"]
                    self._setstatus(f"Reference: {ref['name']}", GREEN)
                    _write(self._compare_txt, f"✓ Reference loaded: {ref['name']}\n\nNow record or upload your session to compare.")
                break

    def _delete_saved_reference(self):
        """Delete a saved reference."""
        selected = self._ref_var.get()
        if selected == "None" or not selected:
            messagebox.showwarning("No Selection", "Select a reference to delete")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Delete reference?\n{selected}"):
            refs = list_references()
            for ref in refs:
                ref_display = f"{ref['name']} ({ref['created']})"
                if selected == ref_display:
                    result = delete_reference(ref["ref_id"])
                    if result["success"]:
                        messagebox.showinfo("Deleted", result["message"])
                        self._reference_result = None
                        self._ref_var.set("None")
                        self._refresh_reference_dropdown()
                    else:
                        messagebox.showerror("Error", result["message"])
                    break

    def _refresh_reference_dropdown(self):
        """Refresh the reference dropdown list."""
        refs_list = self._get_reference_list()
        self._ref_dropdown["values"] = refs_list
        self._ref_var.set("None")

    def _on_audio_ready(self):
        self._analyzing = True
        self._setstatus("Analysing…", ORANGE)
        threading.Thread(target=self._analysis_thread, daemon=True).start()

    def _analysis_thread(self):
        try:
            audio, sr = self._audio, self._sr
            clean      = preprocess(audio, sr)
            times, env = extract_envelope(clean, sr)
            peaks      = detect_peaks_valleys(env, times)
            segs       = segment_breathing(env, times, peaks)
            features   = extract_features(segs, peaks, len(audio)/sr)
            score_res  = compute_score(features)
            score      = score_res["score"]

            history  = load_history()
            clinical = clinical_summary_agent(features)
            coaching = coaching_agent(features)
            followup = followup_agent(
                {"score": score, "features": features}, history)

            save_session(features, score_res, clinical, coaching, followup)

            self.after(0, self._on_done, {
                "features": features, "score": score,
                "clinical": clinical, "coaching": coaching,
                "followup": followup,
                "segments": segs, "envelope": env,
                "times": times, "peaks": peaks,
                "audio": audio, "sr": sr,
                "pose_data": self._pose_data,
            })
        except Exception as e:
            import traceback
            self.after(0, self._on_error, traceback.format_exc())

    def _on_done(self, r):
        self._analyzing = False
        self._progress.stop()
        self._lock(False)
        self._setstatus("Done ✓", GREEN)

        self._score_var.set(str(r["score"]))
        for key, (var, unit) in self._mvars.items():
            v = r["features"].get(key, "—")
            var.set(f"{v}{unit}" if v != "—" else "—")

        _write(self._clinical_txt, r["clinical"])
        _write(self._coaching_txt, r["coaching"])
        _write(self._followup_txt, r["followup"])
        
        # Show comparison if reference is loaded
        if self._reference_result:
            comparison = overall_comparison(r, self._reference_result)
            self._show_comparison(comparison)
        
        self._draw(r)

    def _on_error(self, msg):
        self._analyzing = self._recording = False
        self._progress.stop()
        self._lock(False)
        self._setstatus("Error", RED)
        messagebox.showerror("Error", str(msg)[:800])

    def _show_comparison(self, comparison):
        """Display comparison results in the comparison tab."""
        overall = comparison["overall_alignment_score"]
        pose_acc = comparison["pose_comparison"]["pose_accuracy"]
        breathe_acc = comparison["breathing_comparison"]["breathing_alignment_score"]
        recs = comparison["recommendations"]
        
        text = f"⚖️  COMPARISON ANALYSIS\n"
        text += f"{'='*45}\n\n"
        text += f"Overall Alignment Score: {overall}/100\n"
        text += f"Pose Accuracy: {pose_acc}%\n"
        text += f"Breathing Alignment: {breathe_acc}%\n\n"
        
        # Breathing metrics comparison
        metrics = comparison["breathing_comparison"]["metrics_comparison"]
        text += "📊 Breathing Metrics:\n"
        text += f"{'─'*45}\n"
        for key, data in metrics.items():
            label = data["label"]
            user_val = data["user"]
            ref_val = data["reference"]
            diff = data["diff_pct"]
            text += f"{label:15} You: {user_val:>6}  Ref: {ref_val:>6}  Diff: {diff:>5}%\n"
        
        text += f"\n💡 Recommendations:\n"
        text += f"{'─'*45}\n"
        for rec in recs:
            text += f"• {rec}\n"
        
        _write(self._compare_txt, text)

    # ── Plot ─────────────────────────────────────────────

    def _style_axes(self):
        for ax, title in zip(
            (self._ax1, self._ax2, self._ax3, self._ax4),
            ("Waveform", "Envelope + Phases", "Breath Timeline", "Pose Sync")
        ):
            ax.set_facecolor(PANEL)
            ax.set_title(title, color=MUTED, fontsize=7, pad=2)
            ax.tick_params(colors=MUTED, labelsize=6)
            for sp in ax.spines.values():
                sp.set_color(BORDER)

    def _draw(self, r):
        audio, sr    = r["audio"], r["sr"]
        times, env   = r["times"], r["envelope"]
        peaks, segs  = r["peaks"], r["segments"]
        pose_data    = r.get("pose_data")
        dur = len(audio)/sr

        for ax in (self._ax1, self._ax2, self._ax3, self._ax4):
            ax.cla()
        self._style_axes()

        # Waveform
        self._ax1.plot(np.linspace(0, dur, len(audio)),
                        audio, color=ACCENT, lw=0.4, alpha=0.85)
        self._ax1.set_xlim(0, dur)
        self._ax1.axhline(0, color=BORDER, lw=0.4)

        # Envelope
        for seg in segs:
            self._ax2.axvspan(seg["start"], seg["end"], alpha=0.22,
                               color=PHASE_CLR.get(seg["phase"], BORDER))
        self._ax2.plot(times, env, color="white", lw=1.0)
        pi = peaks["peak_indices"];   pi = pi[pi < len(env)]
        vi = peaks["valley_indices"]; vi = vi[vi < len(env)]
        if len(pi):
            self._ax2.scatter(times[pi], env[pi],
                               color=ACCENT, s=25, zorder=5)
        if len(vi):
            self._ax2.scatter(times[vi], env[vi],
                               color=GREEN, s=18, marker="v", zorder=5)
        self._ax2.set_xlim(0, dur)

        # Timeline
        for seg in segs:
            self._ax3.barh(0, seg["end"]-seg["start"], left=seg["start"],
                            height=0.6,
                            color=PHASE_CLR.get(seg["phase"], BORDER),
                            edgecolor=BG, lw=0.3)
        self._ax3.set_xlim(0, dur)
        self._ax3.set_yticks([])
        self._ax3.set_xlabel("Time (s)", color=MUTED, fontsize=6)
        patches = [mpatches.Patch(color=PHASE_CLR[p], label=p)
                   for p in ("INHALE","EXHALE","HOLD")]
        self._ax3.legend(handles=patches, loc="upper right",
                          facecolor=PANEL, edgecolor=BORDER,
                          labelcolor=TEXT, fontsize=6, ncol=3)

        # Pose Synchronization
        if pose_data:
            pose_dict = {}  # timestamp -> pose label
            for pd in pose_data:
                pose_dict[pd["timestamp"]] = pd["pose"]
            
            # Build pose change timeline
            pose_times = sorted(pose_dict.keys())
            if pose_times:
                pose_labels = [pose_dict[t] for t in pose_times]
                
                # Normalize pose labels for coloring
                pose_colors = {
                    "T Pose": "#FF6B6B",
                    "Warrior II Pose": "#4ECDC4",
                    "Tree Pose": "#45B7D1",
                    "Unknown Pose": "#8B949E",
                    "No pose detected": MUTED,
                    "—": MUTED,
                    "Detecting…": MUTED,
                }
                
                # Draw pose changes as vertical lines
                for i, t in enumerate(pose_times):
                    label = pose_labels[i]
                    color = pose_colors.get(label, MUTED)
                    self._ax4.axvline(t, color=color, alpha=0.5, lw=1.5)
                
                # Add text labels for major poses
                pose_changes = []
                for i, t in enumerate(pose_times):
                    if i == 0 or pose_labels[i] != pose_labels[i-1]:
                        pose_changes.append((t, pose_labels[i]))
                
                for t, label in pose_changes[:5]:  # Show max 5 labels
                    self._ax4.text(t, 0.5, label.replace(" Pose", ""),
                                   rotation=45, fontsize=5, color=TEXT,
                                   alpha=0.7, ha="left")
                
                # Shade breathing phases as background
                for seg in segs:
                    self._ax4.axvspan(seg["start"], seg["end"], alpha=0.1,
                                     color=PHASE_CLR.get(seg["phase"], BORDER))
                
                self._ax4.set_xlim(0, dur)
                self._ax4.set_ylim(0, 1)
                self._ax4.set_yticks([])
                self._ax4.set_xlabel("Time (s)", color=MUTED, fontsize=6)
            else:
                self._ax4.text(0.5, 0.5, "No pose data", ha="center", va="center",
                              color=MUTED, fontsize=8, transform=self._ax4.transAxes)
                self._ax4.set_xlim(0, dur)
        else:
            self._ax4.text(0.5, 0.5, "Upload audio or record with webcam for pose sync",
                          ha="center", va="center", color=MUTED, fontsize=7,
                          transform=self._ax4.transAxes)
            self._ax4.set_xlim(0, dur)

        self._fig.tight_layout(pad=0.8)
        self._canvas.draw()

    # ── Helpers ──────────────────────────────────────────

    def _busy(self):
        return self._recording or self._analyzing

    def _lock(self, on: bool):
        s = tk.DISABLED if on else tk.NORMAL
        self._rec_btn.config(state=s)
        self._upl_btn.config(state=s)

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

    _lbl(win, "Session History", ACCENT, 13, bold=True).pack(
        anchor=tk.W, padx=14, pady=10)

    _style_notebook()   # also styles Treeview
    cols = ("Date", "Score", "BPM", "IE Ratio", "Stability", "Cycles", "Coaching")
    tree = ttk.Treeview(win, columns=cols, show="headings", height=15)
    for col, w in zip(cols, [135,55,60,75,80,60,340]):
        tree.heading(col, text=col)
        tree.column(col, width=w, anchor=tk.W if w>100 else tk.CENTER)

    for s in reversed(history):
        f = s.get("features", {})
        ts = s.get("timestamp","")[:19].replace("T"," ")
        tree.insert("", tk.END, values=(
            ts,
            s.get("score","—"),
            f.get("bpm","—"),
            f"{f.get('inhale_exhale_ratio','—')}x",
            f.get("stability_cv","—"),
            f.get("num_cycles","—"),
            (s.get("coaching","") or "")[:80],
        ))

    vsb = ttk.Scrollbar(win, orient=tk.VERTICAL,   command=tree.yview)
    hsb = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14,0), pady=(0,10))
    vsb.pack(side=tk.LEFT, fill=tk.Y, pady=(0,10))
    hsb.pack(side=tk.BOTTOM, fill=tk.X, padx=14)

    if not history:
        _lbl(win, "No sessions yet.", MUTED, 10).pack(pady=20)


def _style_notebook():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TNotebook",      background=PANEL, borderwidth=0)
    s.configure("TNotebook.Tab",  background=BORDER, foreground=TEXT,
                padding=[8,3], font=("Helvetica",8))
    s.map("TNotebook.Tab",
          background=[("selected", ACCENT)],
          foreground=[("selected", BG)])
    s.configure("Treeview",
                background=PANEL, foreground=TEXT,
                fieldbackground=PANEL, rowheight=22,
                font=("Helvetica",8))
    s.configure("Treeview.Heading",
                background=BORDER, foreground=ACCENT,
                font=("Helvetica",8,"bold"))
    s.map("Treeview",
          background=[("selected", ACCENT)],
          foreground=[("selected", BG)])


# ══════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self, detector):
        super().__init__()
        self.title("RespriaSnap — Breathing + Posture Analyzer")
        self.geometry("1200x820")
        self.minsize(1000, 720)
        self.configure(bg=BG)
        self._detector = detector
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _build(self):
        # Top bar
        bar = tk.Frame(self, bg=PANEL, height=48)
        bar.pack(fill=tk.X, padx=6, pady=6)
        bar.pack_propagate(False)
        _lbl(bar, "RespriaSnap", ACCENT, 14, bold=True).pack(
            side=tk.LEFT, padx=14)
        _lbl(bar, "Breathing + Posture Analyzer", MUTED, 9).pack(
            side=tk.LEFT)
        _btn(bar, "📋  Session History", show_history, ORANGE).pack(
            side=tk.RIGHT, padx=12)

        # Divider
        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X, padx=6)

        # Main paned area
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                                bg=BG, sashwidth=5, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._posture  = PosturePanel(paned, detector)
        self._breath   = BreathingPanel(paned, detector)

        paned.add(self._posture, minsize=370, width=470)
        paned.add(self._breath,  minsize=520)

    def _quit(self):
        self._posture.cleanup()
        self.destroy()


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Download model on MAIN thread (avoids threading errors)
    ensure_model()

    # 2. Build MediaPipe detector on MAIN thread (thread-safe)
    detector = build_detector()
    if detector:
        print("✅ Pose detector ready.")
    else:
        print("⚠️  Pose detector unavailable — webcam will still work without skeleton.")

    # 3. Launch GUI
    App(detector).mainloop()
