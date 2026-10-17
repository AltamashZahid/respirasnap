# RespriaSnap

Yoga practice analyzer that combines **posture matching** (webcam + MediaPipe) with **breathing pattern matching** (microphone). Load an instructor's video as a reference, follow along, and get scored on how closely your postures, breathing rate and breath timing match.

## Features
- **Posture matching**: live webcam skeleton, joint-angle comparison against the reference, live "Match %" while you record, per-joint feedback ("left knee off by ~25°")
- **Breathing analysis**: records audio, detects inhale/exhale/hold phases, extracts BPM, IE ratio, stability, and scores 0–100
- **Reference videos**: load any mp4/mov/mkv/avi; its audio (breathing) and video (postures) are extracted and saved for reuse
- **Comparison**: posture, breathing metrics and breath timing scored separately, plus an overall alignment score
- **AI insights** (optional): clinical summary, coaching tip, progress follow-up via Groq
- **History & trends**: every session is saved; compare any two sessions, view trend charts, export CSV/JSON

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10–3.13. ffmpeg is used for video/mp3 decoding: a system ffmpeg is used if it's on `PATH`, otherwise the one bundled with `imageio-ffmpeg`.

The pose model (`respirasnap/pose_landmarker_lite.task`) is included; it is downloaded automatically if missing.

### AI insights (optional)
```bash
cp .env.example .env
```
and set `GROQ_API_KEY` (free key at [console.groq.com](https://console.groq.com)). Without a key everything else still works.

## Usage

### GUI
```bash
python respirasnap/app.py
```

1. **🧘 Load Video**: pick the instructor's video. Breathing and postures are extracted and saved under *Reference* for next time. The recording length is set to the reference's length (max 180 s).
2. **🎙 Record**: after a 3 s countdown, mic and camera are captured together. The camera panel shows a live match % against the reference pose at that moment.
3. The **⚖️ Comparison** tab shows the overall score, posture match (per joint and per pose), breathing metrics and timing, with recommendations. **📉 Signals** shows the waveform, breath phases and the posture match over time.

Other buttons:
- **📂 Upload**: analyse your own audio or video file instead of recording.
- **⭐ Save Session as Ref**: save your last session as a reference, e.g. a teacher recording their own routine.
- **History … Compare**: compare two past sessions.

Tips: stand far enough back that shoulders to ankles are in frame. Mirroring is handled, so facing the instructor works.

### CLI (breathing only)
```bash
cd respirasnap
python main.py                     # record from mic (15 s)
python main.py --file audio.mp3    # analyse an audio/video file
python main.py --demo              # synthetic signal, no mic
python main.py --history           # session history
python main.py --no-llm            # skip Groq calls
```

### Tests
```bash
cd respirasnap
python -m pytest
```

## How matching works

**Posture.** Each video frame is reduced to 9 angles: elbows, shoulders, hips, knees and torso tilt. Angles make the comparison independent of body size, camera distance and position in the frame. Your sequence is aligned to the reference with dynamic time warping (DTW), so being a second behind doesn't count as wrong. Each aligned frame pair is scored from the RMS angle error (0° = 100%, ≥60° = 0%). The reference is also tried mirrored. Pose names (Warrior II, Tree, Chair, …) are for display only; scoring never depends on them.

**Breathing metrics.** BPM, IE ratio, stability and average inhale/exhale length are compared against tolerances (e.g. a 6 BPM difference scores 0).

**Breath timing.** Both inhale/exhale timelines are sampled at 10 Hz and compared, allowing up to ±2 s of lag.

**Overall** = weighted average (posture 40%, breathing 35%, timing 25%) of the parts that are available. An audio-only session isn't penalised for having no posture data.

## Architecture
| File | Role |
|---|---|
| `app.py` | Tkinter GUI: posture panel (owns the webcam) + breathing panel |
| `pose.py` | MediaPipe detection, joint angles, similarity, pose labels, video pose extraction |
| `compare.py` | Posture DTW, breathing metric + timing comparison, overall score, recommendations |
| `references.py` | Save/load references (features, breath segments, pose frames with angles) |
| `audio.py` | Mic recording, file loading (ffmpeg for video/mp3) |
| `preprocess.py`, `envelope.py`, `detect.py`, `segment.py` | Breathing signal pipeline |
| `features.py`, `score.py` | Breathing metrics and 0–100 score |
| `agents.py` | Groq LLM insights |
| `history.py`, `trending.py`, `export.py` | Session history, trends, CSV/JSON export |
| `main.py` | CLI entry point |
