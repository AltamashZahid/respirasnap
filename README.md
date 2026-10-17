# RespriaSnap

AI-powered breathing analysis tool with real-time posture detection.

## Features
- **Breathing Analysis**: Records audio, analyzes breathing patterns, extracts metrics
- **Posture Detection**: Live webcam feed with skeleton overlay via MediaPipe
- **AI Insights**: Clinical summary, coaching tips, progress tracking
- **Visualization**: Waveform plots and analysis charts
- **Session History**: Track breathing improvements over time

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Setup

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

## Usage

### GUI (Recommended)
```bash
python respirasnap/app.py
```

### CLI
```bash
cd respirasnap
python main.py              # Record from mic (15s)
python main.py --file audio.mp3  # Analyze audio file
python main.py --demo       # Synthetic demo
python main.py --history    # View session history
```

## Requirements
- Python 3.9+
- See `requirements.txt` for dependencies

## Architecture
- `app.py` — GUI interface (Tkinter)
- `main.py` — CLI entry point
- `agents.py` — LLM integration (Groq)
- `audio.py` — Audio recording/loading
- `preprocess.py` — Signal preprocessing
- `envelope.py`, `detect.py`, `segment.py` — Signal processing
- `features.py` — Feature extraction
- `score.py` — Breathing quality scoring
- `references.py` — Reference management
- `visualize.py` — Plotting and visualization
