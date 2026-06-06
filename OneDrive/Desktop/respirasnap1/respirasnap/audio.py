"""
audio.py — Mic recording and audio file loading
Supported formats: .wav, .mp3, .mp4, .m4a, .ogg, .flac, .aac, .webm
"""

import numpy as np
import subprocess
import sounddevice as sd
import librosa
from pathlib import Path

TARGET_SR = 16000

# Formats natively handled by soundfile (no ffmpeg needed)
NATIVE_FORMATS  = {".wav", ".flac", ".ogg"}

# Formats that need ffmpeg to decode first
FFMPEG_FORMATS  = {".mp3", ".mp4", ".m4a", ".aac", ".webm"}

SUPPORTED_FORMATS = NATIVE_FORMATS | FFMPEG_FORMATS


def record_mic(duration: float = 15.0) -> tuple[np.ndarray, int]:
    """Record from microphone for given duration (seconds)."""
    print(f"🎙️  Recording for {duration}s... breathe naturally.")
    audio = sd.rec(
        int(duration * TARGET_SR),
        samplerate=TARGET_SR,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("✅ Recording complete.")
    return audio.flatten(), TARGET_SR


def _load_via_ffmpeg(path: str) -> np.ndarray:
    """
    Use ffmpeg to decode any audio/video file to raw 16kHz mono PCM float32.
    Works cleanly for .mp4, .m4a, .mp3, .aac, .webm — no deprecation warnings.
    """
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "f32le",          # raw 32-bit float little-endian
        "-acodec", "pcm_f32le",
        "-ac", "1",             # mono
        "-ar", str(TARGET_SR),  # resample to 16kHz
        "-loglevel", "error",
        "pipe:1"                # stream to stdout
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on '{path}':\n{result.stderr.decode().strip()}"
        )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio


def load_file(path: str) -> tuple[np.ndarray, int]:
    """
    Load any supported audio file and resample to 16kHz mono.

    Supported: .wav  .flac  .ogg  .mp3  .mp4  .m4a  .aac  .webm
    MP4/M4A:  audio track is extracted automatically via ffmpeg.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    print(f"📂 Loading {ext} file: {p.name}")

    if ext in FFMPEG_FORMATS:
        # ffmpeg path: clean decode, no warnings, handles mp4/mp3/m4a/aac
        audio = _load_via_ffmpeg(path)
    else:
        # soundfile path: fast native decode for wav/flac/ogg
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)

    duration = len(audio) / TARGET_SR
    print(f"✅ Loaded {duration:.2f}s of audio at {TARGET_SR}Hz")
    return audio, TARGET_SR
