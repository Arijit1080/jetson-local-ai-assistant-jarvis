"""Capture 'Hey Jarvis' positive samples through the codec.

Each iteration: 3-second countdown -> HIGH beep -> 1.6s recording -> LOW beep.
Saves WAVs to ~/local-assistant/training/positives/hey_jarvis_NNN.wav.

Tips while recording:
- Vary delivery: normal, fast, slow, soft, loud, mumbled, clear, slightly
  rushed, slightly drawn out. Move your head a bit. Speak from different
  distances (close, normal, slightly off-axis).
- Stop early with Ctrl-C.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

import config


SAMPLES_DIR = Path.home() / "local-assistant" / "training" / "positives"
SR = 16000
RECORD_S = 1.6           # 1.6 s window per sample


def beep(freq: int, ms: int = 120, vol: float = 0.3, sr: int = 22050):
    n = int(sr * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False)
    a = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = int(sr * 0.01)
    a[:fade]  *= np.linspace(0, 1, fade)
    a[-fade:] *= np.linspace(1, 0, fade)
    sd.play(a, samplerate=sr, device=config.OUTPUT_DEVICE); sd.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=30,
                    help="number of samples to record this session (default 30)")
    args = ap.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(SAMPLES_DIR.glob("hey_jarvis_*.wav"))
    start_idx = len(existing) + 1
    print(f"Already have {len(existing)} sample(s). Recording {args.count} more "
          f"(starting at #{start_idx}).")
    print(f"Vary your delivery between takes. Saving to {SAMPLES_DIR}.\n")

    saved = 0
    try:
        for i in range(start_idx, start_idx + args.count):
            for c in range(3, 0, -1):
                print(f"  #{i:>3}/{start_idx+args.count-1}: in {c}...", flush=True)
                time.sleep(1)
            beep(880, 120)
            audio = sd.rec(int(RECORD_S * SR), samplerate=SR, channels=1,
                           dtype="int16", device=config.INPUT_DEVICE)
            sd.wait()
            beep(440, 120)
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            peak = int(np.abs(audio).max())
            path = SAMPLES_DIR / f"hey_jarvis_{i:03d}.wav"
            sf.write(str(path), audio[:, 0], SR, subtype="PCM_16")
            print(f"    saved {path.name}  rms={rms:.0f}  peak={peak}/32767  "
                  f"({peak/327.67:.0f}% FS)")
            saved += 1
            time.sleep(0.6)
    except KeyboardInterrupt:
        print("\nStopped early.")

    total = len(list(SAMPLES_DIR.glob("hey_jarvis_*.wav")))
    print(f"\nDone. Saved {saved} this session.  Total samples in folder: {total}.")


if __name__ == "__main__":
    main()
