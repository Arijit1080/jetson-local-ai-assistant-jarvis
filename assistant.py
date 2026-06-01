"""Main loop: wake -> beep -> record -> STT -> LLM -> TTS, repeat."""
import sys
import time

import numpy as np
import sounddevice as sd

import config
from llm import LLM
from record import record_utterance
from stt import Transcriber
from tts import Speaker
from wake import WakeListener


def beep(freq: int = 880, ms: int = 150, vol: float = 0.3, sr: int = 22050):
    n = int(sr * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False)
    a = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = int(sr * 0.01)
    a[:fade]  *= np.linspace(0, 1, fade)
    a[-fade:] *= np.linspace(1, 0, fade)
    sd.play(a, samplerate=sr, device=config.OUTPUT_DEVICE)
    sd.wait()


def main():
    print("[init] loading models...")
    t0 = time.time()
    stt = Transcriber()
    llm = LLM()
    tts = Speaker()
    wake = WakeListener()
    print(f"[init] ready in {time.time()-t0:.1f}s")

    while True:
        try:
            wake.listen_once()
        except KeyboardInterrupt:
            print("\nbye"); sys.exit(0)

        beep()                  # "go ahead, I'm listening"
        time.sleep(0.1)         # let the codec settle so beep tail doesn't bleed in
        audio = record_utterance()
        text = stt.transcribe(audio)
        print(f"[user] {text!r}")
        if not text or len(text) < 2:
            continue

        print("[llm] streaming...")
        tts.stream(llm.stream_reply(text))


if __name__ == "__main__":
    main()
