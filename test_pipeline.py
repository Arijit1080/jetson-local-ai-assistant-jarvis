"""Mic-free smoke test: feeds a WAV file through STT -> LLM -> TTS."""
import sys
import time

import numpy as np
import soundfile as sf

from llm import LLM
from stt import Transcriber
from tts import Speaker


def load_wav_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        # crude resample by linear interpolation; ok for smoke test
        ratio = 16000 / sr
        idx = np.arange(0, len(audio), 1 / ratio).astype(np.int64)
        idx = idx[idx < len(audio)]
        audio = audio[idx]
    return audio.astype(np.float32)


def main(wav_path: str):
    stt = Transcriber()
    llm = LLM()
    tts = Speaker()

    audio = load_wav_16k(wav_path)
    print(f"[test] input: {len(audio)/16000:.2f}s")

    t0 = time.time()
    text = stt.transcribe(audio)
    print(f"[stt] ({time.time()-t0:.2f}s) {text!r}")

    if not text:
        print("[test] empty transcript, abort")
        return

    print("[llm] streaming reply -> tts")
    t0 = time.time()
    first_tok_t = None
    tokens = []

    def tap():
        nonlocal first_tok_t
        for tok in llm.stream_reply(text):
            if first_tok_t is None:
                first_tok_t = time.time() - t0
                print(f"[llm] first token: {first_tok_t*1000:.0f}ms")
            tokens.append(tok)
            yield tok

    tts.stream(tap())
    print(f"[done] total {time.time()-t0:.2f}s, reply={''.join(tokens)!r}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/piper_test.wav")
