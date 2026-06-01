import re
from typing import Callable, Iterable, Optional

import numpy as np
import sounddevice as sd
from piper import PiperVoice

import config

_SENT_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")


class Speaker:
    def __init__(self, on_envelope: Optional[Callable[[list[float], float], None]] = None):
        """on_envelope(env, duration_s) is called just before playback of each
        synthesised sentence — used to drive UI lip-sync animations."""
        self.voice = PiperVoice.load(str(config.PIPER_VOICE))
        self.sr = self.voice.config.sample_rate           # usually 22050 for Amy medium
        self.on_envelope = on_envelope

    def _synth(self, text: str) -> np.ndarray:
        chunks = []
        for chunk in self.voice.synthesize(text):
            chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)

    def _emit_envelope(self, audio: np.ndarray):
        if self.on_envelope is None or audio.size == 0:
            return
        try:
            win = max(1, int(self.sr * 0.02))    # 20 ms windows
            nf = len(audio) // win
            if nf <= 0:
                return
            af = audio[:nf * win].astype(np.float32).reshape(nf, win)
            rms = np.sqrt((af * af).mean(axis=1)) / 32768.0
            env = np.clip(rms * 6.0, 0.0, 1.0)   # boost for visual range
            duration = len(audio) / self.sr
            self.on_envelope([round(float(v), 3) for v in env], float(duration))
        except Exception as e:
            print(f"[tts] envelope hook failed: {e}")

    def say(self, text: str):
        text = text.strip()
        if not text:
            return
        audio = self._synth(text)
        self._emit_envelope(audio)
        sd.play(audio, samplerate=self.sr, device=config.OUTPUT_DEVICE)
        sd.wait()

    def stream(self, token_iter: Iterable[str]):
        """Buffer tokens until a full sentence, then synth+play. Sequential."""
        buf = ""
        for tok in token_iter:
            buf += tok
            parts = _SENT_END.split(buf)
            if len(parts) > 1:
                *complete, buf = parts
                for sent in complete:
                    self.say(sent)
        if buf.strip():
            self.say(buf)
