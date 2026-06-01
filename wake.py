import queue

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel

import config


class WakeListener:
    def __init__(self):
        self.oww = OWWModel(
            wakeword_models=[config.WAKE_WORD] if config.WAKE_WORD else None,
            inference_framework="tflite",
        )
        self.q: queue.Queue[np.ndarray] = queue.Queue()

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            print(f"[wake] audio status: {status}")
        self.q.put(indata[:, 0].copy())

    def listen_once(self) -> None:
        """Block until wake word fires."""
        self.oww.reset()
        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=config.FRAME_SAMPLES,
            device=config.INPUT_DEVICE,
            callback=self._audio_cb,
        )
        with stream:
            print(f"[wake] listening for '{config.WAKE_WORD}' (threshold {config.WAKE_THRESHOLD})...")
            peak = 0.0
            while True:
                frame = self.q.get()
                scores = self.oww.predict(frame)
                s = float(scores.get(config.WAKE_WORD, 0.0))
                if s > 0.10:               # debug: any non-trivial score
                    print(f"[wake]   score={s:.2f}")
                if s > peak: peak = s
                if s > config.WAKE_THRESHOLD:
                    print(f"[wake] triggered ({s:.2f})  peak-seen={peak:.2f}")
                    return
