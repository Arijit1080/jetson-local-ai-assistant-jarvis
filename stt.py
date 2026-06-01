import numpy as np
from faster_whisper import WhisperModel

import config


class Transcriber:
    def __init__(self):
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device="cuda",
            compute_type=config.WHISPER_COMPUTE,
            download_root=str(config.WHISPER_DIR),
        )
        # warm-up so first real call is fast
        silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        list(self.model.transcribe(silence, beam_size=1)[0])

    def transcribe(self, audio: np.ndarray) -> str:
        # audio: float32 mono @ 16k, range [-1, 1]
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            # bias toward terms the user will say a lot
            initial_prompt="Jetson Orin Nano. Hey Jarvis.",
        )
        return " ".join(s.text.strip() for s in segments).strip()
