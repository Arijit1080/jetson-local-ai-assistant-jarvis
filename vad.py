"""Silero VAD wrapper — onnxruntime CPU only, no torch dep.

Robust to AGC and ambient noise where simple energy-based VAD fails.
First use downloads the ~1.8 MB ONNX model to ~/.cache/silero_vad.onnx.
"""
import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_URL = ("https://github.com/snakers4/silero-vad/raw/master/src/"
             "silero_vad/data/silero_vad.onnx")
DEFAULT_PATH = Path.home() / ".cache" / "silero_vad.onnx"
CHUNK = 512                      # samples per inference at 16 kHz (32 ms)


def _ensure_model(path: Path = DEFAULT_PATH) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[vad] downloading silero VAD -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class SileroVAD:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5,
                 input_gain: float = 8.0,
                 model_path: Path | None = None):
        """input_gain compensates for low-level codec capture before feeding the
        model. ~8x (~18 dB) suits a typical AGC USB codec where peak ends up
        ~8 % of full scale; tweak if your mic delivers different levels."""
        path = _ensure_model(model_path or DEFAULT_PATH)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=opts)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.input_gain = float(input_gain)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def speech_prob(self, chunk_int16: np.ndarray) -> float:
        """chunk_int16: shape (CHUNK,) int16 mono at sample_rate."""
        if len(chunk_int16) != CHUNK:
            raise ValueError(f"expected {CHUNK} samples, got {len(chunk_int16)}")
        x = chunk_int16.astype(np.float32) / 32768.0
        if self.input_gain != 1.0:
            x = np.clip(x * self.input_gain, -1.0, 1.0)
        out, self.state = self.session.run(
            None,
            {
                "input": x[np.newaxis, :],
                "state": self.state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        return float(out.flatten()[0])

    def is_speech(self, chunk_int16: np.ndarray) -> bool:
        return self.speech_prob(chunk_int16) >= self.threshold
