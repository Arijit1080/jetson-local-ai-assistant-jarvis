"""Record an utterance from the mic.

Two modes:
- `record_until_stop(event)` — blocks until the threading.Event is set.
  Used by the push-to-talk web UI.
- `record_utterance()` — uses Silero VAD to end on silence. Used by the
  always-listening assistant.py loop. (May misbehave on heavily AGC'd codecs.)
"""
import threading
import time

import numpy as np
import sounddevice as sd

import config
from vad import SileroVAD, CHUNK


WEBRTC_FRAME = 480     # 30 ms at 16 kHz (WebRTC VAD wants 10/20/30 ms)
WEBRTC_FRAME_MS = 30


def record_until_stop(stop_event: threading.Event,
                       max_seconds: float = 60.0,
                       silence_ms: int = 0,
                       vad_aggressiveness: int = 2,
                       min_speech_ms: int = 300,
                       save_path: str | None = None) -> np.ndarray:
    """Records mic input until either:
      - stop_event.set() is called (manual stop), OR
      - silence_ms of non-speech is detected by WebRTC VAD (auto-stop),
        provided at least min_speech_ms of speech was first observed.
    Pass silence_ms=0 to disable the VAD auto-stop (manual only).
    """
    use_vad = silence_ms > 0
    vad = None
    if use_vad:
        try:
            import webrtcvad
            vad = webrtcvad.Vad(vad_aggressiveness)
        except ImportError:
            print("[rec] webrtcvad not available, falling back to manual-only stop")
            use_vad = False

    silent_frames_needed = silence_ms // WEBRTC_FRAME_MS if use_vad else 0
    frames: list[np.ndarray] = []
    silent_streak = 0
    speech_ms_seen = 0
    started = time.time()

    label = "manual-only" if not use_vad else f"VAD a={vad_aggressiveness}, end={silence_ms}ms"
    print(f"[rec] streaming ({label}, max {max_seconds:.0f}s)...")

    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=WEBRTC_FRAME,
                        device=config.INPUT_DEVICE) as stream:
        while not stop_event.is_set():
            block, _ = stream.read(WEBRTC_FRAME)
            chunk = block[:, 0]
            frames.append(chunk.copy())

            if use_vad:
                is_speech = vad.is_speech(chunk.astype(np.int16).tobytes(),
                                            config.SAMPLE_RATE)
                if is_speech:
                    speech_ms_seen += WEBRTC_FRAME_MS
                    silent_streak = 0
                else:
                    if speech_ms_seen >= min_speech_ms:
                        silent_streak += 1
                if silent_streak >= silent_frames_needed:
                    print(f"[rec] VAD silence stop ({speech_ms_seen} ms speech)")
                    break

            if time.time() - started > max_seconds:
                print("[rec] max length hit")
                break

    if not frames:
        return np.zeros(0, dtype=np.float32)
    full_i16 = np.concatenate(frames)
    rms = float(np.sqrt(np.mean(full_i16.astype(np.float32) ** 2)))
    peak = int(np.abs(full_i16).max())
    dur = len(full_i16) / config.SAMPLE_RATE
    print(f"[rec] captured {dur:.2f}s  rms={rms:.0f}  peak={peak}/32767  "
          f"speech={speech_ms_seen}ms")
    if save_path:
        try:
            import soundfile as sf
            sf.write(save_path, full_i16, config.SAMPLE_RATE, subtype="PCM_16")
        except Exception as e:
            print(f"[rec] wav save failed: {e}")
    return full_i16.astype(np.float32) / 32768.0


def record_utterance(min_speech_ms: int = 300) -> np.ndarray:
    """Block until the user finishes speaking (VAD silence > SILENCE_MS).

    min_speech_ms: ignore tail-silence checks until at least this much speech
    has been observed — keeps us from giving up before the user starts.
    """
    vad = SileroVAD(sample_rate=config.SAMPLE_RATE, threshold=0.3)
    frame_ms = 1000 * CHUNK // config.SAMPLE_RATE   # 32 ms at 16 kHz
    silent_frames_needed = config.SILENCE_MS // frame_ms

    frames: list[np.ndarray] = []
    probs: list[float] = []
    silent_streak = 0
    speech_ms_seen = 0
    started = time.time()

    print(f"[rec] listening (silero VAD t={vad.threshold}, end after {config.SILENCE_MS} ms silence)...")
    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=CHUNK,
                        device=config.INPUT_DEVICE) as stream:
        while True:
            block, _ = stream.read(CHUNK)
            chunk = block[:, 0]
            frames.append(chunk.copy())
            p = vad.speech_prob(chunk)
            probs.append(p)
            if p >= vad.threshold:
                speech_ms_seen += frame_ms
                silent_streak = 0
            else:
                if speech_ms_seen >= min_speech_ms:
                    silent_streak += 1
            if silent_streak >= silent_frames_needed:
                break
            if time.time() - started > config.MAX_UTTERANCE_S:
                print("[rec] max length hit"); break
    if probs:
        ar = np.array(probs)
        print(f"[rec] VAD probs: min={ar.min():.2f} median={np.median(ar):.2f} "
              f"p90={np.percentile(ar,90):.2f} max={ar.max():.2f}  "
              f"frames>=t: {int((ar >= vad.threshold).sum())}/{len(ar)}")

    # ---- diagnostics: dump capture stats + a wav so we can inspect ----
    full_i16 = np.concatenate(frames)
    rms = float(np.sqrt(np.mean(full_i16.astype(np.float32) ** 2)))
    peak = int(np.abs(full_i16).max())
    print(f"[rec] audio: rms={rms:.0f}  peak={peak}/32767  "
          f"({peak/327.67:.1f}% FS)  shape={full_i16.shape}  dtype={full_i16.dtype}")
    try:
        import soundfile as _sf
        _sf.write("/tmp/last_capture.wav", full_i16, config.SAMPLE_RATE,
                  subtype="PCM_16")
        print("[rec] saved /tmp/last_capture.wav")
    except Exception as _e:
        print(f"[rec] wav save failed: {_e}")

    audio = np.concatenate(frames).astype(np.float32) / 32768.0
    dur = len(audio) / config.SAMPLE_RATE
    print(f"[rec] captured {dur:.2f}s  ({speech_ms_seen} ms speech)")
    return audio
