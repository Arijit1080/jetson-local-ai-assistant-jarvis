"""Sparklers Local AI Assistant — push-to-talk web UI with hotword + live SSE.

Flow:
  /events          persistent SSE stream of {phase, partial, token, done, hotword_score, ...}
  /talk/start      kicks off a turn (optionally with VAD auto-stop)
  /talk/stop       manual stop signal
  /talk/wait       block until the current turn finishes (returns final result)
  /hotword/start   enable always-listening wake word
  /hotword/stop    disable
  /settings        GET/POST for system prompt + cooldown/threshold/etc.
"""
import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from flask import Flask, Response, jsonify, render_template, request

import config
from llm import LLM
from record import record_until_stop
from stt import Transcriber
from tts import Speaker
from system_stats import TegrastatsMonitor


# ─── persistent settings ──────────────────────────────────────────────────────
# Allow Docker to mount /app/data for persistent settings
_data_dir = Path("/app/data") if Path("/app/data").is_dir() else config.PROJECT_ROOT
SETTINGS_PATH = _data_dir / "settings.json"
DEFAULT_SETTINGS = {
    "system_prompt": config.SYSTEM_PROMPT,
    "match_threshold": config.MATCH_THRESHOLD if hasattr(config, "MATCH_THRESHOLD") else 0.4,
    "wake_threshold": config.WAKE_THRESHOLD,
    "wake_gain": config.WAKE_INPUT_GAIN,
    "silence_ms": 1200,
    "vad_aggressiveness": 2,
}


def _load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
            return {**DEFAULT_SETTINGS, **saved}
        except Exception as e:
            print(f"[settings] load failed, using defaults: {e}")
    return dict(DEFAULT_SETTINGS)


def _save_settings(s: dict):
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))


SETTINGS = _load_settings()


# ─── model init ───────────────────────────────────────────────────────────────
print("[init] loading models...")
_t0 = time.time()
STT = Transcriber()
LLM_ = LLM()
LLM_.history = [{"role": "system", "content": SETTINGS["system_prompt"]}]
TTS = Speaker(on_envelope=lambda env, dur: emit("tts_play",
                                                  envelope=env, duration=dur))
STATS = TegrastatsMonitor(interval_ms=1500)
print(f"[init] ready in {time.time()-_t0:.1f}s")


# ─── shared state ─────────────────────────────────────────────────────────────
HISTORY: list[dict] = []
TURN_LOCK = threading.Lock()
TURN = {
    "phase": "idle",
    "stop_event": None,
    "thread": None,
    "result": None,
    "t_start": 0.0,
    "silence_ms": 0,
}
HOTWORD = {
    "enabled": False,
    "thread": None,
    "stop": threading.Event(),
    "last_trigger": 0.0,
    "last_score": 0.0,
    "current_score": 0.0,
    "peak_recent": 0.0,
}


# ─── SSE pub/sub ──────────────────────────────────────────────────────────────
SSE_SUBS: list[queue.Queue] = []
SSE_LOCK = threading.Lock()


def emit(evt_type: str, **payload):
    e = {"type": evt_type, "ts": round(time.time(), 3), **payload}
    with SSE_LOCK:
        for q in SSE_SUBS:
            try:
                q.put_nowait(e)
            except queue.Full:
                pass


def _set_phase(p: str):
    TURN["phase"] = p
    emit("phase", phase=p)


# ─── audio helpers ────────────────────────────────────────────────────────────
def _beep(freq: int = 880, ms: int = 150, vol: float = 0.3, sr: int = 22050):
    n = int(sr * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False)
    a = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = int(sr * 0.01)
    a[:fade] *= np.linspace(0, 1, fade)
    a[-fade:] *= np.linspace(1, 0, fade)
    try:
        sd.play(a, samplerate=sr, device=config.OUTPUT_DEVICE); sd.wait()
    except Exception as e:
        print(f"[beep] {e}")


# ─── core turn ────────────────────────────────────────────────────────────────
def _run_turn(stop_event: threading.Event):
    t0 = TURN["t_start"]
    try:
        _set_phase("recording")
        audio = record_until_stop(stop_event, max_seconds=60,
                                   silence_ms=TURN.get("silence_ms", 0),
                                   vad_aggressiveness=SETTINGS.get("vad_aggressiveness", 2),
                                   save_path="/tmp/last_capture.wav")
        t_rec = time.time() - t0
        if audio.size == 0:
            TURN["result"] = {"ok": False, "error": "no audio captured"}
            emit("done", ok=False, error="no audio captured")
            return

        _set_phase("transcribing")
        user_text = STT.transcribe(audio).strip()
        t_stt = time.time() - t0 - t_rec
        print(f"[user] {user_text!r}")
        emit("user", text=user_text)
        if not user_text or len(user_text) < 2:
            TURN["result"] = {"ok": False, "user": user_text,
                              "error": "no speech detected"}
            emit("done", ok=False, error="no speech detected", user=user_text)
            return

        _set_phase("thinking")
        chunks: list[str] = []
        def _collect():
            for tok in LLM_.stream_reply(user_text):
                chunks.append(tok)
                emit("token", text=tok)
                yield tok
        # Stream tokens to UI + feed TTS sentence-by-sentence
        _set_phase("speaking")
        TTS.stream(_collect())
        reply = "".join(chunks).strip()
        t_llm_tts = time.time() - t0 - t_rec - t_stt

        entry = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "user": user_text, "reply": reply,
            "ms_rec": int(t_rec * 1000),
            "ms_stt": int(t_stt * 1000),
            "ms_llm_tts": int(t_llm_tts * 1000),
            "ms_total": int((time.time() - t0) * 1000),
        }
        HISTORY.insert(0, entry)
        TURN["result"] = {"ok": True, **entry}
        emit("done", ok=True, **entry)
    except Exception as e:
        TURN["result"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        emit("done", ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        _set_phase("idle")


def _start_turn(silence_ms: int):
    with TURN_LOCK:
        if TURN["thread"] and TURN["thread"].is_alive():
            return False
        TURN["stop_event"] = threading.Event()
        TURN["result"] = None
        TURN["t_start"] = time.time()
        TURN["silence_ms"] = silence_ms
        _set_phase("recording")
        TURN["thread"] = threading.Thread(target=_run_turn,
                                          args=(TURN["stop_event"],),
                                          daemon=True)
        TURN["thread"].start()
    return True


# ─── hotword loop ─────────────────────────────────────────────────────────────
def _hotword_loop():
    try:
        from openwakeword.model import Model as OWWModel
    except ImportError as e:
        print(f"[hotword] disabled: {e}")
        return
    oww = OWWModel(wakeword_models=[config.WAKE_WORD],
                   inference_framework="tflite")
    print(f"[hotword] listening for '{config.WAKE_WORD}' "
          f"(threshold {SETTINGS['wake_threshold']})...")

    while not HOTWORD["stop"].is_set():
        if TURN["thread"] and TURN["thread"].is_alive():
            time.sleep(0.3); continue

        _set_phase("listening")
        oww.reset()
        fired = 0.0
        try:
            with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                dtype="int16", blocksize=config.FRAME_SAMPLES,
                                device=config.INPUT_DEVICE) as stream:
                while not HOTWORD["stop"].is_set():
                    if TURN["thread"] and TURN["thread"].is_alive():
                        break
                    block, _ = stream.read(config.FRAME_SAMPLES)
                    frame = block[:, 0]
                    gain = SETTINGS.get("wake_gain", config.WAKE_INPUT_GAIN)
                    if gain != 1.0:
                        x = (frame.astype(np.int32) * gain).clip(-32768, 32767)
                        frame = x.astype(np.int16)
                    s = float(oww.predict(frame).get(config.WAKE_WORD, 0.0))
                    HOTWORD["current_score"] = s
                    HOTWORD["peak_recent"] = max(s, HOTWORD["peak_recent"] * 0.93)
                    emit("wake_score", score=round(s, 3))
                    if s > SETTINGS["wake_threshold"]:
                        fired = s
                        HOTWORD["last_trigger"] = time.time()
                        HOTWORD["last_score"] = s
                        print(f"[hotword] triggered ({s:.2f})")
                        emit("wake_fired", score=round(s, 3))
                        break
        except Exception as e:
            print(f"[hotword] stream error: {e}")
            time.sleep(1.0); continue

        _set_phase("idle")
        if fired:
            _beep()
            time.sleep(0.1)
            _start_turn(silence_ms=SETTINGS.get("silence_ms", 1200))
            th = TURN.get("thread")
            if th: th.join()
    print("[hotword] stopped")


# ─── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html", history=HISTORY[:30],
                            backend_info={
                                "whisper": config.WHISPER_MODEL,
                                "llm": config.OLLAMA_MODEL,
                                "input": config.INPUT_DEVICE,
                                "output": config.OUTPUT_DEVICE,
                            })


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    global SETTINGS
    if request.method == "POST":
        sp = request.form.get("system_prompt", "").strip()
        if sp:
            SETTINGS["system_prompt"] = sp
            # apply immediately by resetting LLM history with new prompt
            LLM_.history = [{"role": "system", "content": sp}]
        for k in ("wake_threshold", "wake_gain"):
            v = request.form.get(k)
            if v:
                try: SETTINGS[k] = float(v)
                except ValueError: pass
        for k in ("silence_ms", "vad_aggressiveness"):
            v = request.form.get(k)
            if v:
                try: SETTINGS[k] = int(v)
                except ValueError: pass
        _save_settings(SETTINGS)
        return jsonify({"ok": True, **SETTINGS})
    return render_template("settings.html", settings=SETTINGS,
                           defaults=DEFAULT_SETTINGS)


@app.route("/events")
def events():
    q: queue.Queue = queue.Queue(maxsize=200)
    with SSE_LOCK:
        SSE_SUBS.append(q)

    def gen():
        try:
            # initial state burst
            yield f"data: {json.dumps({'type':'hello','phase':TURN['phase']})}\n\n"
            while True:
                try:
                    e = q.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"   # keepalive comment
                    continue
                yield f"data: {json.dumps(e)}\n\n"
        except GeneratorExit:
            pass
        finally:
            with SSE_LOCK:
                if q in SSE_SUBS:
                    SSE_SUBS.remove(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/talk/start", methods=["POST"])
def talk_start():
    auto = request.args.get("auto") == "1" or request.values.get("auto") == "1"
    silence_ms = int(request.values.get("silence_ms", SETTINGS["silence_ms"]))
    if not _start_turn(silence_ms if auto else 0):
        return jsonify({"ok": False, "error": "already running",
                        "phase": TURN["phase"]}), 409
    return jsonify({"ok": True, "auto": auto, "silence_ms": TURN["silence_ms"]})


@app.route("/talk/stop", methods=["POST"])
def talk_stop():
    ev = TURN.get("stop_event"); th = TURN.get("thread")
    if not ev or not th:
        return jsonify({"ok": False, "error": "not running"}), 409
    ev.set()
    th.join(timeout=60)
    return jsonify(TURN.get("result") or {"ok": False, "error": "no result"})


@app.route("/talk/wait", methods=["POST"])
def talk_wait():
    th = TURN.get("thread")
    if not th:
        return jsonify({"ok": False, "error": "not running"}), 409
    th.join(timeout=120)
    return jsonify(TURN.get("result") or {"ok": False, "error": "no result"})


@app.route("/status")
def status():
    return jsonify({
        "phase": TURN.get("phase", "idle"),
        "hotword_enabled": HOTWORD["enabled"],
        "hotword_threshold": SETTINGS["wake_threshold"],
        "hotword_current": round(HOTWORD["current_score"], 3),
        "hotword_peak": round(HOTWORD["peak_recent"], 3),
    })


@app.route("/api/stats")
def api_stats():
    return jsonify(STATS.get())


@app.route("/hotword/start", methods=["POST"])
def hotword_start():
    if HOTWORD["enabled"]:
        return jsonify({"ok": True, "already": True})
    HOTWORD["stop"].clear()
    HOTWORD["enabled"] = True
    HOTWORD["thread"] = threading.Thread(target=_hotword_loop, daemon=True)
    HOTWORD["thread"].start()
    return jsonify({"ok": True})


@app.route("/hotword/stop", methods=["POST"])
def hotword_stop():
    HOTWORD["stop"].set()
    HOTWORD["enabled"] = False
    th = HOTWORD.get("thread")
    if th: th.join(timeout=5)
    _set_phase("idle")
    return jsonify({"ok": True})


@app.route("/history")
def history():
    return jsonify(HISTORY[:30])


@app.route("/history/clear", methods=["POST"])
def history_clear():
    HISTORY.clear()
    # also reset LLM conversation history (keep system prompt)
    LLM_.history = [{"role": "system", "content": SETTINGS["system_prompt"]}]
    return jsonify({"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"\nserving at http://0.0.0.0:{args.port}/  (system prompt: "
          f"{SETTINGS['system_prompt'][:60]!r}…)\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
