from pathlib import Path

# --- paths on the Jetson ---
PROJECT_ROOT = Path.home() / "local-assistant"
MODELS_DIR = PROJECT_ROOT / "models"
PIPER_VOICE = MODELS_DIR / "piper" / "en_GB-alan-medium.onnx"   # JARVIS-style British male
WHISPER_DIR = MODELS_DIR / "whisper"

# --- model choices ---
WHISPER_MODEL = "small.en"
WHISPER_COMPUTE = "int8_float16"   # Orin Ampere likes fp16
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:1.7b"

# --- wake word ---
WAKE_WORD = "hey_jarvis"           # one of: alexa, hey_jarvis, hey_mycroft, hey_rhasspy
WAKE_THRESHOLD = 0.25
WAKE_INPUT_GAIN = 4.0          # boost mic ~12 dB before oWW (codec runs quiet)

# --- audio ---
SAMPLE_RATE = 16000                # oWW + whisper both want 16k mono
FRAME_MS = 80                      # oWW expects 80ms frames (1280 samples @ 16k)
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

INPUT_DEVICE = "USB PnP Audio Device"   # Waveshare codec (JMTek 0c76:1203)
OUTPUT_DEVICE = "USB PnP Audio Device"  # both record + play through the codec

# --- end-of-speech detection ---
SILENCE_RMS = 300                  # tune after first runs
SILENCE_MS = 1200                  # how long of silence ends a turn
MAX_UTTERANCE_S = 15

# --- LLM behavior ---
SYSTEM_PROMPT = (
    "You are a concise voice assistant running locally on a Jetson Orin Nano. "
    "Reply in one or two short sentences. Use plain text only — no markdown, "
    "no lists, no emojis. Never explain your reasoning."
)
