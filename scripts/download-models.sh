#!/usr/bin/env bash
# Downloads the Piper voice the chatbot expects at the configured path.
# Idempotent — skips files that already exist.
set -euo pipefail

VOICE_DIR="${1:-$(dirname "$0")/../models/piper}"
mkdir -p "$VOICE_DIR"

# en_GB-alan-medium: British male, JARVIS-y. ~61 MB onnx + ~5 KB config.
URL_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium"

for f in en_GB-alan-medium.onnx en_GB-alan-medium.onnx.json; do
    if [[ -f "$VOICE_DIR/$f" ]]; then
        echo "[models] already present: $f"
    else
        echo "[models] downloading $f -> $VOICE_DIR/"
        curl -fsSL -o "$VOICE_DIR/$f" "$URL_BASE/$f"
    fi
done

echo "[models] Piper voice ready in $VOICE_DIR"
echo "[models] Whisper / Silero VAD / openWakeWord models will be downloaded on first use."
