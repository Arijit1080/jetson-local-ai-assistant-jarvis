#!/usr/bin/env bash
# Entrypoint for the Docker image. Ensures the Piper voice is present
# (downloads on first run) before launching the chatbot server.
set -euo pipefail

VOICE_PATH="/app/models/piper/en_GB-alan-medium.onnx"
if [[ ! -f "$VOICE_PATH" ]]; then
    echo "[entrypoint] Piper voice not found, downloading…"
    /app/scripts/download-models.sh /app/models/piper
fi

# Pass-through to the server (defaults to chatbot_server.py)
exec python "$@"
