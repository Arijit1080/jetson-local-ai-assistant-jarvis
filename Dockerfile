# ── Jetson Local AI Assistant — JARVIS ──────────────────────────
# Runtime image for the chatbot on a Jetson Orin Nano Super.
# Based on NVIDIA L4T JetPack 6.2 (CUDA 12.6, cuDNN 9.3, Python 3.10).

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive

# ─── System deps ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        portaudio19-dev libsndfile1 ffmpeg \
        cmake ninja-build build-essential \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── Python env ─────────────────────────────────────────────────
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip wheel

# Pin numpy<2 first — tflite-runtime 2.14 (used by openWakeWord) needs it.
RUN pip install 'numpy<2'

# onnxruntime-gpu (TensorRT + CUDA EPs) from NVIDIA Jetson AI Lab
RUN pip install \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        onnxruntime-gpu

# Build CTranslate2 from source with CUDA + cuDNN — the jetson-ai-lab.io
# wheel is CPU-only at the time of writing, and PyPI's aarch64 wheel is
# CPU-only too. Building yourself ensures faster-whisper can use the GPU.
# sm_87 = Orin Ampere (Orin Nano / Orin NX).
RUN cd /tmp && \
    git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && mkdir build && cd build && \
    cmake .. -G Ninja \
      -DWITH_CUDA=ON -DWITH_CUDNN=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP && \
    ninja -j3 && ninja install && ldconfig && \
    cd ../python && \
    pip install -r install_requirements.txt && \
    pip install . && \
    cd / && rm -rf /tmp/CTranslate2

# Rest of the Python deps (faster-whisper has to come with --no-deps so it
# doesn't try to re-pull the CPU ctranslate2 wheel and overwrite our GPU one)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-deps faster-whisper==1.2.1 \
 && pip install -r /tmp/requirements.txt \
 && pip install huggingface-hub tokenizers av tqdm cffi

# Pre-download openWakeWord model files (~6 MB). They are NOT bundled with
# the pip package; openWakeWord normally downloads them on first init.
# Doing it here means the hotword loop works the moment the container starts.
RUN python -c "import openwakeword.utils as u; u.download_models()"

# ─── App ────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

RUN chmod +x /app/scripts/*.sh

# Models persisted via volume mount (see docker-compose.yml).
ENV HF_HOME=/app/models/hf \
    XDG_CACHE_HOME=/app/models/cache

EXPOSE 8080

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["chatbot_server.py", "--port", "8080"]
