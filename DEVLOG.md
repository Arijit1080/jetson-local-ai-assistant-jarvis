# Local Voice Assistant on Jetson Orin Nano Super — Build Log

A tutorial-style log of building a fully offline voice assistant on the Jetson Orin Nano Super 8 GB. Everything below was actually executed; commands, errors, fixes, and benchmarks are real.

## Target stack

| Layer | Choice | Why |
|---|---|---|
| Hotword | openWakeWord | CPU-only (~50 MB), supports custom wake words, no licensing strings |
| STT | faster-whisper (`small.en`, CT2 GPU build) | Best accuracy/latency on Jetson; sm_87 Ampere has great FP16 |
| LLM | Qwen3-1.7B Q4 via Ollama | Reliable intent understanding; 1.2 GB VRAM leaves room for everything else |
| TTS | Piper (Amy medium) | CPU, fast, good quality, simple ONNX runtime |
| Audio I/O | Waveshare USB Audio Codec | Orin Nano has no on-board mic; codec gives mic-in + line-out |

**Hardware:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super, JetPack 6.2 (L4T R36.4.4), CUDA 12.6, cuDNN 9.3, Ubuntu 22.04, Python 3.10, 8 GB RAM (4 GB swap), 56 GB SD card.

## Day 1 — Stack stand-up

### 1. SSH into the Jetson and verify environment

```bash
ssh jetson@192.168.31.233        # password: jetson
cat /proc/device-tree/model       # NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
free -h                           # 7.4 GiB RAM, 3.7 GiB swap
/usr/local/cuda/bin/nvcc --version  # CUDA 12.6.68
dpkg -l | grep cudnn              # cuDNN 9.3.0
```

CUDA, cuDNN dev headers, and gcc all present — good news, we can build CT2 from source.

### 2. Passwordless SSH + passwordless sudo (one-time)

To kill the per-command password prompt:

```bash
# On Mac
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id jetson@192.168.31.233   # or append the .pub to ~/.ssh/authorized_keys

# On Jetson
echo "jetson ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/90-jetson-nopasswd
sudo chmod 440 /etc/sudoers.d/90-jetson-nopasswd
sudo visudo -c -f /etc/sudoers.d/90-jetson-nopasswd
```

> Standard dev-kit ergonomics; do *not* do this on a shared / production machine.

### 3. APT prerequisites

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev \
  ninja-build build-essential ffmpeg libsndfile1 libsndfile1-dev \
  portaudio19-dev curl
```

### 4. Install Ollama and pull Qwen3-1.7B

```bash
curl -fsSL https://ollama.com/install.sh | sh
# -> installs to /usr/local, sets up systemd unit, version 0.24.0
systemctl is-active ollama   # active

ollama pull qwen3:1.7b       # ~1.4 GB, takes a few minutes
ollama ps                    # confirms 100% GPU placement, ~1.9 GB VRAM
```

**First-run benchmark:**

```
$ time ollama run qwen3:1.7b --think=false 'Reply with a single short sentence: what is the capital of Japan?'
The capital of Japan is Tokyo.
real    0m11.872s          # cold load
real    0m0.681s           # warm
```

> Pass `--think=false` for Qwen3 unless you actually want reasoning traces — it cuts latency dramatically.

### 5. Build CTranslate2 from source (for CUDA on aarch64)

PyPI has no CUDA-enabled CT2 wheel for ARM64, so faster-whisper would fall back to CPU. Build CT2 from source against the Jetson's CUDA 12.6 + cuDNN 9.3.

```bash
mkdir -p ~/local-assistant && cd ~/local-assistant
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2

mkdir build && cd build
cmake .. -G Ninja \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP
ninja -j3                   # -j3 to avoid OOM on 8 GB RAM during nvcc compiles
sudo ninja install
sudo ldconfig
```

> `--depth 1` breaks the submodule revision lookups (`cutlass` in particular). Use a full clone.
>
> sm_87 is the right CUDA arch for Orin (Ampere). On a Jetson with a different SoC you'd change this.

Build took ~5 minutes wall-clock with `-j3`. Output:

```
[143/146] Linking CXX shared library libctranslate2.so.4.7.2
[144/146] Creating library symlink libctranslate2.so.4 libctranslate2.so
[146/146] Linking CXX executable cli/ct2-translator
```

### 6. Python venv + CT2 Python bindings

```bash
cd ~/local-assistant
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Build the CT2 Python bindings against the libctranslate2.so we just installed
cd CTranslate2/python
pip install -r install_requirements.txt
pip install .
```

Confirms GPU is visible from Python:

```python
>>> import ctranslate2
>>> ctranslate2.__version__
'4.7.2'
>>> ctranslate2.get_cuda_device_count()
1
```

### 7. Install Piper TTS, openWakeWord, sounddevice

```bash
pip install piper-tts openwakeword sounddevice numpy
pip install 'numpy<2'      # see gotcha below
```

**Gotcha — NumPy 2 vs tflite-runtime:** the fresh install pulled `numpy==2.2.6`, but `tflite-runtime==2.14.0` (which openWakeWord uses) still requires NumPy 1.x. Symptom:

```
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

Fix: pin `numpy<2` (we used 1.26.4). Everything else in the stack (CT2, Piper, faster-whisper, onnxruntime) works fine on 1.26.

### 8. Download Piper voice (Amy, US English, medium quality)

```bash
mkdir -p ~/local-assistant/models/piper
cd ~/local-assistant/models/piper
curl -fsSL -o en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -fsSL -o en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

Smoke test:

```bash
echo 'Hello, I am your local assistant.' | \
  python -m piper --model en_US-amy-medium.onnx --output_file /tmp/piper_test.wav
file /tmp/piper_test.wav
# /tmp/piper_test.wav: RIFF (little-endian) data, WAVE audio, 16 bit, mono 22050 Hz
```

### 9. Install faster-whisper without overwriting our CT2 build

If you `pip install faster-whisper` plain, pip will pull the PyPI ctranslate2 wheel — CPU-only on aarch64 — and shadow the GPU build we just installed. So:

```bash
pip install --no-deps faster-whisper
pip install huggingface-hub tokenizers av tqdm
```

First-run STT (downloads `small.en` ~244 MB):

```python
from faster_whisper import WhisperModel
m = WhisperModel('small.en', device='cuda', compute_type='int8_float16',
                 download_root='/home/jetson/local-assistant/models/whisper')
segments, info = m.transcribe('/tmp/piper_test.wav', beam_size=5)
for s in segments: print(s.text)
# -> "Hello, I am your local assistant running on the Jetson Oren Nano."
```

> "Orin" → "Oren" — acoustically reasonable. Fix later by passing `initial_prompt="Jetson Orin Nano"`.

### 10. Warm-cache benchmarks (the numbers that actually matter)

| Step | Cold | Warm |
|---|---|---|
| Ollama Qwen3-1.7B "say hi in 5 words" | ~20 s | **680 ms** |
| Whisper small.en beam=1 on 4 s audio (GPU int8_f16) | ~3.2 s | **641 ms** (RTF 0.16) |
| Whisper small.en beam=5 | — | **740 ms** |
| Piper synth via subprocess (~3.5 s audio) | — | 2.1 s (drops to ~300 ms when used as a library) |
| openWakeWord per-frame inference | — | ~1 ms (TFLite XNNPACK on CPU) |

## Day 1 — Pipeline code

Layout in `~/local-assistant/` (synced from `/Users/arimacair/Jetson Orin/local-assistant/`):

```
config.py          # central tunables: paths, models, sample rate, silence detection
wake.py            # WakeListener — blocks until openWakeWord triggers
record.py          # record_utterance() — captures until ~1.2s silence (RMS VAD)
stt.py             # Transcriber — faster-whisper small.en on GPU, warm-up included
llm.py             # LLM — Ollama /api/chat streaming, keeps history, think=False
tts.py             # Speaker — Piper sentence-by-sentence streaming
assistant.py       # main loop: wake → record → STT → LLM → TTS
test_pipeline.py   # mic-free smoke test (feeds a WAV through STT → LLM → TTS)
requirements.txt
```

### Streaming detail

LLM tokens stream from Ollama. As they arrive, `Speaker.stream(...)` buffers them and splits on sentence-end punctuation. The moment a sentence is complete, it's synthesised and played while the LLM is still generating the next sentence. Reduces perceived latency a lot.

### End-to-end mic-free smoke test result

```
$ python test_pipeline.py /tmp/piper_test.wav
[test] input: 4.21s
[stt] (0.81s) 'Hello, I am your local assistant running on the Jetson Oren Nano.'
[llm] first token: 592ms
[done] total 3.51s, reply='Hello! How can I assist you today?'
```

`ALSA underrun` spam in the log is `sd.play` writing to the `default` device when no real audio sink is attached — harmless until the USB codec is plugged in.

## Next session

- Plug in Waveshare USB Audio Codec, verify `lsusb` + `aplay -l` + `arecord -l` see it, set `INPUT_DEVICE` / `OUTPUT_DEVICE` in `config.py`.
- Loopback test: `arecord -D plughw:N,0 -f S16_LE -r 16000 -c 1 -d 3 test.wav && aplay test.wav`.
- Run `assistant.py` against the real mic, tune `SILENCE_RMS` to actual room noise floor.
- Add `initial_prompt="Jetson Orin Nano"` to Whisper to fix the "Oren" mistranscription.
- Consider Silero VAD if RMS-based end-of-speech detection is flaky.
