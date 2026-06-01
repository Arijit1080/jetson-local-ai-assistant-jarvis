import json
from typing import Iterator

import httpx

import config


class LLM:
    def __init__(self):
        # OLLAMA_URL env var overrides config (useful in Docker)
        import os
        url = os.environ.get("OLLAMA_URL", config.OLLAMA_URL)
        self.client = httpx.Client(base_url=url, timeout=60.0)
        self.history: list[dict] = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]
        # preload model into VRAM
        self.client.post("/api/generate", json={
            "model": config.OLLAMA_MODEL, "prompt": "", "keep_alive": "30m"
        })

    def stream_reply(self, user_text: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": user_text})
        assistant_text = ""
        with self.client.stream("POST", "/api/chat", json={
            "model": config.OLLAMA_MODEL,
            "messages": self.history,
            "stream": True,
            "think": False,
            "options": {"temperature": 0.4, "num_predict": 120},
            "keep_alive": "30m",
        }) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                tok = chunk.get("message", {}).get("content", "")
                if tok:
                    assistant_text += tok
                    yield tok
                if chunk.get("done"):
                    break
        self.history.append({"role": "assistant", "content": assistant_text})
