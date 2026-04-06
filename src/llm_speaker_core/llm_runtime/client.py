from __future__ import annotations

import json
from dataclasses import dataclass

import httpx


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout_s: float

    def generate(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 180
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,
            },
        }

        timeout = httpx.Timeout(
            connect=30.0, read=self.timeout_s, write=30.0, pool=30.0
        )
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(
                f"Invalid ollama response: {json.dumps(data, ensure_ascii=False)}"
            )
        return content.strip()

