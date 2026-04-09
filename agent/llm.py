"""SER — LLM Client for Gemini 3 Pro via local proxy."""

import json
import logging
import time
import urllib.request
import urllib.error
from typing import Optional

from config import LLM_PROXY_URL, LLM_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around the Gemini 3 Pro proxy."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: str = LLM_PROXY_URL,
        api_key: str = LLM_API_KEY,
        max_retries: int = 8,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        stop_sequences: Optional[list[str]] = None,
    ) -> str:
        """Send a chat request and return the text response.

        `messages` follow the Claude-style format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        We convert them to Gemini's `contents` format:
            [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", ...}]
        """
        # Convert messages
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            content = msg["content"]
            if isinstance(content, list):
                # Multi-part (e.g. images) — take text parts only for simplicity
                text = " ".join(
                    part["text"] for part in content if isinstance(part, dict) and "text" in part
                )
            else:
                text = str(content)
            contents.append({"role": role, "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "thinkingConfig": {
                    "thinkingLevel": "HIGH",
                    "includeThoughts": False,
                },
            },
        }

        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self.base_url}/models/{self.model}:generateContent"
        data = json.dumps(payload).encode()

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url, data=data)
                req.add_header("Content-Type", "application/json")
                req.add_header("api-key", self.api_key)

                with urllib.request.urlopen(req, timeout=350) as resp:
                    result = json.loads(resp.read().decode())

                candidates = result.get("candidates", [])
                if not candidates:
                    raise ValueError(f"Empty candidates in response: {result}")

                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if "text" in p)

                # Token usage (best-effort)
                usage = result.get("usageMetadata", {})
                self.total_input_tokens += usage.get("promptTokenCount", 0)
                self.total_output_tokens += usage.get("candidatesTokenCount", 0)

                return text

            except urllib.error.HTTPError as e:
                body = e.read().decode() if e.fp else str(e)
                logger.warning(f"HTTP {e.code} (attempt {attempt+1}): {body[:200]}")
                if e.code in (429, 500, 502, 503):
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except (urllib.error.URLError, OSError) as e:
                wait = min(2 ** attempt * 5, 60)
                logger.warning(f"Connection error (attempt {attempt+1}): {e}. Retry in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(10)
                else:
                    raise

        raise RuntimeError(f"LLM request failed after {self.max_retries} retries")

    def get_usage(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
