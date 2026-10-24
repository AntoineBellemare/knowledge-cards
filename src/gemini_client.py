"""
Gemini backend for the JSON-producing LLM calls, using the google-genai SDK.

Each call builds a client from the API key the caller supplies (normally the
user's own key), so concurrent jobs never share credentials.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import errors, types

# Retried with backoff: rate limits and transient server errors.
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

# Reasoning depth per model. Hidden thinking is billed as output, and extraction
# needs little of it. Models not listed keep their default: 3.5 Flash-Lite already
# defaults to "minimal", Argon (the "best quality" choice) to "medium", and 2.5
# models don't take thinking levels.
THINKING_LEVELS = {"gemini-3.8-flash": "low"}

# Flex: half price, best-effort capacity. Requests can queue for minutes and get 429/503
# when Flex is full, with no automatic fallback, so we back off and then use Standard.
FLEX_TIMEOUT_MS = 600_000
FLEX_ATTEMPTS = 3
FLEX_BACKOFF_S = 15.0


def call_gemini_text(model: str, system_msg: str, user_msg: str, api_key: str, retries: int = 3,
                     flex: bool = False) -> str:
    """Send one system+user prompt to Gemini in JSON mode and return the response text.
    flex: try the half-price Flex tier first, falling back to Standard if it stays full."""
    level = THINKING_LEVELS.get(model)
    base = dict(
        system_instruction=system_msg,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_level=level) if level else None,
    )
    config = types.GenerateContentConfig(**base)
    http_options = types.HttpOptions(timeout=FLEX_TIMEOUT_MS) if flex else None
    with genai.Client(api_key=api_key, http_options=http_options) as client:
        if flex:
            flex_config = types.GenerateContentConfig(**base, service_tier="flex")
            for attempt in range(FLEX_ATTEMPTS):
                try:
                    resp = client.models.generate_content(model=model, contents=user_msg, config=flex_config)
                    return resp.text or ""
                except errors.APIError as e:
                    if e.code not in _RETRYABLE_CODES:
                        raise
                    print(f"[Gemini] Flex busy ({e.code}) on attempt {attempt + 1}, backing off")
                    time.sleep(FLEX_BACKOFF_S * 2 ** attempt)
            print(f"[Gemini] Flex capacity unavailable for {model}; using the standard tier")
        for attempt in range(retries):
            try:
                print(f"[Gemini] {model}: prompt {len(user_msg)} chars")
                resp = client.models.generate_content(model=model, contents=user_msg, config=config)
                return resp.text or ""
            except errors.APIError as e:
                if e.code not in _RETRYABLE_CODES or attempt == retries - 1:
                    raise
                print(f"[Gemini] {e.code} on attempt {attempt + 1}, retrying: {e.message}")
                time.sleep(2.0 * (attempt + 1))
    return ""
