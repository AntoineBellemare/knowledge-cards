"""
Provider-neutral entry point for the JSON-producing LLM calls.

Models named "claude-*" go to Anthropic, everything else to Google Gemini.
Every call uses the API key the user supplied for that provider. The server's
own keys (GOOGLE_API_KEY / ANTHROPIC_API_KEY) are only used as a fallback when
ALLOW_SERVER_API_KEYS=true, which is off by default.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from claude_client import is_claude_model, call_claude_text
from gemini_client import call_gemini_text

PROVIDERS = {
    "google": {"env": "GOOGLE_API_KEY", "label": "Google AI (Gemini)"},
    "anthropic": {"env": "ANTHROPIC_API_KEY", "label": "Anthropic (Claude)"},
}

ApiKeys = Optional[Dict[str, Optional[str]]]


class MissingApiKeyError(RuntimeError):
    def __init__(self, providers: List[str]):
        self.providers = providers
        labels = ", ".join(PROVIDERS[p]["label"] for p in providers)
        super().__init__(f"Missing API key for {labels}. Add your key under 'API keys'.")


def provider_for(model: str) -> str:
    return "anthropic" if is_claude_model(model) else "google"


def server_keys_allowed() -> bool:
    return os.getenv("ALLOW_SERVER_API_KEYS", "").strip().lower() in {"1", "true", "yes"}


def server_key_available(provider: str) -> bool:
    return server_keys_allowed() and bool(os.getenv(PROVIDERS[provider]["env"]))


def env_api_keys() -> Dict[str, Optional[str]]:
    """The operator's own keys from the environment, for the command-line scripts."""
    return {p: os.getenv(info["env"]) for p, info in PROVIDERS.items()}


def resolve_api_key(model: str, api_keys: ApiKeys) -> str:
    provider = provider_for(model)
    key = (api_keys or {}).get(provider)
    if key:
        return key
    if server_key_available(provider):
        return os.environ[PROVIDERS[provider]["env"]]
    raise MissingApiKeyError([provider])


def missing_providers(models: Iterable[Optional[str]], api_keys: ApiKeys) -> List[str]:
    """Providers needed by these models that have neither a user key nor an allowed server key."""
    needed = {provider_for(m) for m in models if m}
    return sorted(
        p for p in needed
        if not (api_keys or {}).get(p) and not server_key_available(p)
    )


def parse_json_text(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def _save_malformed(text: str) -> None:
    results_dir = Path(__file__).resolve().parent.parent / "results"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"llm_malformed_{int(time.time())}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"[LLM] Malformed response saved to {path}")
    except OSError as e:
        print(f"[LLM] Could not save malformed response: {e}")


def call_llm_json(model: str, system_msg: str, user_msg: str, api_keys: ApiKeys = None,
                  retries: int = 3, flex: bool = False) -> Dict[str, Any]:
    """Send a system+user prompt to the model's provider and return the parsed JSON object.
    flex: use Gemini's half-price Flex tier (ignored for Claude)."""
    api_key = resolve_api_key(model, api_keys)
    text = ""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        if is_claude_model(model):
            text = call_claude_text(model, system_msg, user_msg, api_key)
        else:
            text = call_gemini_text(model, system_msg, user_msg, api_key, flex=flex)
        try:
            return parse_json_text(text)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"[LLM] JSON decode error on attempt {attempt + 1}: {e}")
    _save_malformed(text)
    raise last_err  # type: ignore[misc]
