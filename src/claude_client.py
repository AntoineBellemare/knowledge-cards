"""
Claude (Anthropic) backend for the JSON-producing LLM calls.

llm.call_llm_json routes any model name starting with "claude-" here, so Claude
models can be picked anywhere a Gemini model can. Each call uses the API key
the caller supplies (normally the user's own key).

Env:
  CARDS_CLAUDE_EFFORT  low | medium | high | xhigh | max   (default: medium)
"""

from __future__ import annotations

import os

import anthropic

CLAUDE_MODELS = {
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
}

CLAUDE_EFFORT = os.getenv("CARDS_CLAUDE_EFFORT", "medium")
# Room for adaptive thinking plus a full card: a merged card for a whole book can pass 16k
# output tokens. Caps this large need streaming; only tokens actually generated are billed.
CLAUDE_MAX_TOKENS = 64000

# Haiku 5.5 is billed $0.10/$0.50 per MTok for prompts up to 100k tokens and
# $0.50/$2.50 beyond, so prompts are never sent past that line.
TIERED_PROMPT_LIMITS = {"claude-haiku-5-5": 100_000}
# count_tokens is an estimate; keep a 5% margin below the price boundary.
PROMPT_LIMIT_MARGIN = 0.95
# Below this many characters a prompt can't approach 95k tokens, so skip counting.
_CHARS_ALWAYS_UNDER_LIMIT = 40_000

# Server-side refusal fallback ("default" routing) on the Claude API.
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

JSON_ONLY_INSTRUCTION = (
    "\n\nRespond with a single valid JSON object and nothing else: "
    "no Markdown code fences, no commentary before or after it."
)


class PromptTooLongError(RuntimeError):
    """The prompt would be billed at the model's long-context rate, so it was not sent."""

    def __init__(self, model: str, tokens: int, limit: int):
        self.model, self.tokens, self.limit = model, tokens, limit
        super().__init__(
            f"Prompt is {tokens:,} tokens; {model} switches to long-context pricing "
            f"above {limit:,}. Split the input instead."
        )


def is_claude_model(model: str) -> bool:
    return bool(model) and model.startswith("claude-")


def _make_client(api_key: str) -> anthropic.Anthropic:
    # Parallel chunk workers can hit rate limits; the SDK backs off on 429/5xx.
    return anthropic.Anthropic(api_key=api_key, max_retries=4)


def check_prompt_size(client: anthropic.Anthropic, model: str, system: str, user: str) -> None:
    """Raise PromptTooLongError if the prompt would cross the model's pricing tier."""
    limit = TIERED_PROMPT_LIMITS.get(model)
    if limit is None or len(system) + len(user) < _CHARS_ALWAYS_UNDER_LIMIT:
        return
    tokens = client.messages.count_tokens(
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
    ).input_tokens
    if tokens > limit * PROMPT_LIMIT_MARGIN:
        raise PromptTooLongError(model, tokens, limit)


def call_claude_text(model: str, system_msg: str, user_msg: str, api_key: str) -> str:
    """Send one system+user prompt to Claude and return the response text (expected to be JSON)."""
    client = _make_client(api_key)
    system = system_msg + JSON_ONLY_INSTRUCTION
    check_prompt_size(client, model, system, user_msg)

    params = dict(
        model=model,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        output_config={"effort": CLAUDE_EFFORT},
    )
    if model in _FALLBACK_MODELS:
        stream = client.beta.messages.stream(**params, betas=[_FALLBACK_BETA], fallbacks="default")
    else:
        stream = client.messages.stream(**params)
    with stream as s:
        resp = s.get_final_message()

    print(f"[Claude] {resp.model}: {resp.usage.input_tokens} in / {resp.usage.output_tokens} out tokens")
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{model} declined the request ({getattr(resp, 'stop_details', None)})")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"{model} hit max_tokens={CLAUDE_MAX_TOKENS} before finishing the JSON")
    return "".join(block.text for block in resp.content if block.type == "text")
