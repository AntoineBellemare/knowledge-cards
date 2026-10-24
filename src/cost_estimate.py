"""
Rough cost estimates for a card-building job, per model, before it runs.

Input tokens come from the same PDF extraction, chunking and prompt builders the
pipeline uses, so the call plan matches what a job sends. Output tokens are
estimated from the template's size. Prices are list prices (USD per 1M tokens).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from claude_client import TIERED_PROMPT_LIMITS, PROMPT_LIMIT_MARGIN
from gemini_template_creation import (
    DEFAULT_MODE, EXTRACTION_MODES, SYSTEM_CARD, build_chunks, extract_pdf, text_size_ok,
    prompt_map_chunk, prompt_reduce, prompt_single_pass,
)
from generate_meta_card_UI import META_SYSTEM_PROMPT, SPECULATION_SYSTEM_PROMPT

PRICES_AS_OF = "2026-10-23"

# input / output: USD per 1M tokens (standard tier).
# thinking: extra hidden reasoning tokens, as a multiple of visible output, at the
#   reasoning level the app uses (billed as output; used for the high end of the range).
# changes: scheduled list-price changes as (effective date, new rates).
MODEL_PRICES: Dict[str, Dict[str, Any]] = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "thinking": 0.25},   # "minimal" reasoning
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "thinking": 0.5,          # "low" reasoning
                         "changes": [("2027-01-01", {"input": 1.50, "output": 7.50})]},
    "gemini-4-argon": {"input": 2.00, "output": 10.00, "thinking": 1.0,           # "medium" reasoning
                       "changes": [("2026-12-01", {"input": 4.00, "output": 20.00})]},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "thinking": 0.0},    # legacy, no reasoning
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "thinking": 1.0},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "thinking": 1.0},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "thinking": 1.0},
}

CHARS_PER_TOKEN = 4.0
JSON_INDENT_OVERHEAD = 1.2      # partial cards are re-sent pretty-printed (indent=2)
LOW_FACTOR, HIGH_FACTOR = 0.8, 1.3   # estimation error band around the token counts
FLEX_DISCOUNT = 0.5                 # Gemini Flex tier: half the standard price

_cache_lock = threading.Lock()
_paper_cache: Dict[Tuple[str, float, int], "PaperStats"] = {}


@dataclass
class PaperStats:
    name: str
    title: str
    sections: List[Tuple[str, str]]   # (section name, text), as the pipeline extracts them
    full_text_chars: int


@dataclass
class Call:
    role: str          # "extract" | "reduce" | "meta"
    input_tokens: float
    output_tokens: float


def _tok(chars: float) -> float:
    return chars / CHARS_PER_TOKEN


def _paper_stats(pdf: Path) -> PaperStats:
    st = pdf.stat()
    key = (str(pdf), st.st_mtime, st.st_size)
    with _cache_lock:
        if key in _paper_cache:
            return _paper_cache[key]
    with contextlib.redirect_stdout(io.StringIO()):  # extract_pdf logs every step
        title, sections = extract_pdf(pdf)
    full_text = "\n\n".join(f"[{s}]\n{t}" for s, t in sections)
    stats = PaperStats(pdf.name, title, sections, len(full_text))
    with _cache_lock:
        _paper_cache[key] = stats
    return stats


def _count_leaves(obj: Any) -> int:
    if isinstance(obj, dict) and obj:
        return sum(_count_leaves(v) for v in obj.values())
    return 1


def _reduce_calls(base_in: float, n: int, partial_out: float, card_out: float, batch_size: int) -> List[Call]:
    """Mirror build_card_for_pdf / reduce_in_batches: one merge for <=15 partials, else hierarchical."""
    if n <= 1:
        return []
    if n <= 15:
        return [Call("reduce", base_in + n * partial_out * JSON_INDENT_OVERHEAD, card_out)]
    calls: List[Call] = []
    size, item_out = n, partial_out
    while size > 1:
        total_chars = size * item_out * CHARS_PER_TOKEN
        if (size <= batch_size and total_chars < 30000) or size == 2:
            calls.append(Call("reduce", base_in + size * item_out * JSON_INDENT_OVERHEAD, card_out))
            break
        bs = 2 if size <= batch_size else batch_size
        groups = math.ceil(size / bs)
        for g in range(groups):
            in_group = min(bs, size - g * bs)
            calls.append(Call("reduce", base_in + in_group * item_out * JSON_INDENT_OVERHEAD, card_out))
        size, item_out = groups, card_out
    return calls


def plan_job(pdfs: List[Path], schema: Dict[str, Any], question: str = "",
             batch_size: int = 4, mode: str = DEFAULT_MODE) -> Dict[str, Any]:
    """Model-independent list of calls a job will make in this cost mode, with estimated token counts."""
    cfg = EXTRACTION_MODES[mode]
    schema_tokens = _tok(len(json.dumps(schema, ensure_ascii=False, indent=2)))
    leaves = _count_leaves(schema)
    # Final cards from past runs were ~1.2x (schema + 60 tokens per field), median of 12 runs.
    card_out = 1.2 * (schema_tokens + leaves * 60)
    if cfg["sparse_partials"]:
        # Only filled fields come back; bigger chunks fill more of them.
        partial_out = min(card_out, 0.3 * schema_tokens + leaves * 15 * math.sqrt(cfg["chunk_words"] / 1000))
    else:
        partial_out = schema_tokens + leaves * 15
    system_tokens = _tok(len(SYSTEM_CARD))

    calls: List[Call] = []
    single_pass_papers = 0
    for pdf in pdfs:
        p = _paper_stats(pdf)
        if text_size_ok(p.sections, cfg["single_pass_words"]):
            single_pass_papers += 1
            prompt = prompt_single_pass(schema, p.title, p.name, "", question=question)
            calls.append(Call("extract", system_tokens + _tok(len(prompt) + p.full_text_chars), card_out))
            continue
        chunks = build_chunks(p.sections, cfg["chunk_words"], cfg["overlap"], cfg["pack_sections"])
        for ch in chunks:
            prompt = prompt_map_chunk(schema, p.title, p.name, ch["section"], "", question=question,
                                      sparse=cfg["sparse_partials"])
            calls.append(Call("extract", system_tokens + _tok(len(prompt) + len(ch["text"])), partial_out))
        base_in = system_tokens + _tok(len(prompt_reduce(schema, p.title, p.name, [], question=question)))
        calls += _reduce_calls(base_in, len(chunks), partial_out, card_out, batch_size)

    n = len(pdfs)
    meta_out = schema_tokens + leaves * 80 + 1500
    meta_in = _tok(len(META_SYSTEM_PROMPT)) + 600 + n * (card_out * JSON_INDENT_OVERHEAD + 25)
    spec_in = _tok(len(SPECULATION_SYSTEM_PROMPT)) + 1200 + meta_out * JSON_INDENT_OVERHEAD + n * 25
    return {
        "papers": n,
        "mode": mode,
        "flex": cfg["flex"],
        "calls": calls,
        "meta": {"input": meta_in, "output": meta_out, "speculation_input": spec_in,
                 "speculation_output": 2000.0},
        "single_pass_papers": single_pass_papers,
    }


def _meta_calls(plan: Dict[str, Any], model: str) -> List[Call]:
    """Meta-card + speculation; Haiku-style tiered models split the meta-card to stay in tier."""
    m = plan["meta"]
    calls: List[Call] = []
    limit = TIERED_PROMPT_LIMITS.get(model)
    if limit and m["input"] > limit * PROMPT_LIMIT_MARGIN:
        groups = math.ceil(m["input"] / (limit * PROMPT_LIMIT_MARGIN * 0.9))
        calls += [Call("meta", m["input"] / groups, m["output"]) for _ in range(groups)]
        calls += [Call("meta", 2 * m["output"] * JSON_INDENT_OVERHEAD + 1000, m["output"])
                  for _ in range(groups - 1)]
    else:
        calls.append(Call("meta", m["input"], m["output"]))
    calls.append(Call("meta", m["speculation_input"], m["speculation_output"]))
    return calls


def current_rates(model: str, today: Optional[date] = None) -> Dict[str, float]:
    """The model's list rates in effect on `today` (applies scheduled price changes)."""
    today = today or date.today()
    price = MODEL_PRICES[model]
    rates = {"input": price["input"], "output": price["output"]}
    for effective, new_rates in price.get("changes", []):
        if today >= date.fromisoformat(effective):
            rates = dict(new_rates)
    return rates


def next_price_change(model: str, today: Optional[date] = None) -> Optional[str]:
    """ISO date of the model's next scheduled list-price change, if any."""
    today = today or date.today()
    upcoming = [d for d, _ in MODEL_PRICES[model].get("changes", []) if date.fromisoformat(d) > today]
    return min(upcoming) if upcoming else None


def _call_cost(model: str, call: Call, with_thinking: bool, today: date, flex: bool = False) -> float:
    rates = current_rates(model, today)
    out = call.output_tokens * (1 + MODEL_PRICES[model]["thinking"] if with_thinking else 1)
    cost = (call.input_tokens * rates["input"] + out * rates["output"]) / 1_000_000
    # Per-paper calls run on Gemini's Flex tier in flex modes (meta-card calls stay Standard)
    if flex and model.startswith("gemini-") and call.role != "meta":
        cost *= FLEX_DISCOUNT
    return cost


def price_plan(plan: Dict[str, Any], model: str, model_reduction: Optional[str] = None,
               today: Optional[date] = None) -> Dict[str, Any]:
    """Cost range for running the plan with `model` (extraction + meta) and `model_reduction`."""
    today = today or date.today()
    model_reduction = model_reduction or model
    calls = plan["calls"] + _meta_calls(plan, model)
    low = high = 0.0
    tokens_in = tokens_out = 0.0
    for c in calls:
        m = model_reduction if c.role == "reduce" else model
        low += _call_cost(m, c, with_thinking=False, today=today, flex=plan["flex"])
        high += _call_cost(m, c, with_thinking=True, today=today, flex=plan["flex"])
        tokens_in += c.input_tokens
        tokens_out += c.output_tokens
    changes = [d for d in (next_price_change(model, today), next_price_change(model_reduction, today)) if d]
    return {
        "model": model,
        "model_reduction": model_reduction,
        "low": round(low * LOW_FACTOR, 4),
        "high": round(high * HIGH_FACTOR, 4),
        "calls": len(calls),
        "input_tokens": int(tokens_in),
        "output_tokens": int(tokens_out),
        "price_change": min(changes) if changes else None,
    }


def estimate_job(pdfs: List[Path], schema: Dict[str, Any], question: str, model: str,
                 model_reduction: Optional[str], batch_size: int = 4,
                 mode: str = DEFAULT_MODE) -> Dict[str, Any]:
    plans = {m: plan_job(pdfs, schema, question, batch_size, m) for m in EXTRACTION_MODES}
    plan = plans[mode]
    selected = price_plan(plan, model, model_reduction)
    per_model = [price_plan(plan, m) for m in MODEL_PRICES]
    per_mode = [{"mode": m, **price_plan(plans[m], model, model_reduction)} for m in EXTRACTION_MODES]
    return {
        "papers": plan["papers"],
        "single_pass_papers": plan["single_pass_papers"],
        "mode": mode,
        "selected": selected,
        "per_model": per_model,
        "per_mode": per_mode,
        "prices_as_of": PRICES_AS_OF,
        "note": ("Rough estimate from list prices. The range covers hidden thinking tokens "
                 "and about ±25% error in the token counts."),
    }
