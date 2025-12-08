#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a META-CARD from multiple completed reading cards (schemas)
stored in a single .json file.

Designed for inputs like your example, where the file contains multiple
JSON objects for different papers, e.g.:

1) A full "card" object with a schema:
   {
     "id": "...",
     "name": "...",
     "question": "...",
     "createdAt": "...",
     "schema": { ...full schema... },
     "_file": "...",
     "citation": {...}
   }

2) A bare schema object:
   {
     "metadata": { ... },
     "orientation_and_scope": { ... },
     ...
   }

The file can be:
- A standard JSON value (single dict or list), OR
- Multiple JSON objects separated by newlines (JSONL-style).

For each object:
- If it has a dict under "schema", we use that.
- Otherwise, we treat the object itself as the schema.

The result is a single "meta-card" JSON that mirrors the schema structure
(e.g. metadata, orientation_and_scope, biological_systems_analyzed, etc.),
but summarises across all the input schemas.

Requires:
- GOOGLE_API_KEY set in your environment or in a .env file.

Usage example:
    python generate_meta_card_from_cards.py \
        --input cards/networks_cards.json \
        --output cards/networks_meta_card.json \
        --model gemini-2.5-flash-lite
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, DeadlineExceeded


# ---------------- Gemini setup ----------------

META_SYSTEM_PROMPT = (
    "You are a meta-synthesizer for research reading cards.\n"
    "You are given MULTIPLE completed reading card SCHEMAS as JSON objects. Each schema "
    "shares the same overall structure (keys / nested fields) and corresponds to one paper or text.\n\n"
    "YOUR TASK\n"
    "- Produce a SINGLE 'meta-card' JSON object that summarizes across ALL the schemas.\n"
    "- The meta-card should follow the SAME OVERALL SCHEMA as the individual schemas "
    "  (same main sections and nested structure), but:\n"
    "  * Each field should contain a cross-paper synthesis rather than a copy from a single schema.\n"
    "  * Where appropriate, merge, cluster, and summarise overlapping ideas.\n"
    "  * Highlight majority patterns vs. minority/unique positions when possible.\n\n"
    "DETAILED REQUIREMENTS\n"
    "- Output MUST be a SINGLE JSON object (no lists at the top level, no extra text).\n"
    "- Preserve all major top-level sections (e.g. metadata, orientation_and_scope, "
    "  biological_systems_analyzed, network_dynamics_facets, drivers_of_change, "
    "  methods_and_metrics, findings_and_comparisons, limitations_and_future_directions, etc.), "
    "  if they exist across schemas.\n"
    "- Within each section:\n"
    "  * Merge textual fields (strings, lists of strings) into concise cross-paper summaries.\n"
    "  * If multiple schemas list similar constructs or measures, group them and indicate which "
    "    are widely used vs. rare or unique.\n"
    "  * If numeric or boolean fields appear, summarise the trend qualitatively "
    "    (e.g. 'most papers', 'a minority of schemas').\n"
    "- For the metadata section:\n"
    "  * Include a brief overview of the corpus in a field such as 'papers_covered' or similar, "
    "    listing title, authors, year, venue when available.\n"
    "- For 'citations_or_quotes' arrays (which may appear in many subsections):\n"
    "  * Prefer to keep them concise by selecting only a few (max ~5 per section) representative "
    "    snippets from across schemas.\n"
    "  * If you cannot safely select, you may leave them as empty arrays [].\n"
    "- DO NOT introduce new top-level sections unless they are clearly needed to capture "
    "  cross-paper patterns. Prefer to reuse and fill the existing structure.\n"
    "- Be concise but informative. The meta-card should be usable as a high-level overview "
    "  of the entire set of papers.\n"
    "- DO NOT add any prose outside of the JSON. Return valid JSON only."
)


def init_gemini() -> None:
    """Configure Gemini from GOOGLE_API_KEY (env or .env)."""
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    genai.configure(api_key=key)


def call_gemini_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Call Gemini to get JSON output, with a few retries on transient errors.
    """
    m = genai.GenerativeModel(
        model,
        system_instruction=system_prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = m.generate_content(user_prompt)
            return json.loads(resp.text)
        except (ResourceExhausted, DeadlineExceeded, GoogleAPIError, Exception) as e:
            last_err = e
            print(f"[WARN] Gemini attempt {attempt + 1} failed: {e}")
    raise last_err  # type: ignore[arg-type]


# ---------------- Data helpers ----------------

def _extract_schema_from_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a raw JSON object from the file, extract the schema-like payload.

    - If obj has a 'schema' key that is a dict, return obj['schema'].
      (this matches your first example with id/name/question/schema/_file/citation)
    - Otherwise, assume obj itself is already a schema
      (this matches your second example with Hutchins where metadata is top-level).
    """
    schema = obj.get("schema")
    if isinstance(schema, dict):
        return schema
    return obj


def load_schemas_from_file(path: Path) -> List[Dict[str, Any]]:
    """
    Load all schema objects from a single .json file.

    The file can be:
    - A single JSON value:
        * a dict (one card or one schema)
        * a list of card/schema objects
        * a dict with a 'cards' list
    - OR multiple JSON objects separated by newlines (JSON Lines style).

    Returns a list of schema dicts (one per paper).
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file {path} is empty.")

    raw_objects: List[Dict[str, Any]] = []

    # Try to parse as a single JSON value first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            # list of cards/schemas
            for el in data:
                if isinstance(el, dict):
                    raw_objects.append(el)
                else:
                    raise ValueError("List elements must be JSON objects (dicts).")
        elif isinstance(data, dict):
            # Either:
            # - one card/schema
            # - or a wrapper with "cards": [...]
            if "cards" in data and isinstance(data["cards"], list):
                for el in data["cards"]:
                    if isinstance(el, dict):
                        raw_objects.append(el)
                    else:
                        raise ValueError("Elements under 'cards' must be JSON objects (dicts).")
            else:
                raw_objects.append(data)
        else:
            raise ValueError("Top-level JSON must be an object or a list.")
    except json.JSONDecodeError:
        # Fallback: assume JSONL / multiple objects per file (like your pasted example)
        raw_objects = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("Each JSON line must be an object (dict).")
            raw_objects.append(obj)

    if not raw_objects:
        raise ValueError("No JSON objects found in the input file.")

    schemas = [_extract_schema_from_object(obj) for obj in raw_objects]

    # Simple sanity check: each schema should at least have 'metadata' or something similar
    for i, sc in enumerate(schemas):
        if not isinstance(sc, dict):
            raise ValueError(f"Schema #{i} is not a dict.")
    print(f"[INFO] Loaded {len(schemas)} schemas from {path}")
    return schemas


def build_meta_prompt(schemas: List[Dict[str, Any]]) -> str:
    """
    Build the user prompt that passes all schemas to Gemini.
    """
    schemas_str = json.dumps(schemas, ensure_ascii=False, indent=2)

    prompt = (
        "You are given multiple completed reading card SCHEMAS in JSON format. "
        "Each schema corresponds to one paper and shares the same overall structure.\n\n"
        "SCHEMAS JSON:\n"
        "```json\n"
        f"{schemas_str}\n"
        "```\n\n"
        "Please synthesise these into a SINGLE 'meta-card' JSON as described in the system prompt."
    )
    return prompt


def generate_meta_card(
    schemas: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash-lite",
) -> Dict[str, Any]:
    """
    High-level function that generates a meta-card dict from a list of schemas.
    """
    init_gemini()
    prompt = build_meta_prompt(schemas)
    meta_card = call_gemini_json(model, META_SYSTEM_PROMPT, prompt)

    if not isinstance(meta_card, dict):
        raise ValueError("Gemini did not return a JSON object for the meta-card.")

    return meta_card


# ---------------- CLI ----------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a META-CARD by summarising across multiple reading card schemas."
    )
    ap.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input .json file containing multiple cards/schemas (like your card.json).",
    )
    ap.add_argument(
        "--output",
        "-o",
        default="meta_card.json",
        help="Path for the output meta-card JSON file.",
    )
    ap.add_argument(
        "--model",
        "-m",
        default="gemini-2.5-flash-lite",
        help="Gemini model name (default: gemini-2.5-flash-lite).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"[INFO] Loading schemas from {in_path} ...")
    schemas = load_schemas_from_file(in_path)

    print("[INFO] Generating meta-card with Gemini...")
    meta_card = generate_meta_card(schemas, model=args.model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta_card, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Meta-card saved to {out_path}")


if __name__ == "__main__":
    main()
