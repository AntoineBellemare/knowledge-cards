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
    "- CRITICAL FOR 'citations_or_quotes' arrays:\n"
    "  * EVERY quote MUST include a paper reference in parentheses at the end.\n"
    "  * Format: \"Quote text here\" (Author et al., Year) or \"Quote text\" (Paper Title)\n"
    "  * Example: \"Consciousness may be substrate-independent\" (Tononi & Koch, 2015)\n"
    "  * Select only a few (max ~5 per section) representative snippets from across schemas.\n"
    "  * If you cannot safely attribute a quote to a specific paper, do not include it.\n"
    "- DO NOT introduce new top-level sections unless they are clearly needed to capture "
    "  cross-paper patterns. Prefer to reuse and fill the existing structure.\n"
    "- Be concise but informative. The meta-card should be usable as a high-level overview "
    "  of the entire set of papers.\n"
    "- DO NOT add any prose outside of the JSON. Return valid JSON only."
)

# Speculation system prompt for generating novel insights from synthesized knowledge
SPECULATION_SYSTEM_PROMPT = (
    "You are a creative research synthesizer and speculative theorist.\n"
    "Your task is to generate NOVEL SPECULATIVE INSIGHTS by combining and extending ideas "
    "from multiple research papers. You think like a visionary researcher who sees connections "
    "others miss and proposes bold but grounded hypotheses.\n\n"
    "GUIDING PRINCIPLES:\n"
    "- Look for UNEXPECTED CONNECTIONS between papers that address different aspects of the question.\n"
    "- Identify TENSIONS or CONTRADICTIONS between papers as fertile ground for new thinking.\n"
    "- Generate hypotheses that are NOVEL but still GROUNDED in the evidence across papers.\n"
    "- Think across DISCIPLINARY BOUNDARIES - what would a philosopher, artist, or scientist from another field notice?\n"
    "- Consider EMERGENT PROPERTIES - what new understanding arises from the combination that isn't in any single paper?\n"
    "- Be BOLD but HONEST about the speculative nature of your insights.\n\n"
    "CITATION REQUIREMENTS:\n"
    "- ALWAYS reference which papers your insights draw from using (Author, Year) or (Paper Title) format.\n"
    "- When combining ideas from multiple papers, cite all relevant sources.\n"
    "- Example: \"Building on X's notion of embodied cognition (Smith, 2020) and Y's network theory (Jones, 2018)...\"\n\n"
    "OUTPUT FORMAT:\n"
    "Return a JSON object with the 'speculative_synthesis' structure as specified.\n"
    "Each insight should be substantive (2-4 sentences) and reference which papers/ideas it draws from.\n"
    "DO NOT just summarize - SYNTHESIZE and EXTEND beyond what the papers explicitly state."
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
    last_response_text: Optional[str] = None
    for attempt in range(3):
        try:
            resp = m.generate_content(user_prompt)
            last_response_text = resp.text
            return json.loads(resp.text)
        except (ResourceExhausted, DeadlineExceeded, GoogleAPIError, Exception) as e:
            last_err = e
            print(f"[WARN] Gemini attempt {attempt + 1} failed: {e}")
    
    # Save failed response for debugging
    if last_response_text:
        from pathlib import Path
        import os
        # Use /app/results in production or ../results locally
        results_dir = Path(os.getenv("RESULTS_DIR", "/app/results"))
        if not results_dir.exists():
            results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        error_file = results_dir / "metacard_error_response.json"
        print(f"[ERROR] All attempts failed. Saving raw response to {error_file}")
        print(f"[ERROR] Response preview (first 500 chars): {last_response_text[:500]}")
        try:
            error_file.write_text(last_response_text, encoding="utf-8")
            print(f"[ERROR] Full response saved to {error_file}")
            print(f"[ERROR] You can download this file from the results directory")
        except Exception as save_err:
            print(f"[ERROR] Could not save error response: {save_err}")
    
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
    
    # Extract paper references for the prompt
    paper_refs = []
    for schema in schemas:
        metadata = schema.get("metadata", {})
        citation = schema.get("citation", {})
        title = metadata.get("title") or citation.get("title") or "Unknown"
        authors = metadata.get("authors") or citation.get("authors") or []
        year = metadata.get("year") or citation.get("year") or ""
        if isinstance(authors, list) and authors:
            # Handle case where authors[0] might be a dict or a string
            first_author_raw = authors[0]
            if isinstance(first_author_raw, str):
                first_author = first_author_raw.split(",")[0].split()[-1]
            elif isinstance(first_author_raw, dict):
                # If it's a dict, try to extract name field
                first_author = first_author_raw.get("name", "Unknown")
            else:
                first_author = "Unknown"
            paper_refs.append(f"- {title} ({first_author}, {year})" if year else f"- {title} ({first_author})")
        else:
            paper_refs.append(f"- {title}")
    
    papers_list = "\n".join(paper_refs) if paper_refs else "Multiple papers"

    prompt = (
        "You are given multiple completed reading card SCHEMAS in JSON format. "
        "Each schema corresponds to one paper and shares the same overall structure.\n\n"
        f"PAPERS IN THIS CORPUS:\n{papers_list}\n\n"
        "SCHEMAS JSON:\n"
        "```json\n"
        f"{schemas_str}\n"
        "```\n\n"
        "IMPORTANT: For ALL 'citations_or_quotes' fields, every quote MUST include a paper reference "
        "in parentheses, e.g., \"Quote text\" (Author, Year). Use the paper list above for references.\n\n"
        "Please synthesise these into a SINGLE 'meta-card' JSON as described in the system prompt."
    )
    return prompt


# Schema for the speculative synthesis section
SPECULATION_SCHEMA = {
    "speculative_synthesis": {
        "emergent_hypotheses": {
            "cross_paper_connections": [
                "Novel connection 1: [describe insight linking 2+ papers]",
                "Novel connection 2: [describe another unexpected link]"
            ],
            "novel_theoretical_bridges": [
                "Bridge 1: [theoretical framework that unifies disparate findings]"
            ],
            "synthetic_propositions": [
                "Proposition 1: [bold claim emerging from combined evidence]"
            ]
        },
        "generative_questions": {
            "questions_arising_from_gaps": [
                "What remains unexplored at the intersection of X and Y?"
            ],
            "questions_from_tensions": [
                "How can we reconcile the apparent contradiction between A and B?"
            ],
            "interdisciplinary_provocations": [
                "What would a [different field] perspective reveal about these findings?"
            ]
        },
        "speculative_predictions": {
            "testable_hypotheses": [
                "If X is true across papers, we should observe Y in context Z"
            ],
            "methodological_innovations": [
                "Combining methods from papers A and B could enable..."
            ],
            "potential_paradigm_shifts": [
                "These findings collectively suggest we may need to rethink..."
            ]
        },
        "creative_extensions": {
            "metaphorical_insights": [
                "The pattern across papers is like... [rich metaphor]"
            ],
            "cross_domain_applications": [
                "These principles could transform how we approach..."
            ],
            "experiential_or_artistic_implications": [
                "For lived experience, this suggests..."
            ]
        },
        "synthesis_narrative": "A 3-5 sentence integrative narrative that weaves together the most provocative speculative insights, creating a coherent vision of what these papers collectively point toward that none of them individually articulates."
    }
}


def build_speculation_prompt(
    meta_card: Dict[str, Any], 
    schemas: List[Dict[str, Any]], 
    question: Optional[str] = None
) -> str:
    """
    Build a prompt for generating speculative insights from the meta-card and original schemas.
    
    Args:
        meta_card: The synthesized meta-card
        schemas: The original individual paper schemas
        question: The original research question that guided the analysis (optional but valuable)
    """
    # Extract key information from meta-card for focused speculation
    meta_summary = json.dumps(meta_card, ensure_ascii=False, indent=2)
    
    # Get paper titles/authors for reference
    paper_refs = []
    for schema in schemas:
        metadata = schema.get("metadata", {})
        citation = schema.get("citation", {})
        title = metadata.get("title") or citation.get("title") or "Unknown"
        authors = metadata.get("authors") or citation.get("authors") or []
        if isinstance(authors, list):
            authors = ", ".join(authors[:2]) + ("..." if len(authors) > 2 else "")
        paper_refs.append(f"- {title} ({authors})")
    
    papers_list = "\n".join(paper_refs)
    
    question_section = ""
    if question:
        question_section = f"""
ORIGINAL RESEARCH QUESTION:
\"\"\"{question}\"\"\"

Use this question as a lens for your speculation. What novel insights emerge specifically 
in relation to this question that no single paper addresses?
"""

    prompt = f"""
You are generating SPECULATIVE INSIGHTS from a synthesis of multiple research papers.

{question_section}

PAPERS ANALYZED:
{papers_list}

META-SYNTHESIS (combined findings across all papers):
```json
{meta_summary}
```

YOUR TASK:
Generate a 'speculative_synthesis' section that goes BEYOND what any individual paper says.
Look for:
1. EMERGENT PATTERNS that only become visible when papers are combined
2. TENSIONS between papers that could spark new research directions  
3. GAPS where the collective evidence points but no paper goes
4. UNEXPECTED CONNECTIONS between concepts from different papers
5. BOLD HYPOTHESES grounded in the combined evidence

OUTPUT SCHEMA:
```json
{json.dumps(SPECULATION_SCHEMA, ensure_ascii=False, indent=2)}
```

IMPORTANT GUIDELINES:
- Reference specific papers or findings when making connections
- Be genuinely speculative but ground insights in the evidence
- Prioritize novelty - don't just summarize what's already in the meta-card
- Think like a creative researcher seeing the big picture
- For the synthesis_narrative, write in an engaging, thought-provoking style
- Generate at least 3 items per list field

Return ONLY the JSON object for 'speculative_synthesis'.
"""
    return prompt


def generate_speculation(
    meta_card: Dict[str, Any],
    schemas: List[Dict[str, Any]],
    question: Optional[str] = None,
    model: str = "gemini-2.5-flash-lite",
) -> Dict[str, Any]:
    """
    Generate speculative insights from the meta-card and original schemas.
    
    Args:
        meta_card: The synthesized meta-card
        schemas: The original individual paper schemas  
        question: The original research question (optional)
        model: Gemini model to use
        
    Returns:
        The speculative_synthesis section as a dict
    """
    prompt = build_speculation_prompt(meta_card, schemas, question)
    speculation = call_gemini_json(model, SPECULATION_SYSTEM_PROMPT, prompt)
    
    # Ensure we have the expected structure
    if "speculative_synthesis" in speculation:
        return speculation["speculative_synthesis"]
    return speculation


def generate_meta_card(
    schemas: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash-lite",
    question: Optional[str] = None,
    include_speculation: bool = True,
) -> Dict[str, Any]:
    """
    High-level function that generates a meta-card dict from a list of schemas.
    
    Args:
        schemas: List of individual paper schemas
        model: Gemini model to use
        question: The original research question (used for speculation)
        include_speculation: Whether to generate speculative insights (default True)
        
    Returns:
        Complete meta-card with optional speculative_synthesis section
    """
    init_gemini()
    prompt = build_meta_prompt(schemas)
    meta_card = call_gemini_json(model, META_SYSTEM_PROMPT, prompt)

    if not isinstance(meta_card, dict):
        raise ValueError("Gemini did not return a JSON object for the meta-card.")

    # Generate speculative insights if requested
    if include_speculation:
        try:
            print("[INFO] Generating speculative synthesis...")
            speculation = generate_speculation(meta_card, schemas, question, model)
            meta_card["speculative_synthesis"] = speculation
            print("[INFO] Speculative synthesis added to meta-card.")
        except Exception as e:
            print(f"[WARN] Could not generate speculation: {e}")
            # Add empty speculation section on failure
            meta_card["speculative_synthesis"] = {
                "error": str(e),
                "emergent_hypotheses": {},
                "generative_questions": {},
                "speculative_predictions": {},
                "creative_extensions": {},
                "synthesis_narrative": ""
            }

    return meta_card


def add_speculation_to_existing_meta(
    meta_json_path: Path,
    cards_jsonl_path: Path,
    question: Optional[str] = None,
    model: str = "gemini-2.5-flash-lite",
) -> Dict[str, Any]:
    """
    Add speculative synthesis to an existing meta-card.
    
    Useful for regenerating speculation without reprocessing all papers.
    
    Args:
        meta_json_path: Path to existing meta-card JSON
        cards_jsonl_path: Path to the original cards JSONL
        question: Research question (if not embedded in cards)
        model: Gemini model to use
        
    Returns:
        Updated meta-card with speculative_synthesis
    """
    init_gemini()
    
    # Load existing meta-card
    with meta_json_path.open("r", encoding="utf-8") as f:
        meta_card = json.load(f)
    
    # Load original schemas
    schemas = load_schemas_from_file(cards_jsonl_path)
    
    # Try to extract question from cards if not provided
    if not question:
        for schema in schemas:
            q = schema.get("_template_question")
            if q:
                question = q
                break
    
    print(f"[INFO] Adding speculation to existing meta-card...")
    print(f"[INFO] Question: {question[:100]}..." if question else "[INFO] No question found")
    
    # Generate speculation
    speculation = generate_speculation(meta_card, schemas, question, model)
    meta_card["speculative_synthesis"] = speculation
    
    # Save updated meta-card
    with meta_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta_card, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Speculation added to {meta_json_path}")
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
        help="Path to input .json/.jsonl file containing multiple cards/schemas.",
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
    ap.add_argument(
        "--question",
        "-q",
        default=None,
        help="Research question to guide speculation (optional).",
    )
    ap.add_argument(
        "--no-speculation",
        action="store_true",
        help="Disable speculative synthesis generation.",
    )
    ap.add_argument(
        "--add-speculation",
        type=str,
        default=None,
        help="Add speculation to existing meta-card. Provide path to cards JSONL.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    # Mode: Add speculation to existing meta-card
    if args.add_speculation:
        cards_path = Path(args.add_speculation)
        add_speculation_to_existing_meta(
            in_path,  # Existing meta-card
            cards_path,  # Original cards
            question=args.question,
            model=args.model,
        )
        return

    # Mode: Generate new meta-card
    print(f"[INFO] Loading schemas from {in_path} ...")
    schemas = load_schemas_from_file(in_path)

    print("[INFO] Generating meta-card with Gemini...")
    meta_card = generate_meta_card(
        schemas, 
        model=args.model,
        question=args.question,
        include_speculation=not args.no_speculation,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta_card, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Meta-card saved to {out_path}")


if __name__ == "__main__":
    main()
