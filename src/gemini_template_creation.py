#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gemini -> Structured Cards over PDFs (schema from JSON)

- Input:  PDFs in ./papers
- Schema: ./schema.json (arbitrary JSON dict; you control fields)
- Output: out/cards.jsonl (one JSON per line), out/cards_summary.csv

Modes:
  - single-pass if the paper fits in the model’s context
  - map-reduce (chunk -> partial cards -> merged final card) for long papers

Env:
  GOOGLE_API_KEY (or in a .env file)
"""

from __future__ import annotations
import os, re, json, time, math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError, DeadlineExceeded

from cards_to_pdf import build_pdf  # <- use the build_pdf you showed
from generate_meta_card_UI import load_schemas_from_file, generate_meta_card



from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd

from cards_to_pdf import build_pdf

# ---- Project paths (same logic as api.py) ----
ROOT_DIR    = Path(__file__).resolve().parent.parent
CARDS_DIR   = ROOT_DIR / "cards"
PAPERS_DIR  = ROOT_DIR / "papers"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model + context settings
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")   # change if you like
MAX_INPUT_TOKENS = 10000   # was 20000 – force map-reduce for long papers
CHUNK_TOK_TARGET = 6000   # smaller chunks
CHUNK_OVERLAP = 1000       # proportionally smaller overlap


# Section heading heuristics (expand as needed)
SECTION_HINTS = [
    r"^abstract\b", r"^summary\b",
    r"^introduction\b", r"^background\b", r"^overview\b", r"^literature review\b",
    r"^theory\b", r"^framework\b", r"^related work\b",
    r"^methods?\b", r"^materials\b", r"^data collection\b", r"^analysis\b", r"^experimental design\b",
    r"^experiments?\b",
    r"^findings?\b", r"^results?\b", r"^evaluation\b", r"^outcomes?\b",
    r"^discussion\b", r"^interpretation\b", r"^limitations?\b",
    r"^conclusion(s)?\b", r"^implications\b", r"^future work\b",
    r"^appendix\b", r"^supplementary\b"
]

# ------------------ IO Helpers ------------------
def load_schema(path: Path) -> Dict[str, Any]:
    """Load the card schema dictionary you want the model to fill."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(rows: List[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def drop_refs_and_supp(text: str) -> str:
    cut = re.split(
        r'\n\s*(references|bibliography|supplementary materials?|appendix)\s*\n',
        text, flags=re.I
    )[0]
    return cut.strip()

# ------------------ PDF parsing ------------------
def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\xad", "")
    s = re.sub(r"-\n(\w)", r"\1", s)          # hyphenated line breaks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def likely_heading(line: str) -> bool:
    L = line.strip()
    if not (3 <= len(L) <= 120):
        return False
    if re.match("|".join(SECTION_HINTS), L.lower()):
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z].+", L):   # 2. Methods
        return True
    if L.isupper() and 3 <= len(L.split()) <= 8:
        return True
    return False

def split_into_sections(full_text: str) -> List[Tuple[str, str]]:
    lines = full_text.splitlines()
    sections, cur_title, buf = [], "preamble", []
    def push():
        nonlocal buf, sections, cur_title
        if buf:
            sections.append((cur_title, "\n".join(buf).strip()))
            buf = []
    for ln in lines:
        if likely_heading(ln):
            push()
            cur_title = re.sub(r"\s+", " ", ln.strip())
        else:
            buf.append(ln)
    push()

    # merge tiny sections
    merged = []
    for title, sec in sections:
        if merged and len(sec) < 300:
            pt, ptxt = merged[-1]
            merged[-1] = (pt, ptxt + f"\n[{title}]\n" + sec)
        else:
            merged.append((title, sec))
    # hybrid fallback
    if len(merged) < 3:
        return [("fulltext", full_text)]
    return merged

def extract_pdf(path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    print(f"[PDF] Opening {path.name} ...")
    doc = fitz.open(path)
    pages = []
    for i, p in enumerate(doc):
        txt = normalize_text(p.get_text("text"))
        pages.append(txt)
        if i == 0:
            # quick peek at first page length
            print(f"[PDF] Page 1 length for {path.name}: {len(txt.split())} words")
    full = "\n\n".join(pages)
    full = drop_refs_and_supp(full)
    print(f"[PDF] Total length after refs-drop for {path.name}: {len(full.split())} words")

    # crude title guess
    title = "Unknown Title"
    for line in full.splitlines()[:50]:
        L = line.strip()
        if 10 <= len(L) <= 120 and not re.search(r"(arxiv|doi|copyright|©)", L, re.I):
            title = L
            break
    print(f"[PDF] Guessed title for {path.name}: {title[:80]}")
    sections = split_into_sections(full)
    print(f"[PDF] {path.name}: split into {len(sections)} sections.")
    return title, sections


# ------------------ Chunking ------------------
def chunk_section(text: str, target_words=CHUNK_TOK_TARGET, overlap=CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + target_words)
        chunks.append(" ".join(words[i:j]))
        if j == len(words): break
        i = max(0, j - overlap)
    return chunks

def build_chunks(sections: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    chunks = []
    for sec_name, sec_txt in sections:
        for i, ch in enumerate(chunk_section(sec_txt)):
            chunks.append({
                "section": sec_name if i == 0 else f"{sec_name} (cont.)",
                "text": ch
            })
    return chunks

# ------------------ Gemini calls ------------------
def init_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set (env or .env).")
    genai.configure(api_key=api_key)

def call_gemini_json(model: str, system_msg: str, user_msg: str, retries=3) -> Dict[str, Any]:
    m = genai.GenerativeModel(
        model,
        system_instruction=system_msg,
        generation_config={"response_mime_type": "application/json"}
    )
    last_err = None
    for t in range(retries):
        try:
            # print(f"[Gemini] Call attempt {t+1}/{retries} with model={model}...")
            print(f"[Gemini] Prompt length: {len(user_msg.split())} words, {len(user_msg)} chars")
            resp = m.generate_content(user_msg)
            # print("[Gemini] Call succeeded.")
            return json.loads(resp.text)
        except (ResourceExhausted, DeadlineExceeded) as e:
            last_err = e
            print(f"[Gemini] Resource/Deadline error on attempt {t+1}: {e}")
            time.sleep(2.0 * (t + 1))
        except GoogleAPIError as e:
            last_err = e
            print(f"[Gemini] API error on attempt {t+1}: {e}")
            time.sleep(1.0 * (t + 1))
        except Exception as e:
            last_err = e
            print(f"[Gemini] Unknown error on attempt {t+1}: {e}")
            break
    print("[Gemini] All retries failed, raising last error.")
    raise last_err


SYSTEM_CARD = (
    "You are a meticulous research assistant. You extract **only** what is present in the text. "
    "Return strictly valid JSON matching the schema provided. Do not add fields.\n"
    "- NEVER copy the full references / bibliography section into any field.\n"
    "- For any field whose name includes 'citation' or 'quote', store only SHORT, RELEVANT excerpts "
    "(1–3 sentences, ≤300 characters), not entire reference entries.\n"
    "- Ignore any plain reference listings when filling quote/citation fields (e.g., lines that are just "
    "author/year/journal/DOI).\n"
    "- Be concise, factual, and add tiny location hints like [Intro], [Methods], [Results] when obvious.\n"
)


def prompt_single_pass(schema: Dict[str, Any], title: str, filename: str, fulltext: str) -> str:
    return f"""
Fill the following JSON schema exactly (valid JSON only, no comments):

SCHEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PAPER:
- approx_title: "{title}"
- file: {filename}

TEXT:
\"\"\"{fulltext}\"\"\"


RULES (VERY IMPORTANT):
- Populate every field in the schema. If unknown, use "" or [].
- Do NOT invent content.
- findings: short bullet points with concrete outcomes.
- methods: include type (e.g., behavioral/EEG/fMRI/corpus/comp.), data, N if visible, and measures.
- For ANY field whose name contains 'citation' or 'quote':
    * Use ONLY short, relevant excerpts from the body of the paper, not from the References section.
    * Each item should be 1–3 sentences (≤300 characters), e.g. a key claim or a sentence with an in-text citation.
    * Add a short location hint in square brackets when obvious, e.g. [Intro], [Methods], [Results], [Discussion].
    * Max 5 items per such field.
- NEVER copy the full references / bibliography list into the JSON.
- Ignore blocks that are clearly just reference entries (e.g., “Smith, J. (2019) … doi: …”).
- Embodiment-like facets (if present in the schema): add short bullets with location hints like [Methods]/[Results]/[Discussion] when obvious.
- JSON only.
"""

def prompt_map_chunk(schema: Dict[str, Any], title: str, filename: str, section: str, chunk_text: str) -> str:
    return f"""
You will fill the same JSON schema **partially** from this chunk **only**. If a field isn't covered here, leave it "" or [].

SCHEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PAPER:
- approx_title: "{title}"
- file: {filename}
- chunk_section: {section}

CHUNK:
\"\"\"{chunk_text}\"\"\"


RULES (VERY IMPORTANT):
- Extract ONLY from this chunk.
- Do NOT invent content.
- For any field whose name contains 'citation' or 'quote':
    * Use only short, relevant excerpts from THIS chunk (1–3 sentences, ≤300 characters).
    * Add a location hint using the chunk_section, e.g. [chunk: {section}].
    * Max 3 items per such field in a single partial card.
    * Do NOT copy long reference lists or raw bibliographic entries.
- Ignore pure reference entries (author/year/journal/DOI).
- Be concise. JSON only.
"""

def prompt_reduce(schema: Dict[str, Any], title: str, filename: str, partial_cards: List[Dict[str, Any]]) -> str:
    return f"""
Merge these PARTIAL JSON cards into a single, concise FINAL card that still conforms to the SCHEMA exactly.

SCHEMA:
{json.dumps(schema, ensure_ascii=False, indent=2)}

PAPER:
- approx_title: "{title}"
- file: {filename}

PARTIALS:
{json.dumps(partial_cards, ensure_ascii=False, indent=2)}

MERGE RULES (VERY IMPORTANT):
- Union/merge lists; deduplicate similar items.
- Prefer more specific/quantified info when conflicts.
- Keep methods coherent (type, data, N if present, measures).
- findings: short bullets (≤6) reflecting the paper’s core results.
- For any field whose name contains 'citation' or 'quote':
    * Merge and deduplicate quotes/citations from partial cards.
    * Drop entries that look like plain reference lines or overly long bibliographic text.
    * Keep at most 5 of the most informative items per field, each ≤300 characters.
- If a field remains unknown overall, keep "" or [].
- Return FINAL JSON only.
"""

# ------------------ Pipeline ------------------
def text_size_ok(sections: List[Tuple[str, str]]) -> bool:
    # crude: use words count as proxy for tokens
    words = sum(len(t.split()) for _, t in sections)
    return words <= MAX_INPUT_TOKENS

def build_card_for_pdf(pdf_path: Path, schema: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n[PIPELINE] ==== Processing {pdf_path.name} ====")
    title, sections = extract_pdf(pdf_path)
    total_words = sum(len(t.split()) for _, t in sections)
    print(f"[PIPELINE] {pdf_path.name}: {total_words} words across {len(sections)} sections.")

    if text_size_ok(sections):
        print(f"[PIPELINE] {pdf_path.name}: using SINGLE-PASS mode.")
        full_text = "\n\n".join(f"[{s}]\n{t}" for s, t in sections)
        user = prompt_single_pass(schema, title, pdf_path.name, full_text)
        data = call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
        data["_file"] = pdf_path.name
        if "citation" in data and isinstance(data["citation"], dict):
            data["citation"].setdefault("title", title)
        else:
            data["citation"] = {"title": title}
        print(f"[PIPELINE] {pdf_path.name}: card built in single-pass.")
        return data

    # map-reduce
    print(f"[PIPELINE] {pdf_path.name}: using MAP-REDUCE mode.")
    chunks = build_chunks(sections)
    print(f"[PIPELINE] {pdf_path.name}: built {len(chunks)} chunks.")
    partials = []
    for idx, ch in enumerate(chunks):
        print(f"[PIPELINE] {pdf_path.name}: calling Gemini on chunk {idx+1}/{len(chunks)} "
              f"({ch['section'][:40]}).")
        user = prompt_map_chunk(schema, title, pdf_path.name, ch["section"], ch["text"])
        try:
            part = call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
            partials.append(part)
        except Exception as e:
            print(f"[WARN|chunk] {pdf_path.name} chunk {idx+1}: {e}")
            continue

    if not partials:
        print(f"[PIPELINE] {pdf_path.name}: no partials, falling back to clipped single-pass.")
        clipped = " ".join("\n\n".join(t for _, t in sections).split()[:30000])
        user = prompt_single_pass(schema, title, pdf_path.name, clipped)
        data = call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
        data["_file"] = pdf_path.name
        if "citation" in data and isinstance(data["citation"], dict):
            data["citation"].setdefault("title", title)
        else:
            data["citation"] = {"title": title}
        print(f"[PIPELINE] {pdf_path.name}: card built via fallback single-pass.")
        return data

    print(f"[PIPELINE] {pdf_path.name}: reducing {len(partials)} partial cards.")
    user = prompt_reduce(schema, title, pdf_path.name, partials)
    data = call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
    data["_file"] = pdf_path.name
    if "citation" in data and isinstance(data["citation"], dict):
        data["citation"].setdefault("title", title)
    else:
        data["citation"] = {"title": title}
    print(f"[PIPELINE] {pdf_path.name}: final reduced card built.")
    return data

def _first_string_from(obj):
    """Get a short string from obj whether it's a list, dict, or str."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        for x in obj:
            s = _first_string_from(x)
            if s: return s
        return ""
    if isinstance(obj, dict):
        # look for common keys first
        for key in ("core_results", "claims", "evidence_snippets", "notes", "summary", "findings"):
            if key in obj:
                s = _first_string_from(obj[key])
                if s: return s
        # otherwise scan values
        for v in obj.values():
            s = _first_string_from(v)
            if s: return s
    return ""

def _nested_len(obj):
    """Count leaf items (strings) in lists/dicts; returns 0 for non-iterables."""
    if isinstance(obj, str) or obj is None:
        return 1 if isinstance(obj, str) and obj.strip() else 0
    if isinstance(obj, list):
        return sum(_nested_len(v) for v in obj)
    if isinstance(obj, dict):
        return sum(_nested_len(v) for v in obj.values())
    return 0

def _flat_counts(prefix, obj, out):
    """
    Flatten nested list/dict sizes into counts, e.g.
    embodied_facets:gesture_action:claims = 3
    """
    if isinstance(obj, list):
        out[f"{prefix}"] = len(obj)
        # also count leaf strings
        out[f"{prefix}:leaf_count"] = _nested_len(obj)
        return
    if isinstance(obj, dict):
        # total leafs under this node
        out[f"{prefix}:leaf_count"] = _nested_len(obj)
        for k, v in obj.items():
            _flat_counts(f"{prefix}:{k}", v, out)
        return
    # scalar
    out[f"{prefix}"] = 1 if (isinstance(obj, str) and obj.strip()) else 0

def compact_row(card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Schema-agnostic compact row:
    - always writes citation basics if present
    - extracts a first meaningful finding string from any nested structure
    - adds nested counts for major sections to help quick browsing
    """
    cit = card.get("citation") or {}
    row = {
        "file": card.get("_file", ""),
        "title": cit.get("title", ""),
        "year": cit.get("year", ""),
        "venue": cit.get("venue", ""),
        "doi": cit.get("doi", "")
    }

    # Try to extract one representative finding
    finding_1 = ""
    if "findings" in card:
        finding_1 = _first_string_from(card["findings"])
    elif "results" in card:
        finding_1 = _first_string_from(card["results"])
    row["finding_1"] = (finding_1 or "")[:200]

    # Methods summary (best-effort)
    meth = card.get("methods") or {}
    if isinstance(meth, dict):
        row["methods_type"] = meth.get("study_type", "") or meth.get("type", "")
        row["N"] = meth.get("n", "")

    # Add nested counts for a few common big sections if present
    for top_key in ("embodied_facets", "embodiment_facets", "perceptual_systems", "symbolic_systems_and_models"):
        if top_key in card:
            _flat_counts(top_key, card[top_key], row)

    # Also count any top-level lists
    for k, v in card.items():
        if k.startswith("_"): 
            continue
        if isinstance(v, list):
            row[f"len_{k}"] = len(v)
    return row


# ------------------ Public runner for API / CLI ------------------

def run_gemini_cards(
    schema_name: str,
    papers_folder: str,
    model: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Run the Gemini cards pipeline over all PDFs in papers/<papers_folder>,
    using schema cards/<schema_name>.

    Returns: (jsonl_path, csv_path, pdf_path) as absolute Paths.
    """

    # --- resolve schema ---
    schema_file = schema_name if schema_name.endswith(".json") else schema_name + ".json"
    schema_path = CARDS_DIR / schema_file
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    # --- resolve papers dir ---
    papers_dir = PAPERS_DIR / papers_folder
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers folder not found: {papers_dir}")

    # --- set model if provided ---
    from gemini_cards import GEMINI_MODEL as _GEMINI_MODEL
    if model:
        # dirty but simple: overwrite global
        import gemini_cards
        gemini_cards.GEMINI_MODEL = model

    print(f"[UI] Using schema: {schema_path}")
    print(f"[UI] Papers folder: {papers_dir}")
    print(f"[UI] Results dir:   {RESULTS_DIR}")

    # --- initialise Gemini, load schema ---
    init_gemini()
    schema = load_schema(schema_path)

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {papers_dir}")

    cards: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for pdf in pdfs:
        try:
            card = build_card_for_pdf(pdf, schema)
            cards.append(card)
        except Exception as e:
            print(f"[WARN|card] {pdf.name}: {e}")
            continue

        try:
            rows.append(compact_row(card))
        except Exception as e:
            print(f"[WARN|row ] {pdf.name}: {e}")

    if not cards:
        raise RuntimeError("No cards were generated.")

    # --- build base name with schema + folder like you asked ---
    schema_stem = schema_path.stem          # e.g. "networks"
    folder_stem = Path(papers_folder).name  # e.g. "plant_cognition"
    base_name   = f"cards_{schema_stem}__{folder_stem}"

    jsonl_path = RESULTS_DIR / f"{base_name}.jsonl"
    csv_path   = RESULTS_DIR / f"{base_name}_summary.csv"

    # save JSONL + CSV
    with jsonl_path.open("w", encoding="utf-8") as f:
        for c in cards:
            import json
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"[UI] Saved JSONL to {jsonl_path}")
    print(f"[UI] Saved CSV   to {csv_path}")

        # --- build PDF directly using your cards_to_pdf helper ---
    pdf_path = RESULTS_DIR / f"{base_name}.pdf"
    build_pdf(jsonl_path, pdf_path)
    print(f"[UI] Saved PDF   to {pdf_path}")

    # --- META-CARD generation + PDF (NEW) ---
    try:
        # 1) Load all schemas/cards from the JSONL you just wrote
        schemas_for_meta = load_schemas_from_file(jsonl_path)

        # 2) Ask Gemini to create a single meta-card
        meta_card = generate_meta_card(
            schemas_for_meta,
            model=model or GEMINI_MODEL,
        )

        # 3) Save meta-card as a single JSON
        meta_json_path = RESULTS_DIR / f"{base_name}__meta.json"
        with meta_json_path.open("w", encoding="utf-8") as f:
            json.dump(meta_card, f, ensure_ascii=False, indent=2)

        # 4) Also save as JSONL with a *single* line so we can reuse build_pdf
        meta_jsonl_path = RESULTS_DIR / f"{base_name}__meta.jsonl"
        with meta_jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta_card, ensure_ascii=False) + "\n")

        # 5) Build PDF for the meta-card
        meta_pdf_path = RESULTS_DIR / f"{base_name}__meta.pdf"
        build_pdf(meta_jsonl_path, meta_pdf_path)

        print(f"[UI] Saved META JSON to {meta_json_path}")
        print(f"[UI] Saved META PDF  to {meta_pdf_path}")

    except Exception as e:
        print(f"[WARN] Could not build meta-card PDF: {e}")

    return jsonl_path.resolve(), csv_path.resolve(), pdf_path.resolve()


def run_gemini_cards_with_progress(
    schema_name: str,
    papers_folder: str,
    model: Optional[str] = None,
    progress_callback=None,
) -> Tuple[Path, Path, Path]:
    """
    Same as run_gemini_cards but with progress callbacks for real-time updates.
    
    progress_callback(stage: str, current: int, total: int, message: str)
    """
    def report(stage: str, current: int, total: int, message: str):
        if progress_callback:
            progress_callback(stage, current, total, message)
        print(f"[{stage}] {current}/{total} - {message}")

    # --- resolve schema ---
    schema_file = schema_name if schema_name.endswith(".json") else schema_name + ".json"
    schema_path = CARDS_DIR / schema_file
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    # --- resolve papers dir ---
    papers_dir = PAPERS_DIR / papers_folder
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers folder not found: {papers_dir}")

    # --- set model if provided ---
    if model:
        import gemini_cards
        gemini_cards.GEMINI_MODEL = model

    report("init", 0, 100, "Initializing Gemini...")
    init_gemini()
    schema = load_schema(schema_path)

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {papers_dir}")

    total_pdfs = len(pdfs)
    cards: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    report("loading", 0, total_pdfs, f"Found {total_pdfs} PDFs to process")

    for idx, pdf in enumerate(pdfs):
        report("cards", idx + 1, total_pdfs, f"Processing: {pdf.name}")
        try:
            card = build_card_for_pdf(pdf, schema)
            cards.append(card)
        except Exception as e:
            print(f"[WARN|card] {pdf.name}: {e}")
            continue

        try:
            rows.append(compact_row(card))
        except Exception as e:
            print(f"[WARN|row ] {pdf.name}: {e}")

    if not cards:
        raise RuntimeError("No cards were generated.")

    # --- build base name ---
    schema_stem = schema_path.stem
    folder_stem = Path(papers_folder).name
    base_name   = f"cards_{schema_stem}__{folder_stem}"

    report("saving", 1, 4, "Saving JSONL...")
    jsonl_path = RESULTS_DIR / f"{base_name}.jsonl"
    csv_path   = RESULTS_DIR / f"{base_name}_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report("saving", 2, 4, "Saving CSV...")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    report("saving", 3, 4, "Generating PDF report...")
    pdf_path = RESULTS_DIR / f"{base_name}.pdf"
    build_pdf(jsonl_path, pdf_path)

    # --- META-CARD ---
    report("meta", 1, 3, "Generating meta-card with Gemini...")
    try:
        schemas_for_meta = load_schemas_from_file(jsonl_path)
        meta_card = generate_meta_card(
            schemas_for_meta,
            model=model or GEMINI_MODEL,
        )

        report("meta", 2, 3, "Saving meta-card...")
        meta_json_path = RESULTS_DIR / f"{base_name}__meta.json"
        with meta_json_path.open("w", encoding="utf-8") as f:
            json.dump(meta_card, f, ensure_ascii=False, indent=2)

        meta_jsonl_path = RESULTS_DIR / f"{base_name}__meta.jsonl"
        with meta_jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta_card, ensure_ascii=False) + "\n")

        report("meta", 3, 3, "Generating meta-card PDF...")
        meta_pdf_path = RESULTS_DIR / f"{base_name}__meta.pdf"
        build_pdf(meta_jsonl_path, meta_pdf_path)

    except Exception as e:
        print(f"[WARN] Could not build meta-card: {e}")

    report("complete", 100, 100, "All done!")
    return jsonl_path.resolve(), csv_path.resolve(), pdf_path.resolve()


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Gemini cards from PDFs with schema loaded from JSON."
    )
    ap.add_argument(
        "--schema", type=str, default="plant_cognition.json",
        help="Schema file name in ../cards (e.g. plant_cognition.json)."
    )
    ap.add_argument(
        "--papers", type=str, default=str(PAPERS_DIR),
        help="Folder with PDFs (absolute or subfolder of ../papers)."
    )
    ap.add_argument(
        "--model", type=str, default=GEMINI_MODEL,
        help="Gemini model name."
    )
    args = ap.parse_args()

    run_gemini_cards(
        schema_name=args.schema,
        papers_folder=args.papers,
        model=args.model,
    )


if __name__ == "__main__":
    main()
