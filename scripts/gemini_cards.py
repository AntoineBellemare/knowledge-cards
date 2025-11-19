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

# ------------------ CONFIG ------------------
PAPERS_DIR = Path("../papers")
OUT_DIR = Path("../results"); OUT_DIR.mkdir(parents=True, exist_ok=True)

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
            print(f"[Gemini] Call attempt {t+1}/{retries} with model={model}...")
            print(f"[Gemini] Prompt length: {len(user_msg.split())} words, {len(user_msg)} chars")
            resp = m.generate_content(user_msg)
            print("[Gemini] Call succeeded.")
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
"Return strictly valid JSON matching the schema provided. Do not add fields. Be concise, factual, and cite tiny location hints like [Intro], [Methods], [Results] when obvious."
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

RULES:
- Populate every field in the schema. If unknown, use "" or [].
- findings: short bullet points with concrete outcomes.
- methods: include type (e.g., behavioral/EEG/fMRI/corpus/comp.), data, N if visible, and measures.
- embodiment-like facets (if present in your schema): add short bullets with location hints like [Methods]/[Results]/[Discussion] when obvious.
- Do not invent content. JSON only.
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

RULES:
- Extract ONLY from this chunk.
- Be concise.
- JSON only.
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

MERGE RULES:
- Union/merge lists; deduplicate.
- Prefer more specific/quantified info when conflicts.
- Keep methods coherent (type, data, N if present, measures).
- findings: short bullets (≤6) reflecting the paper’s core results.
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


def main():
    import argparse
    global GEMINI_MODEL

    ap = argparse.ArgumentParser(description="Gemini cards from PDFs with schema loaded from JSON.")
    ap.add_argument("--schema", type=str, default="plant_cognition.json", help="Path to schema JSON.")
    ap.add_argument("--papers", type=str, default=str(PAPERS_DIR), help="Folder with PDFs.")
    ap.add_argument("--model", type=str, default=GEMINI_MODEL, help="Gemini model name.")
    args = ap.parse_args()

    GEMINI_MODEL = args.model

    print(f"[SETUP] Using schema: {args.schema}")
    print(f"[SETUP] Papers folder: {args.papers}")
    print(f"[SETUP] Gemini model: {GEMINI_MODEL}")

    init_gemini()
    print("[SETUP] Gemini initialized.")

    schema = load_schema(Path("../cards") / args.schema)
    print("[SETUP] Schema loaded.")

    pdfs = sorted(Path(args.papers).glob("*.pdf"))
    print(f"[SETUP] Found {len(pdfs)} PDF(s).")
    if not pdfs:
        print(f"No PDFs found in {args.papers}")
        return

    cards = []
    rows  = []
    for pdf in tqdm(pdfs, desc="Processing PDFs"):
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

    if cards:
        # Extract schema name from path to create dynamic output filenames
        schema_name = Path(args.schema).stem
        jsonl_path = OUT_DIR / f"cards_{schema_name}.jsonl"
        csv_path   = OUT_DIR / f"cards_{schema_name}_summary.csv"
        save_jsonl(cards, jsonl_path)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[DONE] Saved: {jsonl_path} and {csv_path}")
    else:
        print("[DONE] No cards were generated.")

if __name__ == "__main__":
    main()
