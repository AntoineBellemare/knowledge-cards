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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError, DeadlineExceeded

from cards_to_pdf import build_pdf  # <- use the build_pdf you showed
from generate_meta_card_UI import load_schemas_from_file, generate_meta_card, generate_speculation

# Thread-local storage for cancellation check
_thread_locals = threading.local()

def set_cancellation_check(check_func):
    """Set the cancellation check function for the current thread."""
    _thread_locals.cancellation_check = check_func

def get_cancellation_check():
    """Get the cancellation check function for the current thread."""
    return getattr(_thread_locals, 'cancellation_check', None)



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
MAX_INPUT_TOKENS = 2000    # single-pass only for very short papers
CHUNK_TOK_TARGET = 1000    # ~1k words per chunk for focused extraction
CHUNK_OVERLAP = 200        # 20% overlap to maintain context


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
    """Drop references/bibliography, but only if it's in the last 30% of the document."""
    words = text.split()
    total_words = len(words)
    
    # Only look for refs/bibliography in the last 30% of the document
    cutoff_start = int(total_words * 0.7)
    tail_text = " ".join(words[cutoff_start:])
    
    # Try to find references section in the tail
    match = re.search(
        r'\n\s*(references|bibliography|works cited|supplementary materials?|appendix)\s*\n',
        tail_text, flags=re.I
    )
    
    if match:
        # Found it in the tail - cut from there
        refs_start_in_tail = len(tail_text[:match.start()].split())
        cut_at = cutoff_start + refs_start_in_tail
        result = " ".join(words[:cut_at])
        print(f"[PDF] Dropped references section (kept {cut_at}/{total_words} words)")
        return result.strip()
    
    # No references found or it's in the main body - keep everything
    return text.strip()

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
    total_pages = len(doc)
    print(f"[PDF] {path.name} has {total_pages} pages")
    
    for i, p in enumerate(doc):
        # Try multiple extraction methods
        txt = p.get_text("text")
        
        # If text extraction yields nothing, try "blocks" method
        if len(txt.strip()) < 50:
            blocks = p.get_text("blocks")
            # blocks is a list of tuples: (x0, y0, x1, y1, "text", block_no, block_type)
            txt = " ".join([block[4] for block in blocks if len(block) > 4])
        
        # Now normalize the text
        txt = normalize_text(txt)
        pages.append(txt)
        
        if i == 0:
            # quick peek at first page length
            print(f"[PDF] Page 1 length for {path.name}: {len(txt.split())} words")
        if i == min(5, total_pages - 1):
            # Check a middle page too
            print(f"[PDF] Page {i+1} length for {path.name}: {len(txt.split())} words")
    
    full = "\n\n".join(pages)
    print(f"[PDF] Total extracted BEFORE refs-drop for {path.name}: {len(full.split())} words")
    full = drop_refs_and_supp(full)
    print(f"[PDF] Total length AFTER refs-drop for {path.name}: {len(full.split())} words")

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
    # Check for cancellation before making API call
    check_func = get_cancellation_check()
    if check_func and check_func():
        raise RuntimeError("Job cancelled by user")
    
    m = genai.GenerativeModel(
        model,
        system_instruction=system_msg,
        generation_config={"response_mime_type": "application/json"}
    )
    last_err = None
    for t in range(retries):
        # Check cancellation before each retry
        if check_func and check_func():
            raise RuntimeError("Job cancelled by user")
        
        try:
            # print(f"[Gemini] Call attempt {t+1}/{retries} with model={model}...")
            print(f"[Gemini] Prompt length: {len(user_msg.split())} words, {len(user_msg)} chars")
            resp = m.generate_content(user_msg)
            # print("[Gemini] Call succeeded.")
            raw_text = resp.text.strip()
            # Clean up common JSON formatting issues
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return json.loads(raw_text.strip())
        except json.JSONDecodeError as e:
            last_err = e
            print(f"[Gemini] JSON decode error on attempt {t+1}: {e}")
            if t < retries - 1:
                print(f"[Gemini] Response preview: {resp.text[:500]}...")
                time.sleep(1.0 * (t + 1))
            else:
                print(f"[Gemini] Full malformed response saved for debugging")
                # Save the malformed response for debugging
                try:
                    with open(f"debug_malformed_{int(time.time())}.txt", "w", encoding="utf-8") as f:
                        f.write(resp.text)
                except:
                    pass
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
    * Extract quotes that are insightful, contain key claims, or have vivid phrasing.
    * Max 3-4 quotes per chunk, each ≤250 characters.
    * Add location hint: [chunk: {section}].
    * Do NOT copy reference lists or bibliographic entries.
- For list fields: extract relevant items from this chunk.
- Ignore pure reference entries (author/year/journal/DOI).
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

MERGE RULES (VERY IMPORTANT):
- Union/merge lists; deduplicate similar items.
- Prefer more specific/quantified info when conflicts.
- Keep methods coherent (type, data, N if present, measures).
- findings: Concise bullets (max 6) reflecting the paper's main results. Summarize related findings together.
- For any field whose name contains 'citation' or 'quote':
    * Keep the MOST impactful, non-redundant quotes (aim for 8-12 total for long papers).
    * Merge or drop quotes that say essentially the same thing.
    * Each quote should be ≤250 characters. Trim longer ones while preserving meaning.
    * Drop generic statements, reference lines, or bibliographic text.
    * Prioritize quotes with specific claims, data, or unique phrasing.
- For list fields (except citations/quotes): max 6 items. Merge similar items.
- If a field remains unknown overall, keep "" or [].
- Return FINAL JSON only.
"""

def reduce_in_batches(schema: Dict[str, Any], title: str, filename: str, partial_cards: List[Dict[str, Any]], 
                      batch_size: int = 4, progress_callback=None, global_progress: dict = None) -> Dict[str, Any]:
    """
    Hierarchically reduce partial cards in batches to avoid overwhelming the LLM.
    
    For books with many chunks (e.g., 30), reduces them in groups:
    - Batch 1-4 → intermediate card 1
    - Batch 5-8 → intermediate card 2
    - etc.
    Then recursively reduces those intermediate cards → final card
    
    Uses smaller batch size (4) to keep prompts under ~50KB.
    """
    def report_progress(message):
        if progress_callback and global_progress:
            global_progress['current'] += 1
            progress_callback(
                "process", 
                global_progress['current'], 
                global_progress['total'], 
                message
            )
        print(f"[REDUCE] {message}")
    
    # BASE CASE: If only 1 card, just return it (nothing to reduce)
    if len(partial_cards) == 1:
        print("[REDUCE] Single card - no reduction needed")
        return partial_cards[0]
    
    # Check total size of cards to determine if we need smaller batches
    total_chars = sum(len(json.dumps(c)) for c in partial_cards)
    
    # If total is small enough (< 30KB) and few cards, reduce directly in one pass
    if len(partial_cards) <= batch_size and total_chars < 30000:
        report_progress(f"{filename}: final merge of {len(partial_cards)} cards")
        user = prompt_reduce(schema, title, filename, partial_cards)
        return call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
    
    # If we have few cards but they're too large, we still need to reduce them
    # (2+ cards that are individually large)
    if len(partial_cards) == 2 and total_chars >= 30000:
        # Can't batch further, just try to reduce the 2 cards
        report_progress(f"{filename}: merging 2 large cards")
        user = prompt_reduce(schema, title, filename, partial_cards)
        return call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
    
    # If we have few cards but they're large, reduce batch size
    if len(partial_cards) <= batch_size:
        batch_size = 2
    
    # Recursive case: split into batches, reduce each, then reduce the results
    total_batches = (len(partial_cards) + batch_size - 1) // batch_size
    print(f"[REDUCE] Splitting {len(partial_cards)} cards into {total_batches} batches")
    intermediate_cards = []
    
    for i in range(0, len(partial_cards), batch_size):
        batch = partial_cards[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        report_progress(f"{filename}: reduce batch {batch_num}/{total_batches}")
        
        user = prompt_reduce(schema, title, filename, batch)
        intermediate = call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
        intermediate_cards.append(intermediate)
    
    print(f"[REDUCE] Batch processing complete. Reducing {len(intermediate_cards)} intermediate cards")
    # Recursively reduce the intermediate cards (in case there are many batches)
    return reduce_in_batches(schema, title, filename, intermediate_cards, batch_size, progress_callback, global_progress)

# ------------------ Pipeline ------------------
def text_size_ok(sections: List[Tuple[str, str]]) -> bool:
    # crude: use words count as proxy for tokens
    words = sum(len(t.split()) for _, t in sections)
    return words <= MAX_INPUT_TOKENS

def build_card_for_pdf(pdf_path: Path, schema: Dict[str, Any], progress_callback=None, 
                       global_progress: dict = None, cancellation_check=None) -> Dict[str, Any]:
    """
    Build a card for a single PDF. Uses single-pass for small docs, map-reduce for large ones.
    
    progress_callback(stage: str, current: int, total: int, message: str)
    global_progress: dict with 'current', 'total', 'pdf_name' for global tracking
    cancellation_check: Optional callable that returns True if job should be cancelled
    """
    def report_progress(message, current=None, total=None):
        # If current and total are provided (for chunk tracking), use them
        # Otherwise use global progress tracking
        if current is not None and total is not None and progress_callback:
            progress_callback("process", current, total, message)
        elif progress_callback and global_progress:
            global_progress['current'] += 1
            progress_callback(
                "process", 
                global_progress['current'], 
                global_progress['total'], 
                message
            )
        print(f"[PROGRESS] {message}")
    
    print(f"\n[PIPELINE] ==== Processing {pdf_path.name} ====")
    title, sections = extract_pdf(pdf_path)
    total_words = sum(len(t.split()) for _, t in sections)
    print(f"[PIPELINE] {pdf_path.name}: {total_words} words across {len(sections)} sections.")

    if text_size_ok(sections):
        report_progress(f"{pdf_path.name}: processing...")
        full_text = "\n\n".join(f"[{s}]\n{t}" for s, t in sections)
        user = prompt_single_pass(schema, title, pdf_path.name, full_text)
        data = call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
        data["_file"] = pdf_path.name
        if "citation" in data and isinstance(data["citation"], dict):
            data["citation"].setdefault("title", title)
        else:
            data["citation"] = {"title": title}
        return data

    # map-reduce with parallel chunk processing
    chunks = build_chunks(sections)
    total_chunks = len(chunks)
    
    # Determine parallelism: use 3-5 concurrent workers for Gemini API
    # (Gemini has rate limits, so we don't want too many parallel calls)
    max_workers = min(5, max(1, total_chunks // 2))  # 3-5 workers typical
    
    print(f"[PIPELINE] Processing {total_chunks} chunks with {max_workers} parallel workers")
    
    def process_chunk(idx, ch):
        """Process a single chunk - designed to run in parallel."""
        # Check for cancellation before processing
        if cancellation_check and cancellation_check():
            raise RuntimeError("Job cancelled by user")
        
        user = prompt_map_chunk(schema, title, pdf_path.name, ch["section"], ch["text"])
        try:
            part = call_gemini_json(GEMINI_MODEL, SYSTEM_CARD, user)
            
            # Check for cancellation after API call
            if cancellation_check and cancellation_check():
                raise RuntimeError("Job cancelled by user")
            
            return idx, part, None
        except Exception as e:
            # Re-raise cancellation errors immediately
            if "cancelled" in str(e).lower():
                raise
            print(f"[WARN|chunk] {pdf_path.name} chunk {idx+1}: {e}")
            return idx, None, str(e)
    
    partials = []
    completed_count = 0
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        futures = {executor.submit(process_chunk, idx, ch): idx for idx, ch in enumerate(chunks)}
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                idx, part, error = future.result()
                completed_count += 1
                
                # Report progress
                report_progress(
                    f"{pdf_path.name} ({total_chunks} chunks): completed {completed_count}/{total_chunks}",
                    current=completed_count,
                    total=total_chunks
                )
                
                if part is not None:
                    partials.append((idx, part))  # Keep track of index for ordering
            except Exception as e:
                # Re-raise cancellation errors
                if "cancelled" in str(e).lower():
                    raise
                print(f"[WARN|chunk] Error in parallel processing: {e}")
    
    # Sort partials by original index to maintain document order
    partials.sort(key=lambda x: x[0])
    partials = [part for idx, part in partials]  # Extract just the parts

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

    report_progress(f"{pdf_path.name}: reducing {len(partials)} chunks into final card...")
    
    # For long documents (books), use hierarchical batch reduction
    if len(partials) > 15:
        data = reduce_in_batches(schema, title, pdf_path.name, partials, 
                                  progress_callback=progress_callback, global_progress=None)
    else:
        # Single-pass reduction for shorter documents
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
    template_question: Optional[str] = None,
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
    
    # Load full schema file to extract question if not provided
    with schema_path.open("r", encoding="utf-8") as f:
        schema_data = json.load(f)
    
    # Extract the actual schema and question
    if "schema" in schema_data:
        schema = schema_data["schema"]
        if not template_question:
            template_question = schema_data.get("question", "")
    else:
        schema = schema_data
    
    print(f"[UI] Template question: {template_question[:100]}..." if template_question else "[UI] No template question found")

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

    # --- META-CARD generation + PDF (with SPECULATION) ---
    try:
        # 1) Load all schemas/cards from the JSONL you just wrote
        schemas_for_meta = load_schemas_from_file(jsonl_path)

        # 2) Ask Gemini to create a single meta-card (now includes speculation)
        meta_card = generate_meta_card(
            schemas_for_meta,
            model=model or GEMINI_MODEL,
            question=template_question,  # Pass question for speculative synthesis
            include_speculation=True,
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
    schema_data: Optional[Dict[str, Any]] = None,
    template_question: Optional[str] = None,
    cancellation_check=None,
) -> Tuple[Path, Path, Path]:
    """
    Same as run_gemini_cards but with progress callbacks for real-time updates.
    
    progress_callback(stage: str, current: int, total: int, message: str)
    schema_data: If provided, use this schema directly instead of loading from file.
    template_question: The original question that generated the template schema.
    cancellation_check: Optional callable that returns True if job should be cancelled.
    """
    # Set cancellation check in thread-local storage so it's accessible everywhere
    set_cancellation_check(cancellation_check)
    
    def report(stage: str, current: int, total: int, message: str):
        if progress_callback:
            progress_callback(stage, current, total, message)
        print(f"[{stage}] {current}/{total} - {message}")

    # --- resolve papers dir ---
    papers_dir = PAPERS_DIR / papers_folder
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers folder not found: {papers_dir}")

    # --- set model if provided ---
    global GEMINI_MODEL
    if model:
        GEMINI_MODEL = model
        print(f"[CONFIG] Using model: {model}")

    report("init", 0, 100, "Initializing Gemini...")
    init_gemini()
    
    # --- resolve schema ---
    schema_stem = schema_name.replace('.json', '')  # Default stem from name
    if schema_data:
        # Use provided schema data directly (from database)
        schema = schema_data
        print(f"[CONFIG] Using schema from database: {schema_name}")
    else:
        # Load from file
        schema_file = schema_name if schema_name.endswith(".json") else schema_name + ".json"
        schema_path = CARDS_DIR / schema_file
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        schema = load_schema(schema_path)
        schema_stem = schema_path.stem

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {papers_dir}")

    total_pdfs = len(pdfs)
    cards: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    # Simple progress tracking: each PDF is a unit, subdivide progress within each
    report("scan", 0, total_pdfs, f"Found {total_pdfs} PDFs to process")
    
    # Global progress tracker - use PDF count as base, each PDF worth 100 points
    global_progress = {
        'current': 0, 
        'total': total_pdfs * 100,  # Simple: each PDF is 100 points
        'current_pdf': 0
    }

    for idx, pdf in enumerate(pdfs):
        # Check for cancellation before processing each PDF
        if cancellation_check and cancellation_check():
            raise RuntimeError("Job cancelled by user")
        
        global_progress['current_pdf'] = idx
        pdf_base_progress = idx * 100  # Starting point for this PDF
        
        report("process", pdf_base_progress, global_progress['total'], f"Processing {idx+1}/{total_pdfs}: {pdf.name}")
        
        try:
            # Wrapper for progress callback that includes PDF context and chunk info
            def pdf_progress_callback(stage: str, current: int, total: int, message: str):
                if total > 0:
                    # Map chunk progress (current/total) to this PDF's allocated range (0-100)
                    pdf_percent = (current / total) * 100
                    overall_current = pdf_base_progress + int(pdf_percent)
                    # Pass the detailed message through (includes chunk info)
                    progress_callback(stage, overall_current, global_progress['total'], message)
            
            card = build_card_for_pdf(
                pdf, schema, 
                progress_callback=pdf_progress_callback, 
                global_progress=None,
                cancellation_check=cancellation_check
            )
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
    folder_stem = Path(papers_folder).name
    base_name   = f"cards_{schema_stem}__{folder_stem}"

    # Saving files - last 5% of progress
    save_progress_base = total_pdfs * 100
    report("save", save_progress_base, save_progress_base + 5, "Saving JSONL...")
    jsonl_path = RESULTS_DIR / f"{base_name}.jsonl"
    csv_path   = RESULTS_DIR / f"{base_name}_summary.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for c in cards:
            # Add the original question to each card if available
            if template_question:
                c["_template_question"] = template_question
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report("save", save_progress_base + 1, save_progress_base + 5, "Saving CSV...")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    report("save", save_progress_base + 2, save_progress_base + 5, "Generating PDF...")
    pdf_path = RESULTS_DIR / f"{base_name}.pdf"
    build_pdf(jsonl_path, pdf_path)

    # --- META-CARD with SPECULATION ---
    report("meta", save_progress_base + 3, save_progress_base + 5, "Building meta-card...")
    try:
        schemas_for_meta = load_schemas_from_file(jsonl_path)
        meta_card = generate_meta_card(
            schemas_for_meta,
            model=model or GEMINI_MODEL,
            question=template_question,
            include_speculation=True,
        )
        
        if template_question:
            meta_card["_template_question"] = template_question

        report("meta", save_progress_base + 4, save_progress_base + 5, "Saving meta-card...")
        meta_json_path = RESULTS_DIR / f"{base_name}__meta.json"
        with meta_json_path.open("w", encoding="utf-8") as f:
            json.dump(meta_card, f, ensure_ascii=False, indent=2)

        meta_jsonl_path = RESULTS_DIR / f"{base_name}__meta.jsonl"
        with meta_jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta_card, ensure_ascii=False) + "\n")

        meta_pdf_path = RESULTS_DIR / f"{base_name}__meta.pdf"
        build_pdf(meta_jsonl_path, meta_pdf_path)

    except Exception as e:
        print(f"[WARN] Could not build meta-card: {e}")

    report("complete", save_progress_base + 5, save_progress_base + 5, "All done!")
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
