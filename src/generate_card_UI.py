#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a JSON SCHEMA/TEMPLATE from a question.
Ensures the schema always includes sub-sections for textual evidence in each category:
    "citations_or_quotes": []

Can be used:
- as a CLI script
- as a library function: generate_schema(question, model, pointers)
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, DeadlineExceeded


# ---------------- Gemini setup ----------------

SYSTEM_SCHEMA = (
    "You are a schema designer for research and conceptual reading cards. "
    "Given a user question, you design a JSON TEMPLATE (schema) that someone could use "
    "to systematically annotate and compare papers or other texts that address that question.\n\n"
    "GENERAL PRINCIPLES\n"
    "- Your output is ONLY a JSON object (no prose around it).\n"
    "- The JSON is a TEMPLATE: it defines fields and structure, not content.\n"
    "- All values must be EMPTY placeholders: \"\", [], false, or empty dicts {}.\n"
    "- Keys should be short, semantic, and snake_case (e.g. main_domain, core_claims, "
    "  evidence_snippets, network_measures_suggested).\n"
    "- Use at most 2–3 levels of nesting: broad sections → subfields → optional fine-grained slots.\n"
    "- A GOOD TEMPLATE is richer than a single block: aim for SEVERAL top-level sections "
    "  (typically 4–8) that capture different dimensions of analysis.\n\n"
    "DIMENSIONS TO CONSIDER WHEN DESIGNING THE SCHEMA\n"
    "Treat the following as a MENU of possible dimensions, not a fixed list. "
    "Choose and adapt only those dimensions that are clearly relevant to the user's question, "
    "and feel free to create new ones when appropriate.\n"
    "From the user's question, infer what they likely want to compare or track across texts. "
    "Design the schema so that it can capture multiple orthogonal aspects, which MAY include:\n"
    "1) ORIENTATION & SCOPE (what kind of system or phenomenon, what domain, what type of work).\n"
    "   Examples of fields: orientation_and_scope, system_focus, main_domain, work_type, "
    "   timeframe_or_context, scope_of_inference, research_question, thesis_claim.\n"
    "   Each such section/subsection MUST include a subfield \"citations_or_quotes\": [] for textual evidence.\n"
    "2) CONCEPTUAL / THEORETICAL FRAME.\n"
    "   How the text frames the phenomenon (e.g. distributed cognition, information processing, "
    "   embodiment, symbolic systems, networks). Include fields for core_metaphors, key_constructs, "
    "   links_to_theories, and relations_to_cognition_or_agency. Each subsection should include "
    "   \"citations_or_quotes\": [].\n"
    "3) DIMENSIONS OR FACETS SPECIFIC TO THE QUESTION.\n"
    "   Create one or more sections that capture the main axes implied by the question "
    "   (e.g. dimensions_of_cognition, embodied_facets, perceptual_facets, triad_alignment, "
    "   scales_and_substrates, network_properties). Use subfields for the key facets that someone "
    "   would want to rate, summarise, or list evidence for. Each facet should typically have "
    "   slots like claims, evidence_snippets, strength_or_confidence, and a \"citations_or_quotes\": [].\n"
    "4) PERCEPTUAL / SENSORIMOTOR / MULTISCALE ASPECTS (when relevant).\n"
    "   If the question involves perception, embodiment, or multiscale organisation, include a "
    "   section for perceptual_systems_or_modalities with subfields (e.g. auditory, visual, "
    "   proprioceptive_vestibular, interoceptive_affective, crossmodal_mappings, "
    "   temporal_dynamics, spatial_cognition), each with mentions, tasks_measures, notes, and "
    "   \"citations_or_quotes\": [].\n"
    "5) METHODS / EVIDENCE / METRICS (for empirical or mixed work).\n"
    "   Include a methods or methods_and_metrics section with subfields for study_type, "
    "   population_or_species, tasks, measures, n, data_types, analysis, "
    "   manipulations_or_conditions, and any metrics that align with the question "
    "   (e.g. complexity_measures, network_measures, information_measures). Each methods-related "
    "   subsection should include \"citations_or_quotes\": [].\n"
    "6) FINDINGS / CLAIMS / LIMITATIONS.\n"
    "   Include a findings or claims_and_evidence section with subfields such as core_results, "
    "   effect_directions, boundary_conditions, limitations, and confounds_or_alternative_explanations. "
    "   Provide \"citations_or_quotes\": [] inside these subsections to capture textual evidence.\n"
    "7) MODELS / SYMBOLIC OR COMPUTATIONAL SYSTEMS (when the question suggests it).\n"
    "   If relevant, add a section for models_or_symbolic_systems (e.g. symbolic_archives, "
    "   computational_interfaces, latent_spaces_or_embeddings, geometric_or_topological_rules), "
    "   each with hypotheses, objectives, patterns_motifs, examples, and \"citations_or_quotes\": [].\n"
    "8) ETHICS / COMMUNITY / CONTEXT (when applicable).\n"
    "   If the question touches on communities, indigenous knowledge, or protocols, include an "
    "   ethics_or_community section with subfields for methodologies, data_governance_protocols, "
    "   community_benefit_or_reciprocity, fieldwork_considerations, and notes. Each subsection must "
    "   contain \"citations_or_quotes\": [].\n"
    "9) QUOTES / TEXTUAL EVIDENCE.\n"
    "   DO NOT use a single top-level \"citations_or_quotes\" array. Instead, for each meaningful "
    "   subsection or subcategory include a \"citations_or_quotes\": [] slot to store citations, quotes, or snippets.\n"
    "10) OPEN QUESTIONS / VIGNETTES / FUTURE DIRECTIONS.\n"
    "   Include fields for open_questions, testable_predictions_or_vignettes, or similar, so the "
    "   template can capture where the text points beyond itself. These subsections should also have "
    "   \"citations_or_quotes\": [] where relevant.\n"
    "11) METAPHYSICAL AND POETIC DIMENSIONS (when relevant).\n"
    "   If the question engages with metaphysical, ontological, or poetic aspects, include a section "
    "   for metaphysical_and_poetic_dimensions with subfields such as ontological_claims, "
    "   poetic_or_literary_framings, emergent_or_ineffable_qualities, aesthetic_dimensions, "
    "   and philosophical_implications. Each subfield should include \"citations_or_quotes\": [].\n\n"
    "MANDATORY STRUCTURAL REQUIREMENTS\n"
    "- If the question suggests working with papers or articles, include a sub-level citation-like block, "
    "  for example under the key \"metadata\" or a per-item block such as \"conceptual/citation\" with fields like:\n"
    "    title, authors, year, venue, doi, keywords, and a \"citations_or_quotes\": [].\n"
    "- DO NOT provide a single top-level field named \"citations_or_quotes\". Instead ensure each subsection/subcategory "
    "  includes its own \"citations_or_quotes\": [] array for textual evidence.\n"
    "- Avoid having just one large top-level block. Prefer several named sections that separate focus, "
    "  facets/dimensions, methods/evidence, models/symbolic aspects, context/ethics, and open questions.\n"
    "- If in doubt, err on the side of grouping related fields into named sections rather than having many "
    "  flat top-level keys.\n"
    "- Adapt the section names and structure to the specific question instead of reusing the same fixed set of sections.\n"
    "Return strictly valid JSON. Do not include comments, explanations, or example content."
)



def init_gemini() -> None:
    """Configure Gemini from GOOGLE_API_KEY (env or .env)."""
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    genai.configure(api_key=key)


def call_gemini(model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Low-level wrapper around Gemini JSON output.
    Retries a few times on transient errors.
    """
    m = genai.GenerativeModel(
        model,  # model name must be positional
        system_instruction=system_prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    last_err: Optional[Exception] = None
    for a in range(3):
        try:
            resp = m.generate_content(user_prompt)
            return json.loads(resp.text)
        except (ResourceExhausted, DeadlineExceeded, GoogleAPIError, Exception) as e:
            last_err = e
            print(f"[WARN] Gemini attempt {a+1} failed: {e}")
    raise last_err  # type: ignore[arg-type]


def build_prompt(question: str, pointers: Optional[Dict[str, Any]] = None) -> str:
    """
    Build the user prompt from the question + optional pointers from the UI.

    Example pointers (front-end):
      {
        "emphasize_conceptual": true/false,
        "include_empirical_methods": true/false,
        "include_metaphysical_and_poetic": true/false,
        "include_ethics_context": true/false
      }
    """
    pointers = pointers or {}

    prefs_lines = []
    if pointers.get("emphasize_conceptual"):
        prefs_lines.append(
            "- Add strong conceptual / philosophical sections (ontology, theoretical framing) "
            "in addition to any other relevant sections."
        )
    if pointers.get("include_empirical_methods"):
        prefs_lines.append(
            "- Add explicit structure for empirical methods and metrics (study_type, tasks, measures, n, analysis), "
            "without removing other conceptual or contextual sections."
        )
    if pointers.get("include_metaphysical_and_poetic"):
        prefs_lines.append(
            "- Add specific sections for metaphysical and poetic dimensions where appropriate, "
            "alongside other analytic sections."
        )
    if pointers.get("include_ethics_context"):
        prefs_lines.append(
            "- Add sections for ethics, community, and broader context where relevant, "
            "without replacing other dimensions suggested by the question."
        )

    prefs_block = ""
    if prefs_lines:
        prefs_block = (
            "\nDESIGN PREFERENCES (ADDITIVE):\n"
            "The following preferences should ADD extra sections or emphasis, not replace other "
            "dimensions that are relevant to the question.\n"
            + "\n".join(prefs_lines)
            + "\n\n"
        )

    return f"""
Create a JSON SCHEMA (template) based on the user's question.

QUESTION:
\"\"\"{question}\"\"\"


{prefs_block}RULES:
- Infer meaningful sections tailored to this question (do not reuse one fixed template).
- Use nested dicts when needed (sections → subsections → fields).
- All fields must have empty values: "", [] or false or an empty object.
- For each meaningful section or subsection you define, include a subfield:
    "citations_or_quotes": []
  so that textual evidence can be stored locally in that part of the schema.
- Do NOT create a single top-level "citations_or_quotes" field.
- Do NOT add explanations or sample content.
- Output valid JSON only.
"""



def generate_schema(
    question: str,
    model: str = "gemini-2.5-flash-lite",
    pointers: Optional[Dict[str, Any]] = None,
    ensure_citations_field: bool = False,
) -> Dict[str, Any]:
    """
    High-level function to generate a schema dict from a question (+ optional pointers).
    Can be used by CLI *and* FastAPI.
    """
    init_gemini()
    prompt = build_prompt(question, pointers=pointers)
    schema = call_gemini(model, SYSTEM_SCHEMA, prompt)

    if ensure_citations_field:
        schema.setdefault("citations_or_quotes", [])

    return schema


def main():
    ap = argparse.ArgumentParser(description="Generate JSON schema template from question.")
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--out", default="schema_from_question.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    # place outputs under ../cards/ by default (unless an absolute path was provided)
    if not out_path.is_absolute():
        out_path = Path("../cards") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = generate_schema(
        question=args.question,
        model=args.model,
        pointers=None,              # CLI: you can later add flags and pass them here
        ensure_citations_field=False
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Schema saved to {out_path}")


if __name__ == "__main__":
    main()
