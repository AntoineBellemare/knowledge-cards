#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a JSON SCHEMA/TEMPLATE from a question.
Ensures the schema always includes a section for textual evidence:
    "citations_or_quotes": []
"""

import os, json, argparse
from pathlib import Path
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
    "From the user's question, infer what they likely want to compare or track across texts. "
    "Design the schema so that it can capture multiple orthogonal aspects, which MAY include:\n"
    "1) ORIENTATION & SCOPE (what kind of system or phenomenon, what domain, what type of work).\n"
    "   Examples of fields: orientation_and_scope, system_focus, main_domain, work_type, "
    "   timeframe_or_context, scope_of_inference, research_question, thesis_claim.\n"
    "2) CONCEPTUAL / THEORETICAL FRAME.\n"
    "   How the text frames the phenomenon (e.g. distributed cognition, information processing, "
    "   embodiment, symbolic systems, networks). Include fields for core metaphors, key constructs, "
    "   links_to_theories, and relations_to_cognition_or_agency.\n"
    "3) DIMENSIONS OR FACETS SPECIFIC TO THE QUESTION.\n"
    "   Create one or more sections that capture the main axes implied by the question "
    "   (e.g. dimensions_of_cognition, embodied_facets, perceptual_facets, triad_alignment, "
    "   scales_and_substrates, network_properties). Use subfields for the key facets that someone "
    "   would want to rate, summarise, or list evidence for. Each facet should typically have "
    "   slots like claims, evidence_snippets, and optional strength_or_confidence.\n"
    "4) PERCEPTUAL / SENSORIMOTOR / MULTISCALE ASPECTS (when relevant).\n"
    "   If the question involves perception, embodiment, or multiscale organisation, include a "
    "   section for perceptual_systems_or_modalities with subfields (e.g. auditory, visual, "
    "   proprioceptive_vestibular, interoceptive_affective, crossmodal_mappings, "
    "   temporal_dynamics, spatial_cognition), each with mentions, tasks_measures, and notes.\n"
    "5) METHODS / EVIDENCE / METRICS (for empirical or mixed work).\n"
    "   Include a methods or methods_and_metrics section with subfields for study_type, "
    "   population_or_species, tasks, measures, n, data_types, analysis, "
    "   manipulations_or_conditions, and any metrics that align with the question "
    "   (e.g. complexity_measures, network_measures, information_measures).\n"
    "6) FINDINGS / CLAIMS / LIMITATIONS.\n"
    "   Include a findings or claims_and_evidence section with subfields such as core_results, "
    "   effect_directions, boundary_conditions, limitations, and confounds_or_alternative_explanations.\n"
    "7) MODELS / SYMBOLIC OR COMPUTATIONAL SYSTEMS (when the question suggests it).\n"
    "   If relevant, add a section for models_or_symbolic_systems (e.g. symbolic_archives, "
    "   computational_interfaces, latent_spaces_or_embeddings, geometric_or_topological_rules), "
    "   each with hypotheses, objectives, patterns_motifs, or examples.\n"
    "8) ETHICS / COMMUNITY / CONTEXT (when applicable).\n"
    "   If the question touches on communities, indigenous knowledge, or protocols, include an "
    "   ethics_or_community section with subfields for methodologies, data_governance_protocols, "
    "   community_benefit_or_reciprocity, fieldwork_considerations, and notes.\n"
    "9) QUOTES / TEXTUAL EVIDENCE.\n"
    "   ALWAYS include a dedicated field for textual evidence such as citations, quotes, or snippets. "
    "   Use a top-level field named exactly \"citations_or_quotes\": [] to store these.\n"
    "10) OPEN QUESTIONS / VIGNETTES / FUTURE DIRECTIONS.\n"
    "   Include fields for open_questions, testable_predictions_or_vignettes, or similar, so the "
    "   template can capture where the text points beyond itself.\n\n"
    "MANDATORY STRUCTURAL REQUIREMENTS\n"
    "- If the question suggests working with papers or articles, include a top-level citation-like block, "
    "  for example under the key \"citation\" or \"metadata\", with fields like:\n"
    "    title, authors, year, venue, doi, keywords.\n"
    "- ALWAYS include the top-level field:\n"
    "    \"citations_or_quotes\": []\n"
    "- Avoid having just one large top-level block. Prefer several named sections that separate focus, "
    "  facets/dimensions, methods/evidence, models/symbolic aspects, context/ethics, and open questions.\n"
    "- If in doubt, err on the side of grouping related fields into named sections rather than having many "
    "  flat top-level keys.\n"
    "Return strictly valid JSON. Do not include comments, explanations, or example content."
)


def init_gemini():
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    genai.configure(api_key=key)


def call_gemini(model: str, system_prompt: str, user_prompt: str):
    m = genai.GenerativeModel(
        model,   # FIX: model name must be positional, not keyword
        system_instruction=system_prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    last_err = None
    for a in range(3):
        try:
            resp = m.generate_content(user_prompt)
            return json.loads(resp.text)
        except Exception as e:
            last_err = e
            print(f"[WARN] Gemini attempt {a+1} failed: {e}")
    raise last_err



def build_prompt(question: str) -> str:
    return f"""
Create a JSON SCHEMA (template) based on the user's question.

QUESTION:
\"\"\"{question}\"\"\"


RULES:
- Infer meaningful sections.
- Use nested dicts when needed.
- All fields must have empty values: "" or [].
- MUST include a top-level field:
    "citations_or_quotes": []
- Do NOT add explanations.
- Output valid JSON only.
"""


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

    init_gemini()

    prompt = build_prompt(args.question)
    schema = call_gemini(args.model, SYSTEM_SCHEMA, prompt)

    # failsafe: ensure required field
    schema.setdefault("citations_or_quotes", [])

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Schema saved to {out_path}")


if __name__ == "__main__":
    main()
