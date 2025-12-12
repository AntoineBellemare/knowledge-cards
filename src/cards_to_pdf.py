#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Schema-agnostic: Convert cards.jsonl into a structured PDF.

- Input:  JSONL file with one card per line (e.g. out/cards.jsonl)
- Output: Single PDF with one page per card, sections inferred from keys.

Usage:
  python cards_to_pdf_agnostic.py \
      --cards out/cards.jsonl \
      --out out/cards_report.pdf
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

from xml.sax.saxutils import escape as xml_escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    ListFlowable,
    ListItem,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT


# ------------- generic helpers -------------

def load_cards(jsonl_path: Path) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cards.append(json.loads(line))
    return cards


def nonempty(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return bool(x.strip())
    if isinstance(x, (list, tuple, set)):
        return any(nonempty(v) for v in x)
    if isinstance(x, dict):
        return any(nonempty(v) for v in x.values())
    return True


def bullet_list(items: List[str], style) -> ListFlowable:
    els: List[ListItem] = []
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        p = Paragraph(xml_escape(it), style)
        els.append(ListItem(p))
    return ListFlowable(
        els, bulletType="bullet", start="•", leftIndent=15, bulletFontSize=8
    )


def infer_title_from_card(card: Dict[str, Any]) -> str:
    """Extract paper title, preferring schema.metadata.title over citation.title.
    For meta-cards, generate a descriptive title."""
    # First try schema.metadata.title (the actual paper title)
    schema = card.get("schema") or {}
    if isinstance(schema, dict):
        metadata = schema.get("metadata") or {}
        if isinstance(metadata, dict):
            # Check if this is a meta-card (has papers_covered or similar)
            papers_covered = metadata.get("papers_covered") or metadata.get("papers_analyzed")
            if papers_covered:
                # This is a meta-card - generate descriptive title
                if isinstance(papers_covered, list):
                    n_papers = len(papers_covered)
                    return f"Meta-Analysis: Synthesis of {n_papers} Papers"
                elif isinstance(papers_covered, (int, str)):
                    return f"Meta-Analysis: Cross-Paper Synthesis"
            
            # Regular card - use title
            if metadata.get("title"):
                return str(metadata["title"])
    
    # Check if root level has meta-card indicators (card without schema wrapper)
    root_metadata = card.get("metadata") or {}
    if isinstance(root_metadata, dict):
        papers_covered = root_metadata.get("papers_covered") or root_metadata.get("papers_analyzed")
        if papers_covered:
            if isinstance(papers_covered, list):
                n_papers = len(papers_covered)
                return f"Meta-Analysis: Synthesis of {n_papers} Papers"
            return f"Meta-Analysis: Cross-Paper Synthesis"
        if root_metadata.get("title"):
            return str(root_metadata["title"])
    
    # Fallback to citation.title
    cit = card.get("citation") or {}
    if isinstance(cit, dict):
        if cit.get("title"):
            return str(cit["title"])
    
    if card.get("_file"):
        return str(card["_file"])
    return "Untitled paper"


def join_list_str(lst: List[Any], sep: str = ", ") -> str:
    vals = [str(x).strip() for x in lst if str(x).strip()]
    return sep.join(vals)


# ------------- top block: citation -------------

def render_citation_block(story, card, styles):
    """Render the paper header with title, authors, year, venue, etc."""
    title = infer_title_from_card(card)
    
    # Try to get metadata from schema.metadata first (preferred source)
    schema = card.get("schema") or {}
    metadata = schema.get("metadata") or {} if isinstance(schema, dict) else {}
    
    # Also check root-level metadata (for meta-cards)
    if not metadata or not isinstance(metadata, dict):
        metadata = card.get("metadata") or {}
    
    cit = card.get("citation") or {}
    
    # Check if this is a meta-card
    is_meta_card = False
    papers_covered = None
    if isinstance(metadata, dict):
        papers_covered = metadata.get("papers_covered") or metadata.get("papers_analyzed")
        if papers_covered:
            is_meta_card = True
    
    # Get authors - prefer schema.metadata.authors (skip for meta-cards)
    authors = ""
    if not is_meta_card:
        if isinstance(metadata, dict) and isinstance(metadata.get("authors"), list):
            authors = join_list_str(metadata["authors"], sep=", ")
        elif isinstance(cit, dict) and isinstance(cit.get("authors"), list):
            authors = join_list_str(cit["authors"], sep=", ")

    # Build metadata line (year · venue · DOI)
    meta_parts: List[str] = []
    
    if is_meta_card:
        # For meta-cards, show paper count
        if isinstance(papers_covered, list):
            meta_parts.append(f"Synthesizing {len(papers_covered)} papers")
    else:
        # Year - prefer schema.metadata.year
        year = metadata.get("year") if isinstance(metadata, dict) else None
        if not year and isinstance(cit, dict):
            year = cit.get("year")
        if year:
            meta_parts.append(str(year))
        
        # Venue - prefer schema.metadata.venue
        venue = metadata.get("venue") if isinstance(metadata, dict) else None
        if not venue and isinstance(cit, dict):
            venue = cit.get("venue")
        if venue:
            meta_parts.append(str(venue))
        
        # DOI - prefer schema.metadata.doi
        doi = metadata.get("doi") if isinstance(metadata, dict) else None
        if not doi and isinstance(cit, dict):
            doi = cit.get("doi")
        if doi:
            meta_parts.append(f"DOI: {doi}")

    meta = " · ".join(meta_parts) if meta_parts else ""

    # Render: Title, Authors, Meta line
    story.append(Paragraph(xml_escape(title), styles["Title"]))
    if authors:
        story.append(Paragraph(xml_escape(authors), styles["Italic"]))
    if meta:
        story.append(Paragraph(xml_escape(meta), styles["Small"]))
    
    # For meta-cards, list the papers covered
    if is_meta_card and isinstance(papers_covered, list) and len(papers_covered) > 0:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Papers Included:", styles["Heading2"]))
        for paper in papers_covered[:10]:  # Limit to first 10
            if isinstance(paper, dict):
                paper_title = paper.get("title", "Untitled")
                paper_authors = paper.get("authors", [])
                paper_year = paper.get("year", "")
                if isinstance(paper_authors, list):
                    paper_authors = ", ".join(str(a) for a in paper_authors[:3])
                    if len(paper.get("authors", [])) > 3:
                        paper_authors += " et al."
                paper_info = f"• {paper_title}"
                if paper_authors:
                    paper_info += f" ({paper_authors}"
                    if paper_year:
                        paper_info += f", {paper_year}"
                    paper_info += ")"
                elif paper_year:
                    paper_info += f" ({paper_year})"
                story.append(Paragraph(xml_escape(paper_info), styles["Small"]))
            elif isinstance(paper, str):
                story.append(Paragraph(xml_escape(f"• {paper}"), styles["Small"]))

    # Optional keywords - prefer schema.metadata.keywords
    keywords = metadata.get("keywords") if isinstance(metadata, dict) else None
    if not keywords and isinstance(cit, dict):
        keywords = cit.get("keywords")
    if keywords:
        kw = join_list_str(keywords, sep=", ")
        if kw:
            story.append(
                Paragraph(f"<b>Keywords:</b> {xml_escape(kw)}", styles["BodyText"])
            )

    story.append(Spacer(1, 0.5 * cm))


# ------------- recursive rendering -------------

def heading_for_level(key: str, level: int, styles) -> Paragraph:
    label = key.replace("_", " ").replace("-", " ").strip().title()
    if level <= 1:
        style_name = "Heading2"
    elif level == 2:
        style_name = "Heading3"
    else:
        style_name = "CardHeading4"  # custom style for deeper levels
    return Paragraph(xml_escape(label), styles[style_name])


def render_scalar(key: str, value: Any, story, styles, level: int):
    label = key.replace("_", " ").replace("-", " ").strip().title()
    txt = f"<b>{xml_escape(label)}:</b> {xml_escape(str(value))}"
    story.append(Paragraph(txt, styles["BodyText"]))
    story.append(Spacer(1, 0.1 * cm))


def render_list(key: str, value: List[Any], story, styles, level: int):
    if not value:
        return

    # Are all elements scalar-ish?
    all_scalar = all(
        (not isinstance(v, (dict, list, tuple, set))) for v in value
    )

    story.append(heading_for_level(key, level, styles))
    story.append(Spacer(1, 0.05 * cm))

    if all_scalar:
        # Simple bullet list
        story.append(bullet_list([str(v) for v in value], styles["BodyText"]))
        story.append(Spacer(1, 0.15 * cm))
        return

    # List of dicts / structured items
    for idx, item in enumerate(value):
        if isinstance(item, dict):
            story.append(
                Paragraph(f"<i>Item {idx + 1}</i>", styles["Small"])
            )
            story.append(Spacer(1, 0.05 * cm))
            for subk, subv in item.items():
                if not nonempty(subv):
                    continue
                render_any(subk, subv, story, styles, level + 1)
        else:
            # Fallback: scalar or list in unexpected shape
            render_any(f"{key}[{idx}]", item, story, styles, level + 1)

    story.append(Spacer(1, 0.15 * cm))


def render_dict(key: str, value: Dict[str, Any], story, styles, level: int):
    if not nonempty(value):
        return

    # section heading for this dict
    story.append(heading_for_level(key, level, styles))
    story.append(Spacer(1, 0.1 * cm))

    # preserve original JSON insertion order
    for subk, subv in value.items():
        if not nonempty(subv):
            continue
        render_any(subk, subv, story, styles, level + 1)


    story.append(Spacer(1, 0.15 * cm))


def render_any(key: str, value: Any, story, styles, level: int = 1):
    # Main generic dispatcher
    if value is None:
        return

    if isinstance(value, dict):
        render_dict(key, value, story, styles, level)
    elif isinstance(value, list):
        render_list(key, value, story, styles, level)
    else:
        if str(value).strip():
            render_scalar(key, value, story, styles, level)


# ------------- PDF builder -------------

def build_pdf(cards_path: Path, out_path: Path):
    cards = load_cards(cards_path)
    if not cards:
        print(f"No cards found in {cards_path}")
        return

    styles = getSampleStyleSheet()

    # small text style
    if "Small" not in styles:
        styles.add(
            ParagraphStyle(
                name="Small",
                parent=styles["Normal"],
                fontSize=9,
                leading=11,
                alignment=TA_LEFT,
            )
        )

    # Heading tweaks
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 4
    styles["Heading3"].spaceBefore = 4
    styles["Heading3"].spaceAfter = 2

    # custom deep-level heading
    if "CardHeading4" not in styles:
        styles.add(
            ParagraphStyle(
                name="CardHeading4",
                parent=styles["Normal"],
                fontSize=10,
                leading=12,
                spaceBefore=2,
                spaceAfter=1,
                leftIndent=6,
                textColor="black",
            )
        )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story: List[Any] = []

    for idx, card in enumerate(cards):
        if idx > 0:
            story.append(PageBreak())

        # 1) citation block if present
        render_citation_block(story, card, styles)

        # 2) everything else, schema-agnostic, in original JSON order
        for key in card.keys():
            if key in ("citation", "_file"):
                continue
            val = card.get(key)
            if not nonempty(val):
                continue
            render_any(key, val, story, styles, level=1)


    doc.build(story)
    print(f"Saved PDF to {out_path}")


def _get_pdf_styles():
    """Create and return the styles used for PDF generation."""
    styles = getSampleStyleSheet()

    # small text style
    if "Small" not in styles:
        styles.add(
            ParagraphStyle(
                name="Small",
                parent=styles["Normal"],
                fontSize=9,
                leading=11,
                alignment=TA_LEFT,
            )
        )

    # Heading tweaks
    styles["Heading2"].spaceBefore = 8
    styles["Heading2"].spaceAfter = 4
    styles["Heading3"].spaceBefore = 4
    styles["Heading3"].spaceAfter = 2

    # custom deep-level heading
    if "CardHeading4" not in styles:
        styles.add(
            ParagraphStyle(
                name="CardHeading4",
                parent=styles["Normal"],
                fontSize=10,
                leading=12,
                spaceBefore=2,
                spaceAfter=1,
                leftIndent=6,
                textColor="black",
            )
        )
    
    return styles


# ------------- Functions for in-memory PDF generation -------------

def create_cards_pdf_from_data(cards_data: List[Dict[str, Any]], out_path: str):
    """Generate PDF from a list of card dicts (in-memory, no file read)."""
    from pathlib import Path
    out_path = Path(out_path)
    
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = _get_pdf_styles()
    story: List[Any] = []

    for i, card in enumerate(cards_data):
        if i > 0:
            story.append(PageBreak())

        # 1) citation block if present
        render_citation_block(story, card, styles)

        # 2) everything else, schema-agnostic, in original JSON order
        for key in card.keys():
            if key in ("citation", "_file"):
                continue
            val = card.get(key)
            if not nonempty(val):
                continue
            render_any(key, val, story, styles, level=1)

    doc.build(story)
    print(f"[PDF] Generated cards PDF: {out_path}")


def create_metacard_pdf_from_data(metacard_data: Dict[str, Any], out_path: str):
    """Generate PDF from a single meta-card dict (in-memory, no file read)."""
    from pathlib import Path
    out_path = Path(out_path)
    
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = _get_pdf_styles()
    story: List[Any] = []

    # Render the meta-card
    render_citation_block(story, metacard_data, styles)

    for key in metacard_data.keys():
        if key in ("citation", "_file", "metadata"):
            continue
        val = metacard_data.get(key)
        if not nonempty(val):
            continue
        render_any(key, val, story, styles, level=1)
    
    # Render metadata separately for meta-cards (includes papers_covered)
    metadata = metacard_data.get("metadata")
    if metadata and nonempty(metadata):
        render_any("metadata", metadata, story, styles, level=1)

    doc.build(story)
    print(f"[PDF] Generated metacard PDF: {out_path}")


# ------------- CLI -------------

def main():
    ap = argparse.ArgumentParser(
        description="Schema-agnostic conversion of cards.jsonl to a sectioned PDF."
    )
    ap.add_argument(
        "--cards",
        type=str,
        default="cards.jsonl",
        help="Name of the cards.jsonl file (must be in ../results/).",
    )
    args = ap.parse_args()

    # Base results directory (always ../results relative to this script)
    base = Path(__file__).resolve().parent.parent / "results"
    base.mkdir(exist_ok=True)

    # Always read the cards file from ../results/
    cards_path = base / Path(args.cards).name

    if not cards_path.exists():
        raise FileNotFoundError(f"cards.jsonl not found: {cards_path}")

    # OUTPUT NAME = input filename (without extension) + ".pdf"
    pdf_name = cards_path.stem + ".pdf"
    out_path = base / pdf_name

    print(f"[INFO] Reading cards from: {cards_path}")
    print(f"[INFO] Writing PDF to:      {out_path}")

    build_pdf(cards_path, out_path)
    print("[DONE] PDF generated:", out_path)




if __name__ == "__main__":
    main()
