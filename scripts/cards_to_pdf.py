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
    cit = card.get("citation") or {}
    title = infer_title_from_card(card)

    authors = ""
    if isinstance(cit, dict) and isinstance(cit.get("authors"), list):
        authors = join_list_str(cit["authors"], sep=", ")

    meta_parts: List[str] = []
    if isinstance(cit, dict):
        if cit.get("year"):
            meta_parts.append(str(cit["year"]))
        if cit.get("venue"):
            meta_parts.append(str(cit["venue"]))
        if cit.get("doi"):
            meta_parts.append(f"DOI: {cit['doi']}")

    meta = " · ".join(meta_parts) if meta_parts else ""

    story.append(Paragraph(xml_escape(title), styles["Title"]))
    if authors:
        story.append(Paragraph(xml_escape(authors), styles["Italic"]))
    if meta:
        story.append(Paragraph(xml_escape(meta), styles["Small"]))

    # optional keywords
    if isinstance(cit, dict) and cit.get("keywords"):
        kw = join_list_str(cit["keywords"], sep=", ")
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

    # stable-ish order: alphabetical
    for subk in sorted(value.keys()):
        subv = value[subk]
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

        # 2) everything else, schema-agnostic
        # We skip keys already handled: citation + _file
        for key in sorted(card.keys()):
            if key in ("citation", "_file"):
                continue
            val = card[key]
            if not nonempty(val):
                continue
            render_any(key, val, story, styles, level=1)

    doc.build(story)
    print(f"Saved PDF to {out_path}")


# ------------- CLI -------------

def main():
    ap = argparse.ArgumentParser(
        description="Schema-agnostic conversion of cards.jsonl to a sectioned PDF."
    )
    ap.add_argument(
        "--cards",
        type=str,
        default="out/cards_plant.jsonl",
        help="Path to cards.jsonl (one JSON card per line).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="out/cards_plant_report.pdf",
        help="Output PDF path.",
    )
    args = ap.parse_args()

    cards_path = Path(args.cards)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_pdf(cards_path, out_path)


if __name__ == "__main__":
    main()
