#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json, os
from generate_card_UI import generate_schema
from gemini_template_creation import run_gemini_cards

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR    = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT_DIR / "papers"
RESULTS_DIR = ROOT_DIR / "results"
CARDS_DIR = Path("../cards")
CARDS_DIR.mkdir(exist_ok=True)

class SaveTemplateRequest(BaseModel):
    id: str
    name: str
    question: str
    createdAt: str
    schema: dict


# ---------- Pydantic models ----------

class Pointers(BaseModel):
    emphasize_conceptual: bool = True
    include_empirical_methods: bool = True
    include_network_dimensions: bool = True
    include_ethics_context: bool = False


class SchemaRequest(BaseModel):
    question: str
    pointers: Optional[Pointers] = None


class SchemaResponse(BaseModel):
    schema: Dict[str, Any]
    
    
class SaveSchemaRequest(BaseModel):
    name: str            # e.g. "plant_cognition" or "plant_cognition.json"
    schema: Dict[str, Any]

class SaveSchemaResponse(BaseModel):
    ok: bool
    filename: str

class SchemasListResponse(BaseModel):
    schemas: List[str]   # ["plant_cognition.json", "networks.json", ...]

class PapersFoldersResponse(BaseModel):
    folders: List[str]   # ["plant_cognition", "other_project", ...]

class RunGeminiRequest(BaseModel):
    schema_name: str     # "plant_cognition.json"
    papers_subdir: str   # "plant_cognition"
    model: Optional[str] = None

class RunGeminiResponse(BaseModel):
    ok: bool
    schema_name: str
    papers_folder: str
    jsonl: str         # path relative to project root
    csv: str
    pdf: str




# ---------- FastAPI app ----------

app = FastAPI(title="Knowledge Cards API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/save_template")
def save_template(req: SaveTemplateRequest):
    filename = f"{req.name}.json"
    path = CARDS_DIR / filename
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(req.model_dump(), f, indent=2, ensure_ascii=False)
        return {"status": "ok", "file": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_templates")
def list_templates():
    files = list(CARDS_DIR.glob("*.json"))
    templates = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            templates.append({
                "id": data.get("id", f.stem),
                "name": data.get("name", f.stem),
                "question": data.get("question", ""),
                "createdAt": data.get("createdAt", ""),
                "schema": data.get("schema", {}),
            })
        except:
            continue
    return {"templates": templates}


@app.post("/schema_from_question", response_model=SchemaResponse)
def schema_from_question(req: SchemaRequest) -> SchemaResponse:
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        schema = generate_schema(
            question=q,
            model="gemini-2.5-flash-lite",
            pointers=req.pointers.dict() if req.pointers else {},
            ensure_citations_field=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating schema: {e}")

    return SchemaResponse(schema=schema)

@app.post("/save_schema", response_model=SaveSchemaResponse)
def save_schema(req: SaveSchemaRequest) -> SaveSchemaResponse:
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Schema name must not be empty.")

    if not name.endswith(".json"):
        name = name + ".json"

    path = CARDS_DIR / name
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(req.schema, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save schema: {e}")

    return SaveSchemaResponse(ok=True, filename=name)

@app.get("/schemas", response_model=SchemasListResponse)
def list_schemas() -> SchemasListResponse:
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in CARDS_DIR.glob("*.json"))
    return SchemasListResponse(schemas=names)

@app.get("/papers_folders", response_model=PapersFoldersResponse)
def list_papers_folders() -> PapersFoldersResponse:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    folders = sorted(p.name for p in PAPERS_DIR.iterdir() if p.is_dir())
    return PapersFoldersResponse(folders=folders)

@app.post("/run_gemini", response_model=RunGeminiResponse)
def run_gemini(req: RunGeminiRequest) -> RunGeminiResponse:
    try:
        jsonl_path, csv_path, pdf_path = run_gemini_cards(
            schema_name=req.schema_name,
            papers_folder=req.papers_subdir,
            model=req.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running Gemini cards: {e}")

    # Normalize to Path objects
    jsonl_path = Path(jsonl_path)
    csv_path   = Path(csv_path)
    pdf_path   = Path(pdf_path)

    # Anchor relative paths on ROOT_DIR if needed
    def to_rel(p: Path) -> str:
        p = p.resolve()
        try:
            return str(p.relative_to(ROOT_DIR))
        except ValueError:
            return f"results/{p.name}"


    rel_jsonl = to_rel(jsonl_path)
    rel_csv   = to_rel(csv_path)
    rel_pdf   = to_rel(pdf_path)

    return RunGeminiResponse(
        ok=True,
        schema_name=req.schema_name,
        papers_folder=req.papers_subdir,
        jsonl=rel_jsonl,
        csv=rel_csv,
        pdf=rel_pdf,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=9000, reload=True)
