#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, List
import os
import json
import shutil
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from generate_card_UI import generate_schema
from gemini_template_creation import run_gemini_cards_with_progress

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
    model: Optional[str] = None
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

# CORS: Allow local development and production domains
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://localhost:5500",      # VS Code Live Server
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "https://knowledge-cards.pages.dev",  # <- Update with your Cloudflare Pages URL
]

# Allow all origins for simplicity (API is public anyway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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

    model = req.model or "gemini-2.5-flash-lite"
    print(f"[API] Generating schema with model: {model}")
    
    try:
        schema = generate_schema(
            question=q,
            model=model,
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


@app.get("/run_gemini_stream")
async def run_gemini_stream(schema_name: str, papers_subdir: str, model: Optional[str] = None):
    """
    SSE endpoint for real-time progress during card generation.
    """
    async def event_generator():
        import queue
        import threading
        
        progress_queue = queue.Queue()
        result_holder = {"result": None, "error": None}
        
        def progress_callback(stage: str, current: int, total: int, message: str):
            progress_queue.put({
                "stage": stage,
                "current": current,
                "total": total,
                "message": message,
                "percent": int((current / total) * 100) if total > 0 else 0
            })
        
        def run_task():
            try:
                result = run_gemini_cards_with_progress(
                    schema_name=schema_name,
                    papers_folder=papers_subdir,
                    model=model,
                    progress_callback=progress_callback,
                )
                result_holder["result"] = result
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                progress_queue.put(None)  # Signal completion
        
        # Start background thread
        thread = threading.Thread(target=run_task)
        thread.start()
        
        # Stream progress events
        while True:
            try:
                item = progress_queue.get(timeout=0.5)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
                await asyncio.sleep(0.01)  # Small delay to help with proxy buffering
            except queue.Empty:
                # Send keepalive to keep connection alive and help with buffering
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                await asyncio.sleep(0.01)
        
        thread.join()
        
        # Send final result
        if result_holder["error"]:
            yield f"data: {json.dumps({'type': 'error', 'message': result_holder['error']})}\n\n"
        else:
            jsonl_path, csv_path, pdf_path = result_holder["result"]
            
            def to_rel(p: Path) -> str:
                p = Path(p).resolve()
                try:
                    return str(p.relative_to(ROOT_DIR))
                except ValueError:
                    return f"results/{p.name}"
            
            yield f"data: {json.dumps({'type': 'complete', 'jsonl': to_rel(jsonl_path), 'csv': to_rel(csv_path), 'pdf': to_rel(pdf_path), 'schema_name': schema_name, 'papers_folder': papers_subdir})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable nginx/proxy buffering
            "Transfer-Encoding": "chunked",
        }
    )


@app.post("/run_gemini", response_model=RunGeminiResponse)
def run_gemini(req: RunGeminiRequest) -> RunGeminiResponse:
    """Legacy non-streaming endpoint (kept for compatibility)."""
    from gemini_template_creation import run_gemini_cards
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


# ---------- File Upload / Download Endpoints ----------

@app.get("/list_results")
def list_results():
    """List all result files available for download."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in sorted(RESULTS_DIR.iterdir()):
        if f.is_file():
            files.append({
                "name": f.name,
                "type": f.suffix.lstrip("."),
                "size": f.stat().st_size,
            })
    return {"results": files}


@app.post("/upload_papers/{folder_name}")
async def upload_papers(folder_name: str, files: List[UploadFile] = File(...)):
    """Upload PDF papers to a specific folder."""
    folder_path = PAPERS_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    uploaded = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        dest = folder_path / file.filename
        with dest.open("wb") as f:
            content = await file.read()
            f.write(content)
        uploaded.append(file.filename)
    
    return {"ok": True, "folder": folder_name, "uploaded": uploaded}


@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """Download a result file (jsonl, csv, pdf)."""
    if file_type not in ["results", "cards"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    if file_type == "results":
        file_path = RESULTS_DIR / filename
    else:
        file_path = CARDS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    media_type = "application/octet-stream"
    if filename.endswith(".pdf"):
        media_type = "application/pdf"
    elif filename.endswith(".csv"):
        media_type = "text/csv"
    elif filename.endswith(".jsonl") or filename.endswith(".json"):
        media_type = "application/json"
    
    return FileResponse(file_path, media_type=media_type, filename=filename)


@app.delete("/papers/{folder_name}")
async def delete_papers_folder(folder_name: str):
    """Delete a papers folder."""
    folder_path = PAPERS_DIR / folder_name
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    shutil.rmtree(folder_path)
    return {"ok": True, "deleted": folder_name}


# ---------- Vision Generation Endpoint ----------

class VisionRequest(BaseModel):
    apiKey: str
    prompt: str
    model: str = "gemini-2.5-flash-image"


@app.post("/generate_vision")
async def generate_vision(req: VisionRequest):
    """Generate an image using Google AI API (Gemini or Imagen models)."""
    import httpx
    
    api_key = req.apiKey
    prompt = req.prompt
    model = req.model
    
    print(f"[VISION] Model: {model}")
    print(f"[VISION] Prompt length: {len(prompt)} chars")
    print(f"[VISION] Prompt preview: {prompt[:200]}...")
    
    if not api_key:
        return {"success": False, "error": "API key required"}
    if not prompt:
        return {"success": False, "error": "Prompt required"}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            
            # Imagen models use predict method
            if model.startswith("imagen"):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?key={api_key[:10]}..."
                print(f"[VISION] Using Imagen predict method")
                print(f"[VISION] URL: {url}")
                
                actual_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?key={api_key}"
                payload = {
                    "instances": [{"prompt": prompt}],
                    "parameters": {
                        "sampleCount": 1,
                        "aspectRatio": "1:1",
                        "personGeneration": "DONT_ALLOW"
                    }
                }
                print(f"[VISION] Payload: {json.dumps(payload, indent=2)[:500]}")
                
                response = await client.post(actual_url, json=payload, headers={"Content-Type": "application/json"})
                
                print(f"[VISION] Response status: {response.status_code}")
                print(f"[VISION] Response body preview: {response.text[:500]}")
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", f"API error: {response.status_code}")
                    except:
                        error_msg = f"API error {response.status_code}: {response.text[:300]}"
                    return {"success": False, "error": error_msg}
                
                result = response.json()
                predictions = result.get("predictions", [])
                print(f"[VISION] Got {len(predictions)} predictions")
                
                if not predictions:
                    print(f"[VISION] Full response: {json.dumps(result, indent=2)[:1000]}")
                    return {"success": False, "error": "No image generated by Imagen"}
                
                image_base64 = predictions[0].get("bytesBase64Encoded")
                mime_type = predictions[0].get("mimeType", "image/png")
                
                if not image_base64:
                    print(f"[VISION] Prediction[0] keys: {predictions[0].keys()}")
                    return {"success": False, "error": "No image data in Imagen response"}
                
                print(f"[VISION] SUCCESS - Got image, mime: {mime_type}, size: {len(image_base64)} chars")
                return {"success": True, "imageData": f"data:{mime_type};base64,{image_base64}"}
            
            # Gemini models use generateContent method
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key[:10]}..."
                print(f"[VISION] Using Gemini generateContent method")
                print(f"[VISION] URL: {url}")
                
                actual_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{
                        "parts": [{"text": f"Generate an image: {prompt}"}]
                    }],
                    "generationConfig": {
                        "responseModalities": ["TEXT", "IMAGE"]
                    }
                }
                print(f"[VISION] Payload: {json.dumps(payload, indent=2)[:500]}")
                
                response = await client.post(actual_url, json=payload, headers={"Content-Type": "application/json"})
                
                print(f"[VISION] Response status: {response.status_code}")
                print(f"[VISION] Response body preview: {response.text[:500]}")
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", f"API error: {response.status_code}")
                    except:
                        error_msg = f"API error {response.status_code}: {response.text[:300]}"
                    return {"success": False, "error": error_msg}
                
                result = response.json()
                candidates = result.get("candidates", [])
                print(f"[VISION] Got {len(candidates)} candidates")
                
                if not candidates:
                    print(f"[VISION] Full response: {json.dumps(result, indent=2)[:1000]}")
                    return {"success": False, "error": "No response generated"}
                
                parts = candidates[0].get("content", {}).get("parts", [])
                print(f"[VISION] Got {len(parts)} parts")
                
                # Find the image part
                image_part = None
                for i, part in enumerate(parts):
                    print(f"[VISION] Part {i} keys: {part.keys()}")
                    if "inlineData" in part:
                        image_part = part["inlineData"]
                        break
                
                if not image_part:
                    text_response = " ".join([p.get("text", "") for p in parts if "text" in p])
                    print(f"[VISION] No image found. Text response: {text_response[:300]}")
                    return {"success": False, "error": f"No image generated. Response: {text_response[:200]}"}
                
                image_base64 = image_part.get("data")
                mime_type = image_part.get("mimeType", "image/png")
                
                if not image_base64:
                    print(f"[VISION] inlineData keys: {image_part.keys()}")
                    return {"success": False, "error": "No image data in response"}
                
                print(f"[VISION] SUCCESS - Got image, mime: {mime_type}, size: {len(image_base64)} chars")
                return {"success": True, "imageData": f"data:{mime_type};base64,{image_base64}"}
            
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out (120s). Try again."}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
