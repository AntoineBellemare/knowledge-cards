#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, List
import os
import json
import shutil
import asyncio
import uuid
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Load .env file before other imports
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generate_card_UI import generate_schema
from gemini_template_creation import run_gemini_cards_with_progress

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR    = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT_DIR / "papers"
RESULTS_DIR = ROOT_DIR / "results"
CARDS_DIR = Path("../cards")
CARDS_DIR.mkdir(exist_ok=True)

# ---------- Background Job System ----------
# Job store with disk persistence for long-running tasks
JOBS_DIR = ROOT_DIR / ".jobs"  # Hidden directory for job persistence
JOBS_DIR.mkdir(exist_ok=True)
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

def _serialize_datetime(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def save_job_to_disk(job_id: str):
    """Persist job state to disk. Safe to call without lock (makes a copy)."""
    # Make a copy of the job data outside the lock to avoid blocking
    job_copy = None
    with jobs_lock:
        if job_id in jobs:
            job_copy = jobs[job_id].copy()
    
    if job_copy:
        job_file = JOBS_DIR / f"{job_id}.json"
        try:
            with job_file.open('w') as f:
                json.dump(job_copy, f, default=_serialize_datetime, indent=2)
        except Exception as e:
            print(f"[JOBS] Failed to save {job_id}: {e}")

def load_jobs_from_disk():
    """Load persisted jobs on startup."""
    global jobs
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            with job_file.open('r') as f:
                job_data = json.load(f)
                # Convert ISO datetime strings back to datetime objects
                if 'created_at' in job_data and isinstance(job_data['created_at'], str):
                    job_data['created_at'] = datetime.fromisoformat(job_data['created_at'])
                if 'last_activity' in job_data and isinstance(job_data['last_activity'], str):
                    job_data['last_activity'] = datetime.fromisoformat(job_data['last_activity'])
                jobs[job_data['id']] = job_data
                print(f"[JOBS] Loaded job {job_data['id']} from disk")
        except Exception as e:
            print(f"[JOBS] Failed to load {job_file.name}: {e}")

def cleanup_old_jobs():
    """Remove completed/failed jobs older than 24 hours, keep running jobs indefinitely."""
    cutoff = datetime.now() - timedelta(hours=24)
    with jobs_lock:
        old_jobs = []
        for jid, job in jobs.items():
            # Only clean up completed/failed jobs after 24 hours
            if job.get('status') in ['complete', 'error']:
                if job.get('created_at', datetime.now()) < cutoff:
                    old_jobs.append(jid)
            # For running jobs, check last activity (2 hours timeout for stalled jobs)
            elif job.get('status') == 'running':
                last_activity = job.get('last_activity', job.get('created_at', datetime.now()))
                if datetime.now() - last_activity > timedelta(hours=2):
                    print(f"[JOBS] Marking stalled job {jid} as error")
                    job['status'] = 'error'
                    job['error'] = 'Job stalled (no activity for 2 hours)'
                    save_job_to_disk(jid)
        
        for jid in old_jobs:
            # Delete from memory and disk
            del jobs[jid]
            job_file = JOBS_DIR / f"{jid}.json"
            if job_file.exists():
                job_file.unlink()
            print(f"[JOBS] Cleaned up old job {jid}")

def create_job(user_id: Optional[int], template_name: str, papers_folder: str) -> str:
    """Create a new job and return its ID."""
    job_id = str(uuid.uuid4())[:8]  # Short ID for easier debugging
    now = datetime.now()
    with jobs_lock:
        jobs[job_id] = {
            'id': job_id,
            'status': 'pending',
            'progress': 0,
            'stage': 'init',
            'message': 'Initializing...',
            'user_id': user_id,
            'template_name': template_name,
            'papers_folder': papers_folder,
            'result': None,
            'error': None,
            'created_at': now,
            'last_activity': now,
        }
    save_job_to_disk(job_id)
    return job_id

def update_job_progress(job_id: str, stage: str, current: int, total: int, message: str):
    """Update job progress. Auto-persists to disk for long-running jobs."""
    print(f"[update_job_progress] Acquiring lock for job {job_id}...")
    should_save = False
    with jobs_lock:
        print(f"[update_job_progress] Lock acquired for job {job_id}")
        if job_id in jobs:
            percent = int((current / total) * 100) if total > 0 else 0
            jobs[job_id].update({
                'stage': stage,
                'progress': percent,
                'message': message,
                'status': 'running',
                'last_activity': datetime.now(),
            })
            print(f"[update_job_progress] Updated job {job_id} to {percent}%")
            # Persist to disk less frequently to avoid blocking (every 10% or key milestones)
            should_save = (percent % 10 == 0 or stage in ['init', 'complete', 'error'])
            print(f"[update_job_progress] Should save to disk: {should_save}")
    # Save outside the lock to avoid blocking
    if should_save:
        print(f"[update_job_progress] Saving job {job_id} to disk...")
        save_job_to_disk(job_id)
        print(f"[update_job_progress] Saved job {job_id} to disk")

def complete_job(job_id: str, result: dict):
    """Mark job as complete with results."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update({
                'status': 'complete',
                'progress': 100,
                'stage': 'complete',
                'message': 'Complete!',
                'result': result,
                'last_activity': datetime.now(),
            })
    save_job_to_disk(job_id)

def fail_job(job_id: str, error: str):
    """Mark job as failed."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update({
                'status': 'error',
                'error': error,
                'message': f'Error: {error}',
                'last_activity': datetime.now(),
            })
    save_job_to_disk(job_id)

def get_job(job_id: str) -> Optional[dict]:
    """Get job status."""
    with jobs_lock:
        return jobs.get(job_id, {}).copy() if job_id in jobs else None


def save_run_to_database(
    user_id: int,
    template_id: Optional[int],
    template_name: str,
    papers_folder: str,
    model_used: str,
    jsonl_path: Path
) -> Optional[int]:
    """
    Save a completed run to the database as a Collection with Cards and MetaCard.
    Returns the collection_id if successful.
    """
    from database import SessionLocal
    from models import Collection, Card, MetaCard
    
    db = SessionLocal()
    try:
        # Create collection
        collection = Collection(
            user_id=user_id,
            name=f"{template_name} - {papers_folder}",
            template_id=template_id,
            papers_folder=papers_folder,
            model_used=model_used,
            is_public=False
        )
        db.add(collection)
        db.commit()
        db.refresh(collection)
        
        # Read JSONL and save individual cards
        jsonl_file = Path(jsonl_path)
        order_idx = 0
        print(f"[DB] Looking for JSONL file: {jsonl_file}")
        print(f"[DB] JSONL file absolute: {jsonl_file.resolve()}")
        print(f"[DB] JSONL file exists: {jsonl_file.exists()}")
        
        if jsonl_file.exists():
            print(f"[DB] JSONL file exists, reading cards...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"[DB] Read {len(lines)} lines from JSONL")
            
            for line in lines:
                if line.strip():
                    try:
                        card_data = json.loads(line)
                        # Source file from _file field
                        source_file = card_data.get("_file", "")
                        
                        card = Card(
                            collection_id=collection.id,
                            source_file=source_file,
                            content=card_data,
                            order_index=order_idx
                        )
                        db.add(card)
                        order_idx += 1
                        print(f"[DB] Added card {order_idx}: {source_file}")
                    except json.JSONDecodeError as e:
                        print(f"[DB] JSON decode error: {e}")
                        continue
        else:
            print(f"[DB] JSONL file NOT found: {jsonl_file}")
        
        # Check for meta card file
        meta_json_path = jsonl_file.parent / (jsonl_file.stem.replace('.jsonl', '') + '__meta.json')
        print(f"[DB] Looking for meta card: {meta_json_path}")
        if not meta_json_path.exists():
            # Try alternate naming - the stem already doesn't have .jsonl
            base = jsonl_file.stem
            meta_json_path = jsonl_file.parent / f"{base}__meta.json"
            print(f"[DB] Trying alternate path: {meta_json_path}")
        
        if meta_json_path.exists():
            print(f"[DB] Meta card file found, loading...")
            with open(meta_json_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
                meta_card = MetaCard(
                    collection_id=collection.id,
                    content=meta_data  # Fixed: was 'synthesis', should be 'content'
                )
                db.add(meta_card)
                collection.meta_card_count = 1
                print(f"[DB] Added meta card for collection {collection.id}")
        else:
            print(f"[DB] Meta card file NOT found")
        
        # Update card count on collection
        collection.card_count = order_idx
        
        db.commit()
        print(f"[DB] Saved collection {collection.id} with {order_idx} cards to database")
        return collection.id
        
    except Exception as e:
        db.rollback()
        print(f"[DB] Error saving to database: {e}")
        raise
    finally:
        db.close()

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

# ---------- Database initialization ----------
from database import init_db, engine
from db_routes import router as db_router

@app.on_event("startup")
async def startup_event():
    """Initialize database tables and load persisted jobs on startup."""
    if engine is not None:
        init_db()
        print("✅ Database connected and initialized")
    else:
        print("⚠️  Running without database (DATABASE_URL not set)")
    
    # Load persisted jobs from disk (survives server restarts/redeploys)
    load_jobs_from_disk()
    print(f"✅ Loaded {len(jobs)} persisted job(s) from disk")

# Include database routes
app.include_router(db_router)

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

# Serve static assets (images, favicon, etc.)
app.mount("/assets", StaticFiles(directory=BASE_DIR / "assets"), name="assets")

# Serve the frontend at root
@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "index.html")


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
async def run_gemini_stream(
    papers_subdir: str, 
    schema_name: Optional[str] = None, 
    template_id: Optional[int] = None,
    model: Optional[str] = None,
    user_id: Optional[int] = None
):
    """
    SSE endpoint for real-time progress during card generation.
    Use either schema_name (file-based) or template_id (database-based).
    If user_id is provided, saves the collection to the database.
    """
    # If template_id is provided, fetch schema from database
    schema_data = None
    template_name = schema_name
    template_question = None  # The original question that generated this template
    db_template_id = template_id
    
    if template_id:
        from database import SessionLocal
        from models import Template
        db = SessionLocal()
        try:
            template = db.query(Template).filter(Template.id == template_id).first()
            if template:
                schema_data = template.schema
                template_name = template.name
                template_question = template.question  # Get the original question
            else:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=404, content={"detail": "Template not found"})
        finally:
            db.close()
    
    if not schema_name and not template_id:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": "Either schema_name or template_id is required"})
    
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
                    schema_name=template_name,
                    papers_folder=papers_subdir,
                    model=model,
                    progress_callback=progress_callback,
                    schema_data=schema_data,  # Pass schema directly if from DB
                    template_question=template_question,  # Pass the original question
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
        keepalive_counter = 0
        while True:
            try:
                item = progress_queue.get(timeout=0.1)  # Check more frequently
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
                await asyncio.sleep(0.01)
                keepalive_counter = 0  # Reset on activity
            except queue.Empty:
                keepalive_counter += 1
                # Send keepalive every ~3 seconds (30 x 0.1s) to prevent proxy timeouts
                if keepalive_counter >= 30:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
                    await asyncio.sleep(0.01)
                    keepalive_counter = 0
        
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
            
            # Save to database if user is logged in
            collection_id = None
            if user_id:
                try:
                    collection_id = save_run_to_database(
                        user_id=user_id,
                        template_id=db_template_id,
                        template_name=template_name,
                        papers_folder=papers_subdir,
                        model_used=model or "gemini-2.5-flash-lite",
                        jsonl_path=jsonl_path
                    )
                except Exception as e:
                    print(f"[DB] Failed to save collection: {e}")
            
            yield f"data: {json.dumps({'type': 'complete', 'jsonl': to_rel(jsonl_path), 'csv': to_rel(csv_path), 'pdf': to_rel(pdf_path), 'schema_name': template_name, 'papers_folder': papers_subdir, 'collection_id': collection_id})}\n\n"
    
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


# ---------- Background Job Endpoints (for long-running tasks) ----------

class StartJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress: int
    stage: str
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None


@app.post("/jobs/start", response_model=StartJobResponse)
def start_build_job(
    papers_subdir: str,
    schema_name: Optional[str] = None,
    template_id: Optional[int] = None,
    model: Optional[str] = None,
    user_id: Optional[int] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Start a background job for card generation.
    Returns immediately with a job_id that can be polled for status.
    """
    # Cleanup old jobs periodically
    cleanup_old_jobs()
    
    # Fetch template if needed
    schema_data = None
    template_name = schema_name
    template_question = None
    db_template_id = template_id
    
    if template_id:
        from database import SessionLocal
        from models import Template
        db = SessionLocal()
        try:
            template = db.query(Template).filter(Template.id == template_id).first()
            if template:
                schema_data = template.schema
                template_name = template.name
                template_question = template.question
            else:
                raise HTTPException(status_code=404, detail="Template not found")
        finally:
            db.close()
    
    if not schema_name and not template_id:
        raise HTTPException(status_code=400, detail="Either schema_name or template_id is required")
    
    # Create job
    job_id = create_job(user_id, template_name, papers_subdir)
    print(f"[JOB {job_id}] Created job for {template_name} / {papers_subdir}")
    
    # Define the background task
    def run_job():
        print(f"[JOB {job_id}] Thread started!")
        try:
            print(f"[JOB {job_id}] About to call update_job_progress...")
            # Mark job as running immediately
            update_job_progress(job_id, "init", 0, 100, "Starting initialization...")
            print(f"[JOB {job_id}] update_job_progress completed")
            print(f"[JOB {job_id}] Starting background task")
            
            def progress_callback(stage: str, current: int, total: int, message: str):
                update_job_progress(job_id, stage, current, total, message)
            
            print(f"[JOB {job_id}] Calling run_gemini_cards_with_progress...")
            jsonl_path, csv_path, pdf_path = run_gemini_cards_with_progress(
                schema_name=template_name,
                papers_folder=papers_subdir,
                model=model,
                progress_callback=progress_callback,
                schema_data=schema_data,
                template_question=template_question,
            )
            print(f"[JOB {job_id}] run_gemini_cards_with_progress completed")
            print(f"[JOB {job_id}] run_gemini_cards_with_progress completed")
            
            def to_rel(p: Path) -> str:
                p = Path(p).resolve()
                try:
                    return str(p.relative_to(ROOT_DIR))
                except ValueError:
                    return f"results/{p.name}"
            
            # Save to database if user is logged in
            collection_id = None
            if user_id:
                try:
                    collection_id = save_run_to_database(
                        user_id=user_id,
                        template_id=db_template_id,
                        template_name=template_name,
                        papers_folder=papers_subdir,
                        model_used=model or "gemini-2.5-flash-lite",
                        jsonl_path=jsonl_path
                    )
                except Exception as e:
                    print(f"[JOB {job_id}] Failed to save collection: {e}")
            
            complete_job(job_id, {
                'jsonl': to_rel(jsonl_path),
                'csv': to_rel(csv_path),
                'pdf': to_rel(pdf_path),
                'schema_name': template_name,
                'papers_folder': papers_subdir,
                'collection_id': collection_id,
            })
            print(f"[JOB {job_id}] Completed successfully")
            
        except Exception as e:
            fail_job(job_id, str(e))
            print(f"[JOB {job_id}] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Start in a background thread (not FastAPI BackgroundTasks - those wait for response)
    print(f"[JOB {job_id}] Creating thread...")
    thread = threading.Thread(target=run_job, daemon=True, name=f"job-{job_id}")
    thread.start()
    print(f"[JOB {job_id}] Thread started: {thread.is_alive()}")
    
    return StartJobResponse(
        job_id=job_id,
        status="pending",
        message="Job started. Poll /jobs/{job_id} for status."
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Get the status of a background job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        id=job['id'],
        status=job['status'],
        progress=job['progress'],
        stage=job['stage'],
        message=job['message'],
        result=job.get('result'),
        error=job.get('error'),
    )


@app.get("/jobs")
def list_all_jobs():
    """List all jobs (for debugging)."""
    with jobs_lock:
        return {"jobs": list(jobs.values())}


@app.delete("/jobs/cleanup")
def manual_cleanup_jobs():
    """Manually trigger job cleanup."""
    cleanup_old_jobs()
    with jobs_lock:
        return {"remaining_jobs": len(jobs), "jobs": list(jobs.keys())}


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


# ---------- Text Generation Endpoint (for preprocessing) ----------

class TextRequest(BaseModel):
    apiKey: str
    prompt: str
    model: str = "gemini-2.5-flash"


@app.post("/generate_text")
async def generate_text(req: TextRequest):
    """Generate text using Google AI API (for LLM preprocessing)."""
    import httpx
    
    api_key = req.apiKey
    prompt = req.prompt
    model = req.model
    
    print(f"[TEXT] Model: {model}")
    print(f"[TEXT] Prompt length: {len(prompt)} chars")
    
    if not api_key:
        return {"success": False, "error": "API key required"}
    if not prompt:
        return {"success": False, "error": "Prompt required"}
    
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 4000
                }
            }
            
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            
            print(f"[TEXT] Response status: {response.status_code}")
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", f"API error: {response.status_code}")
                except:
                    error_msg = f"API error {response.status_code}: {response.text[:300]}"
                return {"success": False, "error": error_msg}
            
            result = response.json()
            candidates = result.get("candidates", [])
            
            if not candidates:
                return {"success": False, "error": "No response generated"}
            
            text_content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            if not text_content:
                return {"success": False, "error": "Empty response"}
            
            print(f"[TEXT] Generated {len(text_content)} chars")
            return {"success": True, "analysis": text_content}
            
    except Exception as e:
        print(f"[TEXT] Error: {e}")
        return {"success": False, "error": str(e)}


# ---------- Vision Generation Endpoint ----------

def calculate_image_cost(model: str, prompt_tokens: int, output_tokens: int, thoughts_tokens: int, resolution: str = "1024x1024") -> dict:
    """Calculate estimated cost based on model and token usage."""
    
    # Pricing per 1M tokens (as of Dec 2025)
    # Gemini 3 Pro Image: images are billed as output tokens
    # - 1K/2K (up to 2048px): 1120 tokens per image = $0.134 at $120/1M
    # - 4K (up to 4096px): 2000 tokens per image = $0.24 at $120/1M
    
    pricing = {
        "gemini-2.5-flash-image": {
            "input": 0.30,      # $0.30 / 1M tokens
            "output": 2.50,     # $2.50 / 1M tokens (text)
            "image_1k": 0.039,  # $0.039 per image (1290 tokens at $30/1M)
            "image_2k": 0.039,
            "image_4k": 0.039,  # Same price for flash
        },
        "gemini-2.0-flash": {
            "input": 0.10,
            "output": 0.40,
            "image_1k": 0.039,
            "image_2k": 0.039,
            "image_4k": 0.039,
        },
        "gemini-3-pro-image-preview": {
            "input": 2.00,       # $2.00 / 1M tokens
            "output": 12.00,     # $12.00 / 1M tokens (includes thinking)
            "image_1k": 0.134,   # 1120 tokens at $120/1M
            "image_2k": 0.134,   # Same as 1K
            "image_4k": 0.24,    # 2000 tokens at $120/1M
        },
        "nano-banana-pro-preview": {  # Same as gemini-3-pro-image-preview
            "input": 2.00,
            "output": 12.00,
            "image_1k": 0.134,
            "image_2k": 0.134,
            "image_4k": 0.24,
        },
        "imagen-4.0-generate-001": {
            "image_1k": 0.04,
            "image_2k": 0.04,
            "image_4k": 0.04,
        },
        "imagen-4.0-fast-generate-001": {
            "image_1k": 0.02,
            "image_2k": 0.02,
            "image_4k": 0.02,
        },
        "imagen-4.0-ultra-generate-001": {
            "image_1k": 0.06,
            "image_2k": 0.06,
            "image_4k": 0.06,
        },
        "imagen-3.0-generate-002": {
            "image_1k": 0.03,
            "image_2k": 0.03,
            "image_4k": 0.03,
        },
    }
    
    # Get pricing for model (default to gemini-2.5-flash-image pricing)
    model_pricing = pricing.get(model, pricing["gemini-2.5-flash-image"])
    
    # Determine image price based on resolution
    if "4096" in resolution or "4k" in resolution.lower():
        image_cost = model_pricing.get("image_4k", 0.039)
        res_label = "4K"
    elif "2048" in resolution or "2k" in resolution.lower():
        image_cost = model_pricing.get("image_2k", 0.039)
        res_label = "2K"
    else:
        image_cost = model_pricing.get("image_1k", 0.039)
        res_label = "1K"
    
    # Calculate costs
    input_cost = (prompt_tokens / 1_000_000) * model_pricing.get("input", 0)
    output_cost = ((output_tokens + thoughts_tokens) / 1_000_000) * model_pricing.get("output", 0)
    
    total_cost = input_cost + output_cost + image_cost
    
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "image_cost": round(image_cost, 6),
        "total_cost": round(total_cost, 6),
        "resolution": res_label,
        "model": model,
        "currency": "USD"
    }


class VisionRequest(BaseModel):
    apiKey: str
    prompt: str
    model: str = "gemini-2.5-flash-image"
    resolution: str = "1024x1024"  # "1024x1024", "2048x2048", "4096x4096"


@app.post("/generate_vision")
async def generate_vision(req: VisionRequest):
    """Generate an image using Google AI API (Gemini or Imagen models)."""
    import httpx
    
    api_key = req.apiKey
    prompt = req.prompt
    model = req.model
    resolution = req.resolution
    
    print(f"[VISION] Model: {model}")
    print(f"[VISION] Resolution: {resolution}")
    print(f"[VISION] Prompt length: {len(prompt)} chars")
    print(f"[VISION] Prompt preview: {prompt[:200]}...")
    
    if not api_key:
        return {"success": False, "error": "API key required"}
    if not prompt:
        return {"success": False, "error": "Prompt required"}
    
    # Longer timeout for image generation (can take 2-3 minutes for high-quality models)
    # Use longer timeout for higher quality models
    if "ultra" in model.lower():
        timeout_seconds = 300.0  # 5 minutes for ultra quality
    elif "4.0-generate" in model or "gemini-2" in model:
        timeout_seconds = 180.0  # 3 minutes for standard quality
    else:
        timeout_seconds = 120.0  # 2 minutes for fast models
    
    max_retries = 2
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"[VISION] Retry attempt {attempt + 1}/{max_retries}")
            
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                
                # Imagen models use predict method
                if model.startswith("imagen"):
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?key={api_key[:10]}..."
                    print(f"[VISION] Using Imagen predict method")
                    print(f"[VISION] URL: {url}")
                    
                    actual_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?key={api_key}"
                    
                    # Map resolution to Imagen imageSize parameter
                    imagen_size = "1K"  # default
                    if "2048" in resolution or "2k" in resolution.lower():
                        imagen_size = "2K"
                    
                    payload = {
                        "instances": [{"prompt": prompt}],
                        "parameters": {
                            "sampleCount": 1,
                            "aspectRatio": "16:9",  # Match the aspect ratio you like
                            "imageSize": imagen_size,
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
                    
                    # Imagen doesn't return token counts, just calculate image cost
                    cost_info = calculate_image_cost(model, 0, 0, 0, resolution)
                    
                    print(f"[VISION] SUCCESS - Got image, mime: {mime_type}, size: {len(image_base64)} chars")
                    print(f"[VISION] Estimated cost: ${cost_info['total_cost']:.4f}")
                    
                    return {
                        "success": True, 
                        "imageData": f"data:{mime_type};base64,{image_base64}",
                        "usage": {
                            "promptTokens": 0,
                            "outputTokens": 0,
                            "thoughtsTokens": 0,
                            "totalTokens": 0,
                        },
                        "cost": cost_info
                    }
                
                # Gemini models use generateContent method
                else:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key[:10]}..."
                    print(f"[VISION] Using Gemini generateContent method")
                    print(f"[VISION] URL: {url}")
                    
                    actual_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                    
                    # Build generation config
                    gen_config = {
                        "response_modalities": ["TEXT", "IMAGE"]
                    }
                    
                    # Add image_config for Gemini 3 Pro Image models (supports aspect_ratio and image_size)
                    if "gemini-3" in model or "nano-banana" in model:
                        image_size = "1K"
                        if "4096" in resolution or "4k" in resolution.lower():
                            image_size = "4K"
                        elif "2048" in resolution or "2k" in resolution.lower():
                            image_size = "2K"
                        
                        gen_config["image_config"] = {
                            "aspect_ratio": "16:9",
                            "image_size": image_size
                        }
                    
                    payload = {
                        "contents": [{
                            "parts": [{"text": f"Generate an image: {prompt}"}]
                        }],
                        "generation_config": gen_config
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
                    
                    # Extract usage metadata for cost calculation
                    usage = result.get("usageMetadata", {})
                    prompt_tokens = usage.get("promptTokenCount", 0)
                    output_tokens = usage.get("candidatesTokenCount", 0)
                    thoughts_tokens = usage.get("thoughtsTokenCount", 0)
                    total_tokens = usage.get("totalTokenCount", 0)
                    
                    # Calculate estimated cost based on model and resolution
                    cost_info = calculate_image_cost(model, prompt_tokens, output_tokens, thoughts_tokens, resolution)
                    
                    print(f"[VISION] SUCCESS - Got image, mime: {mime_type}, size: {len(image_base64)} chars")
                    print(f"[VISION] Usage - prompt: {prompt_tokens}, output: {output_tokens}, thoughts: {thoughts_tokens}, total: {total_tokens}")
                    print(f"[VISION] Estimated cost: ${cost_info['total_cost']:.4f} (resolution: {cost_info['resolution']})")
                    
                    return {
                        "success": True, 
                        "imageData": f"data:{mime_type};base64,{image_base64}",
                        "usage": {
                            "promptTokens": prompt_tokens,
                            "outputTokens": output_tokens,
                            "thoughtsTokens": thoughts_tokens,
                            "totalTokens": total_tokens,
                        },
                        "cost": cost_info
                    }
            
        except httpx.TimeoutException as e:
            last_error = f"Request timed out after {int(timeout_seconds)}s"
            print(f"[VISION] Timeout on attempt {attempt + 1}: {last_error}")
            if attempt < max_retries - 1:
                print(f"[VISION] Will retry...")
                continue
        except httpx.ConnectError as e:
            last_error = f"Connection error: {str(e)}"
            print(f"[VISION] Connection error on attempt {attempt + 1}: {last_error}")
            if attempt < max_retries - 1:
                continue
        except Exception as e:
            # Don't retry on other errors (API errors, etc.)
            return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}
    
    # All retries exhausted
    return {"success": False, "error": f"{last_error}. Please try again or use a faster model."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
