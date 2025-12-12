"""
Database API routes for Knowledge Cards.

Provides endpoints for:
- User management (anonymous & email-based)
- Template CRUD with collective database
- Collection management
- Cards and Meta Cards storage
"""
from typing import Optional, List
from datetime import datetime
import uuid
import json
import tempfile
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from pydantic import BaseModel, Field

from database import get_db, engine
from models import KCUser, Template, Collection, Card, MetaCard, TemplateLike


# ============================================================
# Pydantic Schemas
# ============================================================

class UserCreate(BaseModel):
    username: Optional[str] = None  # Unique identifier chosen by user
    email: Optional[str] = None
    display_name: Optional[str] = None

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    display_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: Optional[str]
    email: Optional[str]
    display_name: Optional[str]
    anonymous_id: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class TemplateCreate(BaseModel):
    name: str
    description: Optional[str] = None
    question: Optional[str] = None
    schema: dict
    is_public: bool = False

class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    question: Optional[str] = None
    schema: Optional[dict] = None
    is_public: Optional[bool] = None

class TemplateResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    question: Optional[str]
    schema: dict
    version: int
    is_public: bool
    is_featured: bool
    is_system: bool
    use_count: int
    fork_count: int
    created_at: datetime
    updated_at: datetime
    owner_name: Optional[str] = None
    is_liked: bool = False
    like_count: int = 0
    
    class Config:
        from_attributes = True


class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    template_id: Optional[int] = None
    papers_folder: Optional[str] = None
    model_used: Optional[str] = None
    is_public: bool = False

class CollectionResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    template_id: Optional[int]
    papers_folder: Optional[str]
    model_used: Optional[str]
    is_public: bool
    card_count: int
    meta_card_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class CardCreate(BaseModel):
    content: dict
    source_file: Optional[str] = None
    source_page: Optional[int] = None
    order_index: int = 0

class CardResponse(BaseModel):
    id: int
    collection_id: int
    content: dict
    source_file: Optional[str]
    source_page: Optional[int]
    order_index: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class MetaCardCreate(BaseModel):
    content: dict
    meta_type: Optional[str] = None
    source_card_ids: Optional[List[int]] = None

class MetaCardResponse(BaseModel):
    id: int
    collection_id: int
    content: dict
    meta_type: Optional[str]
    source_card_ids: Optional[List[int]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class BulkCardsCreate(BaseModel):
    """For saving multiple cards at once after generation"""
    cards: List[CardCreate]
    meta_cards: Optional[List[MetaCardCreate]] = None


# ============================================================
# Router
# ============================================================

router = APIRouter(prefix="/db", tags=["database"])


# ============================================================
# User Endpoints
# ============================================================

@router.post("/users/anonymous", response_model=UserResponse)
def create_anonymous_user(db: Session = Depends(get_db)):
    """Create an anonymous user with a unique ID (stored in browser localStorage)."""
    anonymous_id = str(uuid.uuid4())
    user = KCUser(anonymous_id=anonymous_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/users", response_model=UserResponse)
def create_or_get_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Create a user or get existing one.
    
    Priority for finding existing user:
    1. username (if provided)
    2. email (if provided)
    
    If neither exists, creates a new user with optional anonymous_id.
    """
    # Check by username first
    if user_data.username:
        existing = db.query(KCUser).filter(KCUser.username == user_data.username).first()
        if existing:
            return existing
    
    # Check by email
    if user_data.email:
        existing = db.query(KCUser).filter(KCUser.email == user_data.email).first()
        if existing:
            # Update username if not set
            if user_data.username and not existing.username:
                existing.username = user_data.username
                db.commit()
                db.refresh(existing)
            return existing
    
    # Create new user
    user = KCUser(
        username=user_data.username,
        email=user_data.email,
        display_name=user_data.display_name or user_data.username,
        anonymous_id=str(uuid.uuid4()) if not user_data.email else None
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/users/by-username/{username}", response_model=UserResponse)
def get_user_by_username(username: str, db: Session = Depends(get_db)):
    """Get user by username. This is the primary way to identify users."""
    user = db.query(KCUser).filter(KCUser.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/users/check-username/{username}")
def check_username_available(username: str, db: Session = Depends(get_db)):
    """Check if a username is available."""
    existing = db.query(KCUser).filter(KCUser.username == username).first()
    return {"username": username, "available": existing is None}


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by numeric ID."""
    user = db.query(KCUser).filter(KCUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, update: UserUpdate, db: Session = Depends(get_db)):
    """Update user profile (username, email, display_name)."""
    user = db.query(KCUser).filter(KCUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check username uniqueness if changing
    if update.username and update.username != user.username:
        existing = db.query(KCUser).filter(KCUser.username == update.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
        user.username = update.username
    
    # Check email uniqueness if changing
    if update.email and update.email != user.email:
        existing = db.query(KCUser).filter(KCUser.email == update.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        user.email = update.email
    
    if update.display_name:
        user.display_name = update.display_name
    
    db.commit()
    db.refresh(user)
    return user


@router.get("/users/anonymous/{anonymous_id}", response_model=UserResponse)
def get_user_by_anonymous_id(anonymous_id: str, db: Session = Depends(get_db)):
    """Get user by anonymous ID (stored in browser localStorage)."""
    user = db.query(KCUser).filter(KCUser.anonymous_id == anonymous_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/users/anonymous/{anonymous_id}/claim", response_model=UserResponse)
def claim_anonymous_user(
    anonymous_id: str, 
    username: str,
    email: Optional[str] = None,
    display_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Convert an anonymous user to a registered user by adding a username.
    This 'claims' the anonymous account and all its data.
    """
    user = db.query(KCUser).filter(KCUser.anonymous_id == anonymous_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Anonymous user not found")
    
    # Check username availability
    existing = db.query(KCUser).filter(KCUser.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    user.username = username
    user.email = email
    user.display_name = display_name or username
    # Keep anonymous_id for backwards compatibility
    
    db.commit()
    db.refresh(user)
    return user


# ============================================================
# Template Endpoints
# ============================================================

@router.post("/templates", response_model=TemplateResponse)
def create_template(
    template: TemplateCreate,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Create a new template."""
    db_template = Template(
        user_id=user_id,
        name=template.name,
        description=template.description,
        question=template.question,
        schema=template.schema,
        is_public=template.is_public
    )
    db.add(db_template)
    db.commit()
    db.refresh(db_template)
    
    return _template_to_response(db_template, db, user_id)


@router.get("/templates", response_model=List[TemplateResponse])
def list_templates(
    user_id: Optional[int] = None,
    include_public: bool = True,
    include_system: bool = True,
    search: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List templates. Includes:
    - User's own templates (if user_id provided)
    - Public templates from collective (if include_public)
    - System/built-in templates (if include_system)
    """
    query = db.query(Template)
    
    conditions = []
    if user_id:
        conditions.append(Template.user_id == user_id)
    if include_public:
        conditions.append(Template.is_public == True)
    if include_system:
        conditions.append(Template.is_system == True)
    
    if conditions:
        from sqlalchemy import or_
        query = query.filter(or_(*conditions))
    
    if search:
        query = query.filter(
            Template.name.ilike(f"%{search}%") | 
            Template.description.ilike(f"%{search}%")
        )
    
    query = query.order_by(desc(Template.is_featured), desc(Template.use_count), desc(Template.created_at))
    templates = query.offset(offset).limit(limit).all()
    
    return [_template_to_response(t, db, user_id) for t in templates]


@router.get("/templates/collective", response_model=List[TemplateResponse])
def list_collective_templates(
    featured_only: bool = False,
    sort_by: str = Query("popular", regex="^(popular|recent|name)$"),
    search: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Browse the collective template library.
    """
    query = db.query(Template).filter(Template.is_public == True)
    
    if featured_only:
        query = query.filter(Template.is_featured == True)
    
    if search:
        query = query.filter(
            Template.name.ilike(f"%{search}%") | 
            Template.description.ilike(f"%{search}%") |
            Template.question.ilike(f"%{search}%")
        )
    
    if sort_by == "popular":
        query = query.order_by(desc(Template.use_count), desc(Template.fork_count))
    elif sort_by == "recent":
        query = query.order_by(desc(Template.created_at))
    else:
        query = query.order_by(Template.name)
    
    templates = query.offset(offset).limit(limit).all()
    return [_template_to_response(t, db, user_id) for t in templates]


@router.get("/templates/{template_id}", response_model=TemplateResponse)
def get_template(
    template_id: int,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get a specific template by ID."""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Check access
    if not template.is_public and not template.is_system and template.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return _template_to_response(template, db, user_id)


@router.put("/templates/{template_id}", response_model=TemplateResponse)
def update_template(
    template_id: int,
    update: TemplateUpdate,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Update a template. Only the owner can update."""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    if template.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this template")
    
    for key, value in update.dict(exclude_unset=True).items():
        setattr(template, key, value)
    
    template.version += 1
    db.commit()
    db.refresh(template)
    
    return _template_to_response(template, db, user_id)


@router.delete("/templates/{template_id}")
def delete_template(template_id: int, user_id: int, db: Session = Depends(get_db)):
    """Delete a template. Only the owner can delete."""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    if template.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this template")
    
    db.delete(template)
    db.commit()
    return {"ok": True}


@router.post("/templates/{template_id}/fork", response_model=TemplateResponse)
def fork_template(
    template_id: int,
    user_id: int,
    new_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Fork a public template to your own library."""
    original = db.query(Template).filter(Template.id == template_id).first()
    if not original:
        raise HTTPException(status_code=404, detail="Template not found")
    if not original.is_public and not original.is_system and original.user_id != user_id:
        raise HTTPException(status_code=403, detail="Cannot fork private template")
    
    forked = Template(
        user_id=user_id,
        name=new_name or f"{original.name} (fork)",
        description=original.description,
        question=original.question,
        schema=original.schema,
        parent_id=original.id,
        is_public=False
    )
    db.add(forked)
    
    # Increment fork count on original
    original.fork_count += 1
    
    db.commit()
    db.refresh(forked)
    
    return _template_to_response(forked, db, user_id)


@router.post("/templates/{template_id}/contribute")
def contribute_template(
    template_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Make a template public to contribute to the collective library."""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    if template.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    template.is_public = True
    db.commit()
    
    return {"ok": True, "message": "Template is now public in the collective library"}


@router.post("/templates/{template_id}/like")
def like_template(template_id: int, user_id: int, db: Session = Depends(get_db)):
    """Like/favorite a template."""
    existing = db.query(TemplateLike).filter(
        TemplateLike.template_id == template_id,
        TemplateLike.user_id == user_id
    ).first()
    
    if existing:
        return {"ok": True, "action": "already_liked"}
    
    like = TemplateLike(template_id=template_id, user_id=user_id)
    db.add(like)
    db.commit()
    
    return {"ok": True, "action": "liked"}


@router.delete("/templates/{template_id}/like")
def unlike_template(template_id: int, user_id: int, db: Session = Depends(get_db)):
    """Remove like from a template."""
    like = db.query(TemplateLike).filter(
        TemplateLike.template_id == template_id,
        TemplateLike.user_id == user_id
    ).first()
    
    if like:
        db.delete(like)
        db.commit()
    
    return {"ok": True, "action": "unliked"}


@router.post("/templates/{template_id}/use")
def record_template_use(template_id: int, db: Session = Depends(get_db)):
    """Record that a template was used (for popularity tracking)."""
    template = db.query(Template).filter(Template.id == template_id).first()
    if template:
        template.use_count += 1
        db.commit()
    return {"ok": True}


# ============================================================
# Collection Endpoints
# ============================================================

@router.post("/collections", response_model=CollectionResponse)
def create_collection(
    collection: CollectionCreate,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Create a new collection."""
    db_collection = Collection(
        user_id=user_id,
        name=collection.name,
        description=collection.description,
        template_id=collection.template_id,
        papers_folder=collection.papers_folder,
        model_used=collection.model_used,
        is_public=collection.is_public
    )
    db.add(db_collection)
    db.commit()
    db.refresh(db_collection)
    return db_collection


@router.get("/collections", response_model=List[CollectionResponse])
def list_collections(
    user_id: Optional[int] = None,
    include_public: bool = False,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List collections for a user."""
    query = db.query(Collection)
    
    if user_id:
        if include_public:
            from sqlalchemy import or_
            query = query.filter(or_(
                Collection.user_id == user_id,
                Collection.is_public == True
            ))
        else:
            query = query.filter(Collection.user_id == user_id)
    elif include_public:
        query = query.filter(Collection.is_public == True)
    
    collections = query.order_by(desc(Collection.created_at)).offset(offset).limit(limit).all()
    
    # Fix: Recalculate card counts for collections with stale counts
    for c in collections:
        if c.card_count == 0:
            actual_card_count = len(c.cards) if c.cards else 0
            actual_meta_count = len(c.meta_cards) if c.meta_cards else 0
            if actual_card_count > 0 or actual_meta_count > 0:
                c.card_count = actual_card_count
                c.meta_card_count = actual_meta_count
                db.add(c)
    db.commit()
    
    return collections


@router.get("/collections/{collection_id}", response_model=CollectionResponse)
def get_collection(collection_id: int, db: Session = Depends(get_db)):
    """Get a collection by ID."""
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


@router.delete("/collections/{collection_id}")
def delete_collection(collection_id: int, user_id: int, db: Session = Depends(get_db)):
    """Delete a collection and all its cards."""
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if collection.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    db.delete(collection)
    db.commit()
    return {"ok": True}


# ============================================================
# Card Endpoints
# ============================================================

@router.post("/collections/{collection_id}/cards", response_model=CardResponse)
def create_card(
    collection_id: int,
    card: CardCreate,
    db: Session = Depends(get_db)
):
    """Add a single card to a collection."""
    db_card = Card(
        collection_id=collection_id,
        content=card.content,
        source_file=card.source_file,
        source_page=card.source_page,
        order_index=card.order_index
    )
    db.add(db_card)
    
    # Update collection card count
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if collection:
        collection.card_count += 1
    
    db.commit()
    db.refresh(db_card)
    return db_card


@router.post("/collections/{collection_id}/cards/bulk")
def create_cards_bulk(
    collection_id: int,
    bulk: BulkCardsCreate,
    db: Session = Depends(get_db)
):
    """Add multiple cards (and optionally meta cards) at once."""
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # Add cards
    for i, card_data in enumerate(bulk.cards):
        db_card = Card(
            collection_id=collection_id,
            content=card_data.content,
            source_file=card_data.source_file,
            source_page=card_data.source_page,
            order_index=card_data.order_index or i
        )
        db.add(db_card)
    
    # Add meta cards if provided
    if bulk.meta_cards:
        for meta_data in bulk.meta_cards:
            db_meta = MetaCard(
                collection_id=collection_id,
                content=meta_data.content,
                meta_type=meta_data.meta_type,
                source_card_ids=meta_data.source_card_ids
            )
            db.add(db_meta)
    
    # Update counts
    collection.card_count = len(bulk.cards)
    collection.meta_card_count = len(bulk.meta_cards) if bulk.meta_cards else 0
    
    db.commit()
    
    return {
        "ok": True,
        "cards_created": len(bulk.cards),
        "meta_cards_created": len(bulk.meta_cards) if bulk.meta_cards else 0
    }


@router.get("/collections/{collection_id}/cards", response_model=List[CardResponse])
def list_cards(
    collection_id: int,
    limit: int = Query(100, le=500),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get all cards in a collection."""
    cards = db.query(Card).filter(
        Card.collection_id == collection_id
    ).order_by(Card.order_index).offset(offset).limit(limit).all()
    return cards


@router.get("/collections/{collection_id}/meta_cards", response_model=List[MetaCardResponse])
def list_meta_cards(collection_id: int, db: Session = Depends(get_db)):
    """Get all meta cards in a collection."""
    meta_cards = db.query(MetaCard).filter(
        MetaCard.collection_id == collection_id
    ).all()
    return meta_cards


@router.get("/collections/{collection_id}/metacard", response_model=MetaCardResponse)
def get_metacard(collection_id: int, db: Session = Depends(get_db)):
    """Get the primary meta card for a collection."""
    meta_card = db.query(MetaCard).filter(
        MetaCard.collection_id == collection_id
    ).first()
    if not meta_card:
        raise HTTPException(status_code=404, detail="No meta card found for this collection")
    return meta_card


# ============================================================
# Download Endpoints
# ============================================================

@router.get("/collections/{collection_id}/download/cards.json")
def download_cards_json(collection_id: int, db: Session = Depends(get_db)):
    """Download all cards in a collection as JSON."""
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    cards = db.query(Card).filter(
        Card.collection_id == collection_id
    ).order_by(Card.order_index).all()
    
    # Build JSON array of card contents
    cards_data = [card.content for card in cards]
    
    filename = f"{collection.name.replace(' ', '_')}_cards.json"
    json_content = json.dumps(cards_data, indent=2, ensure_ascii=False)
    
    return Response(
        content=json_content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/collections/{collection_id}/download/metacard.json")
def download_metacard_json(collection_id: int, db: Session = Depends(get_db)):
    """Download the meta card of a collection as JSON."""
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    meta_card = db.query(MetaCard).filter(
        MetaCard.collection_id == collection_id
    ).first()
    
    if not meta_card:
        raise HTTPException(status_code=404, detail="No meta card found for this collection")
    
    filename = f"{collection.name.replace(' ', '_')}_metacard.json"
    json_content = json.dumps(meta_card.content, indent=2, ensure_ascii=False)
    
    return Response(
        content=json_content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/collections/{collection_id}/download/cards.pdf")
def download_cards_pdf(collection_id: int, db: Session = Depends(get_db)):
    """Download all cards in a collection as PDF."""
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    cards = db.query(Card).filter(
        Card.collection_id == collection_id
    ).order_by(Card.order_index).all()
    
    if not cards:
        raise HTTPException(status_code=404, detail="No cards found in this collection")
    
    # Import PDF generation utilities
    try:
        from cards_to_pdf import create_cards_pdf_from_data
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF generation not available")
    
    # Prepare cards data
    cards_data = [card.content for card in cards]
    
    # Generate PDF in temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    
    try:
        create_cards_pdf_from_data(cards_data, tmp_path)
        filename = f"{collection.name.replace(' ', '_')}_cards.pdf"
        return FileResponse(
            tmp_path,
            media_type="application/pdf",
            filename=filename,
            background=None  # Don't delete until response is sent
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@router.get("/collections/{collection_id}/download/metacard.pdf")
def download_metacard_pdf(collection_id: int, db: Session = Depends(get_db)):
    """Download the meta card of a collection as PDF."""
    collection = db.query(Collection).filter(Collection.id == collection_id).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    meta_card = db.query(MetaCard).filter(
        MetaCard.collection_id == collection_id
    ).first()
    
    if not meta_card:
        raise HTTPException(status_code=404, detail="No meta card found for this collection")
    
    # Import PDF generation utilities
    try:
        from cards_to_pdf import create_metacard_pdf_from_data
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF generation not available")
    
    # Generate PDF in temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    
    try:
        create_metacard_pdf_from_data(meta_card.content, tmp_path)
        filename = f"{collection.name.replace(' ', '_')}_metacard.pdf"
        return FileResponse(
            tmp_path,
            media_type="application/pdf",
            filename=filename,
            background=None
        )
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ============================================================
# Helper Functions
# ============================================================

def _template_to_response(template: Template, db: Session, user_id: Optional[int] = None) -> TemplateResponse:
    """Convert a Template model to a response with additional computed fields."""
    like_count = db.query(func.count(TemplateLike.id)).filter(
        TemplateLike.template_id == template.id
    ).scalar()
    
    is_liked = False
    if user_id:
        is_liked = db.query(TemplateLike).filter(
            TemplateLike.template_id == template.id,
            TemplateLike.user_id == user_id
        ).first() is not None
    
    owner_name = None
    if template.owner:
        owner_name = template.owner.display_name or template.owner.email or "Anonymous"
    
    return TemplateResponse(
        id=template.id,
        name=template.name,
        description=template.description,
        question=template.question,
        schema=template.schema,
        version=template.version,
        is_public=template.is_public,
        is_featured=template.is_featured,
        is_system=template.is_system,
        use_count=template.use_count,
        fork_count=template.fork_count,
        created_at=template.created_at,
        updated_at=template.updated_at,
        owner_name=owner_name,
        is_liked=is_liked,
        like_count=like_count
    )
