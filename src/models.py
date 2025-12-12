"""
SQLAlchemy models for Knowledge Cards database.

All tables are prefixed with 'kc_' (knowledge-cards) to avoid conflicts
with other apps sharing the same database (e.g., germinarium).

Tables:
- kc_users: User accounts (email-based or anonymous)
- kc_templates: Card generation templates/schemas
- kc_collections: Groups of cards (from a single generation run)
- kc_cards: Individual generated cards
- kc_meta_cards: Meta-level synthesis cards
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, 
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from database import Base

# Table prefix to avoid conflicts with other apps
TABLE_PREFIX = "kc_"


class KCUser(Base):
    """
    User account for Knowledge Cards app.
    Separate from germinarium users table.
    
    Recognition methods (in priority order):
    1. username - unique, user-chosen identifier (like 'antoine')
    2. email - for registered users
    3. anonymous_id - UUID stored in browser localStorage
    """
    __tablename__ = f"{TABLE_PREFIX}users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=True, index=True)  # e.g., 'antoine'
    email = Column(String(255), unique=True, nullable=True, index=True)
    display_name = Column(String(100), nullable=True)  # Friendly name shown in UI
    anonymous_id = Column(String(64), unique=True, nullable=True, index=True)  # For anonymous users
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    templates = relationship("Template", back_populates="owner", cascade="all, delete-orphan")
    collections = relationship("Collection", back_populates="owner", cascade="all, delete-orphan")


class Template(Base):
    """
    A card generation template/schema.
    Users can save their own and optionally contribute to collective library.
    """
    __tablename__ = f"{TABLE_PREFIX}templates"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}users.id"), nullable=True)  # Null = system template
    
    # Template metadata
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    question = Column(Text, nullable=True)  # The question that generated this schema
    
    # The actual schema (JSONB for efficient querying)
    schema = Column(JSON, nullable=False)
    
    # Versioning
    version = Column(Integer, default=1)
    parent_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}templates.id"), nullable=True)  # For forks/versions
    
    # Visibility
    is_public = Column(Boolean, default=False, index=True)  # Contributed to collective
    is_featured = Column(Boolean, default=False)  # Curated by admins
    is_system = Column(Boolean, default=False)  # Built-in templates
    
    # Stats
    use_count = Column(Integer, default=0)
    fork_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("KCUser", back_populates="templates")
    collections = relationship("Collection", back_populates="template")
    
    # Indexes for common queries
    __table_args__ = (
        Index(f"idx_{TABLE_PREFIX}templates_public_featured", "is_public", "is_featured"),
        Index(f"idx_{TABLE_PREFIX}templates_user_name", "user_id", "name"),
    )


class Collection(Base):
    """
    A collection of cards generated from a single run.
    Represents one "session" of card generation.
    """
    __tablename__ = f"{TABLE_PREFIX}collections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}users.id"), nullable=True)
    template_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}templates.id"), nullable=True)
    
    # Collection metadata
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    papers_folder = Column(String(200), nullable=True)  # Source folder name
    
    # Generation metadata
    model_used = Column(String(100), nullable=True)  # e.g., "gemini-2.5-flash"
    generation_params = Column(JSON, nullable=True)  # Any additional params
    
    # Visibility
    is_public = Column(Boolean, default=False, index=True)
    
    # Summary stats (denormalized for performance)
    card_count = Column(Integer, default=0)
    meta_card_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("KCUser", back_populates="collections")
    template = relationship("Template", back_populates="collections")
    cards = relationship("Card", back_populates="collection", cascade="all, delete-orphan")
    meta_cards = relationship("MetaCard", back_populates="collection", cascade="all, delete-orphan")


class Card(Base):
    """
    An individual generated knowledge card.
    """
    __tablename__ = f"{TABLE_PREFIX}cards"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}collections.id"), nullable=False)
    
    # Card content (flexible JSON structure based on template)
    content = Column(JSON, nullable=False)
    
    # Source reference
    source_file = Column(String(500), nullable=True)  # Original PDF filename
    source_page = Column(Integer, nullable=True)
    
    # Ordering within collection
    order_index = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    collection = relationship("Collection", back_populates="cards")
    
    __table_args__ = (
        Index(f"idx_{TABLE_PREFIX}cards_collection_order", "collection_id", "order_index"),
    )


class MetaCard(Base):
    """
    A meta-level synthesis card that summarizes patterns across cards.
    """
    __tablename__ = f"{TABLE_PREFIX}meta_cards"
    
    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}collections.id"), nullable=False)
    
    # Meta card content
    content = Column(JSON, nullable=False)
    
    # Type of meta-analysis
    meta_type = Column(String(100), nullable=True)  # e.g., "synthesis", "network", "timeline"
    
    # Which cards this meta card summarizes (array of card IDs)
    source_card_ids = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    collection = relationship("Collection", back_populates="meta_cards")


# ============================================================
# Collective Database Tables
# ============================================================

class TemplateContribution(Base):
    """
    Tracks when users contribute templates to the collective library.
    Separates the contribution act from the template itself.
    """
    __tablename__ = f"{TABLE_PREFIX}template_contributions"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}templates.id"), nullable=False)
    contributor_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}users.id"), nullable=False)
    
    # Contribution metadata
    contribution_note = Column(Text, nullable=True)
    
    # Moderation status
    status = Column(String(20), default="pending")  # pending, approved, rejected
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("template_id", "contributor_id", name=f"uq_{TABLE_PREFIX}template_contributor"),
    )


class TemplateLike(Base):
    """
    Users can like/favorite public templates.
    """
    __tablename__ = f"{TABLE_PREFIX}template_likes"
    
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}templates.id"), nullable=False)
    user_id = Column(Integer, ForeignKey(f"{TABLE_PREFIX}users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("template_id", "user_id", name=f"uq_{TABLE_PREFIX}template_user_like"),
    )
