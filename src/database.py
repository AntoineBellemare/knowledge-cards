"""
Database connection and session management for Railway PostgreSQL.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Railway provides DATABASE_URL automatically when you link the Postgres service
DATABASE_URL = os.getenv("DATABASE_URL")

# Railway uses "postgres://" but SQLAlchemy needs "postgresql://"
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# For local development, use a fallback or raise an error
if not DATABASE_URL:
    print("⚠️  DATABASE_URL not set. Database features will be disabled.")
    engine = None
    SessionLocal = None
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    Dependency for FastAPI endpoints that need database access.
    Usage: db: Session = Depends(get_db)
    """
    if SessionLocal is None:
        raise Exception("Database not configured. Set DATABASE_URL environment variable.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables in the database.
    Call this once on startup.
    """
    if engine is not None:
        from models import Base as ModelsBase
        ModelsBase.metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
    else:
        print("⚠️  Skipping database initialization - no DATABASE_URL")
