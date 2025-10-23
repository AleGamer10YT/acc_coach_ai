from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
try:  # SQLAlchemy >=1.4
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:  # pragma: no cover - fallback for trimmed builds
    from sqlalchemy.orm import sessionmaker

    def async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession):
        return sessionmaker(bind=engine, expire_on_commit=expire_on_commit, class_=class_)

from shared.utils.config import get_settings
from shared.utils.logging import configure_logging

from .models import Base

logger = configure_logging("analytics.database")

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Migrazione schema completata")


@asynccontextmanager
async def get_session() -> AsyncSession:
    session = SessionLocal()
    try:
        yield session
    finally:
        await session.close()
