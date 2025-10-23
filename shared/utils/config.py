from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class CoreSettings(BaseSettings):
    environment: str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    database_url: str = Field("sqlite+aiosqlite:///./data/app.db", env="DATABASE_URL")

    coach_api_url: str = Field("http://localhost:8082", env="COACH_API_URL")
    overlay_ws_url: str = Field("ws://localhost:8090/ws/feedback", env="OVERLAY_WS_URL")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> CoreSettings:
    # Ensure data directory exists when using default SQLite path
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        os.makedirs("data", exist_ok=True)
    return CoreSettings()
